from __future__ import annotations
from collections.abc import Callable

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import re
pd.options.mode.chained_assignment = None
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>DATA FILTER<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def findData()->pd.DataFrame:
    """
    Find the necessary data from all csv files in raw data to be used for analysis, using id files 

    Returns
    -------
    Data Frame 
        Return as a two-dimensional data structure, with each column contains all filter files as data frames from a snapshot

    """

    root = getRoot()
    config = loadConfig()

    raw_data = pd.DataFrame(
        {date: [pd.read_csv(root / "data/raw" / config["city"] / date / file) for file in config["files"]] for date in config["snapshots"]},
        index=["listings", "calendar", "reviews"]
        )

    list_id_file = pd.read_csv(root / "data/id/listing_id.csv") 
    rev_id_file = pd.read_csv(root / "data/id/review_id.csv") 

    listing_id = list_id_file[["listing_id", "analysis_area"]]
    review_id = rev_id_file["review_id"]

    print("All index data downloaded:")

    needed_data = raw_data
    print("Preparing for merging data")

    missing_txt = "\x1b[91mMissing\x1b[0m"
    right_txt = "\x1b[96mRight amount\x1b[0m"

    for date in needed_data:
        needed_data.at["listings", date] = pd.merge(needed_data.at["listings", date], listing_id, left_on="id", right_on="listing_id", validate="many_to_one").drop("listing_id", axis=1)
        needed_data.at["calendar", date] = pd.merge(needed_data.at["calendar", date], listing_id, validate="many_to_one")
        needed_data.at["reviews", date] = pd.merge(needed_data.at["reviews", date], review_id, left_on="id", right_on="review_id", validate="many_to_one").drop("review_id", axis=1)
        
        print("-"*20)
        print(f"Found all data about listings on {date}: {len(needed_data.loc['listings', date])==1000}")
        print(f"Found all data about calendar on {date}: {len(needed_data.loc['calendar', date])==365000}")
        print(f"Found all data about reviews on {date}: {len(needed_data.loc['reviews', date])==5000}\n")

        no_entire_apt = sum(needed_data.loc["listings", date]["room_type"]=="Entire home/apt")
        no_priv_room = sum(needed_data.loc["listings", date]["room_type"]=="Private room")
        no_0reviews = sum(needed_data.loc["listings", date]["number_of_reviews"]==0)

        print(f"Checking specific datas:")
        print(f"Entire home: {right_txt if no_entire_apt==875 else missing_txt}")
        print(f"Private room: {right_txt if no_priv_room==125 else missing_txt}")
        print(f"Property with no reviews: {right_txt if (((date=='2025-09-22') & (no_0reviews==50)) | ((date=='2025-12-26') & (no_0reviews==30)) | ((date=='2026-03-27') & (no_0reviews==25)) | ((date=='2026-06-26') & (no_0reviews==25))) else missing_txt}")
        if date=='2026-06-26':
            no_notpriced = sum(needed_data.loc["listings", date]["price"].isna())
            print(f"Property with no price: {right_txt if no_notpriced==20 else missing_txt}")
        
    return needed_data

#--------------------------------------------------------------------------------------------------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>PROBLEMATIC DATA HANDLER<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class dataWashingMachine:
    """
        This class only used data extracted by findData()

        Add column you want to FLAG in florence.yaml, in the neccesary_values

        After the data inside got decontaminated, the flags for missing and outlier value are stored in flags attribute, 
        and the report about the qa process is stored inside report attribute as a data frame, and can be exports as a csv file

        The four type of flags are:
            "Critical KPI value": Contaminated values that are used in KPI calculation
            "Critical LLM value": Contaminated values used in LLM evaluation
            "Non-critical value": Contaminated values that are, well, dont have much important
            "Clean": Marks for normal data
        
        The flag attribute stored a dictionary with two keys, "missing" and "outlier"

        Each keys is connected a dictionary of all of the flag table of the respected contamination, missing or outlier

        All pairs in the dictionary has the key in the following format: "(file)-(date)"
            - file is any of the following: "listings", "calendar", "reviews"
            - date is any of the four snapshots date
        
        The value of the pair is the flag data frame

        Please write the report file to csv file after the decontamination process

        Note
        ----
            This class is NOT optimized for memory, things might exploded
            It can take a minute to clean
    """
    def __init__(self, main_data: pd.DataFrame[pd.DataFrame]):
        self.data = main_data
        root = Path(__file__).resolve().parents[1]
        with open(root/"configs/florence.yaml", "r") as config:
            important_value = yaml.safe_load(config)["neccesary_values"]
        self.report = Report()
        self.__important_for_kpi = important_value["kpi"]
        self.__important_for_llm = important_value["llm"]
        self.flags = {"missing": dict(), "outlier": dict()}

    #Ideas for flagging: 1)Add an aditional column to the orginal table, 2) Create another table
    def cleanData(self)->None:
        """
        Check all of data for contamination

        Flags missing and outlier value, delete dupes

        Reformats values if possible otherwise deletes them, force data types on some values for easier calculation

        Note
        ----
        The magnum opus of this class
        """
        for snapshot in self.data:
            for file in ["listings", "calendar", "reviews"]:
                cur_table = self.data.loc[file, snapshot]
                flag_name = "-".join([file, snapshot])

                cur_table, changed_values = self.__invalidFormatHandler(cur_table)
                cur_table = self.__duplicateHandler(cur_table)
                self.flags["missing"][flag_name] = self.__missingFlagger(file, cur_table)
                self.flags["outlier"][flag_name] = self.__outlierFlagger(file, cur_table)

                self.data.at[file, snapshot] = cur_table

                stats = pd.Series([0]*7, index=self.report.content.columns.drop(["total_no._missing_values", "total_no._outlier_values", "no._flagged_values"])) #Spaghetti codes
                miss_flag = self.flags["missing"]["-".join([file, snapshot])]
                out_flag = self.flags["outlier"]["-".join([file, snapshot])]
                stats["no._changed_values"]=changed_values
                flag_type = ["KPI", "LLM", "Non"]
                for col in miss_flag:
                    no_contamination = sum(miss_flag[col]!="Clean")
                    if no_contamination > 0:
                        contamination_type = miss_flag[col][miss_flag[col]!="Clean"].iloc[0]
                        stats.iloc[flag_type.index(re.findall("KPI|LLM|Non", contamination_type)[0])]+=no_contamination
                for col in out_flag:
                    no_contamination = sum(out_flag[col]!="Clean")
                    if no_contamination > 0:
                        contamination_type = out_flag[col][out_flag[col]!="Clean"].iloc[0]
                        stats.iloc[flag_type.index(re.findall("KPI|LLM|Non", contamination_type)[0])+3]+=no_contamination
                print(f"Clean {flag_name}")
                self.report.content = self.report.writeToReport(file, snapshot, stats)  

    def __baseFlagger(self, name: str, data: pd.DataFrame, cond: Callable[[pd.Series], pd.Series[bool]] | dict[str, Callable[[pd.Series], pd.Series[bool]]])->pd.DataFrame:
        """
        A base function to handle different type of data contamination depending on conditions

        The return is a flagged table of the original table

        Parameters
        ----------
            name: str
                Name of the data in needed of flagging(listings, reviews, calendar)
            data: Data Frame
                The data in question
            cond: A function or a dict of functions
                A function that take in a series and applying conditions for marking special datas on it, then return a boolean series

                If you want to use different marking conditions for some special columns, enter a dictionary with the name of the columns as keys, and the functions as values

        Returns
        --------
            Data Frame
                A board contains all the flags 
        Example
        --------
        >>> table=pd.DataFrames{'Price': [10.0, 55.0, 35.0], 'Quantity': [20, 4, 15], index=['Fan', 'Tv', 'Ac']}
            Price   Quantity
        Fan  10.0         20
        Tv   55.0          4 
        Ac   35.0         15

        We can create a function to filter out items with non negative value and pluck it in like this:
        >>> def nonNegative(series):
            return series>0
        >>> baseHandler('tableName', table, nonNegative)
            Price   Quantity
        Fan Clean      Clean
        Tv  Clean      Clean 
        Ac  Clean      Clean
        """
        flag = data.copy()
        number_of_rows=len(data)
        default = [True]*number_of_rows
        print("Flagging")
        if callable(cond):
            for col in flag:
                print(f"flasgging {col}, dtype={flag[col].dtype}")
                if col in self.__important_for_kpi[name]:
                    flag[col]=flag[col].case_when([(cond(flag[col]), "Critical KPI value"), (default, "Clean")])
                elif col in self.__important_for_llm[name]:
                    flag[col]=flag[col].case_when([(cond(flag[col]), "Critical LLM value"), (default, "Clean")])
                else:
                    flag[col]=flag[col].case_when([(cond(flag[col]), "Non-critical value"), (default, "Clean")])
        else:
            for col in flag:
                print(f"flagging {col}, dtpe={flag[col].dtype}")
                spec_cond = col if col in cond else "def"
                if col in self.__important_for_kpi[name]:
                    flag[col]=flag[col].case_when([(cond[spec_cond](flag[col]), "Critical KPI value"), (default, "Clean")])
                elif col in self.__important_for_llm[name]:
                    flag[col]=flag[col].case_when([(cond[spec_cond](flag[col]), "Critical LLM value"), (default, "Clean")])
                else:
                    flag[col]=flag[col].case_when([(cond[spec_cond](flag[col]), "Non-critical value"), (default, "Clean")])
        print("Done flagging")
        return flag

    def __missingFlagger(self, name: str, data: pd.DataFrame)->pd.DataFrame:
        """
        Find missing data and flag them

        Parameters
        ----------
            name: str
                Name of the data in needed of flagging(listings, reviews, calendar)
            data: data frame
                The data in question
    
        Returns
        -------
            Data Frame
                A board contains all the flags

        """
        return self.__baseFlagger(name, data, lambda series: series.isna())


    def __duplicateHandler(self, data: pd.DataFrame)->pd.DataFrame:
        """
        Find all duplicate and collapse them into the first occurence

        Parameters
        ----------
            data: data frame
                The data to dedupe

        Returns
        -------
            Data frame
                The deduped data
        """
        return data.drop_duplicates(keep='first')

    def __outlierFlagger(self, name: str, data: pd.DataFrame)->pd.DataFrame:
        """
        Find outlier data then flag them, detecting by IQR method and pre-determined threshold set in config 

        Parameters
        ----------
            name: str
                Name of the data in needed of flagging(listings, reviews, calendar)
            data: data frame
                The data in question
    
        Returns
        -------
            Data Frame
                A board contains all the flags
        """
        temp = data.copy()
        threshold = loadConfig("force_outlier_threshold")
        outlier = dict()
        for item in threshold:
            outlier[item] = lambda s: self.__outlierMark(s)|(s<threshold[item][0])|(s>threshold[item][1])
        outlier["def"]= self.__outlierMark
        temp = temp.drop([col for col in data.columns if (data[col].dtype=="object" or data[col].dtype=="str" or data[col].dtype=="bool") or re.findall("id", col)], axis=1)
        print(f"Col to check for outlier: {list(temp.columns)}")
        return self.__baseFlagger(name, temp, outlier)

    def __invalidFormatHandler(self, data: pd.DataFrame)->tuple[pd.DataFrame, int]:
        """
        Date-related datas are checked and changed to valid format if possible, otherwise delete them

        Price, rate are forced into float 

        Parameters
        ----------
            data: data frame
                The table for filtering
    
        Returns
        -------
            data frame and int
                A tuple contains the filtered data frame and the number of values that got its content changed

        """
        print("Formatting")
        values_changed=0
        for col in data:
            print(f"working on {col}")
            if col=="price" or re.findall("rate", col):
                if data[col].first_valid_index()==None:
                    continue
                if type(data[col].iloc[data[col].first_valid_index()])==str: 
                    data[col]=data[col].str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
                data[col]=data[col].astype("Float64").abs()
                if re.findall("rate", col): data[col]/100
            elif re.findall("since|first|last|date", col):
                filter_col = self.__checkingInvalidDate(data[col])
                data[col] = filter_col[0]
                values_changed += filter_col[1]
        print("Done formatting")
        return data, values_changed

    def __outlierMark(self, s: pd.Series) -> pd.Series[bool]:
        """
        Detecting all the outlier in the series using the IQR method

        Parameters
        ----------
            s: Pandas series
                The input series for filtering outlier
        
        Returns
        -------
            Pandas boolean series
                The mask for the input series
        """

        first_quar = s[s<=s.median()].median()
        third_quar = s[s>=s.median()].median()
        iqr = third_quar - first_quar
        return (s<(first_quar-1.5*iqr)) | (s>(third_quar+1.5*iqr))

    def __checkingInvalidDate(self, s: pd.Series):
        original = s.copy()

        digits = s.astype("string").str.replace(r"\D", "", regex=True)

        parsed = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")

        iso = parsed.dt.strftime("%Y-%m-%d")
        was_present = original.notna()
        changed = was_present & (parsed.isna() | (original.astype("string") != iso))
        values_changed = int(changed.sum())

        return parsed, values_changed

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>HELPER CLASS & FUNCTION<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class Report:
    def __init__(self):
        self.content = pd.DataFrame(columns=["no._missing_value_for_kpi_calculation", 
                                        "no._missing_value_for_llm_evaluation", 
                                        "no._missing_value_not_affecting_anything", 
                                        "no._outlier_value_for_kpi_calculation", 
                                        "no._outlier_value_for_llm_evaluation", 
                                        "no._outlier_value_not_affecting_anything",
                                        "total_no._missing_values",
                                        "total_no._outlier_values", 
                                        "no._changed_values", 
                                        "no._flagged_values"])
    
    def writeToReport(self, data_name: str, data_date: str, stats: pd.Series[int])->pd.DataFrame:
        """
        Write the statistics of a cleaning session onto the report table

        Insert a new row if the file stat doesnt exist, otherwise write onto the pre-existing row

        Parameters
        ----------
            data_name: str
                Name of the file
            data_date: str
                The date of the file
            stats: series of int
                The statistics of the cleaning session
        
        Returns
        -------
            data frame
                The report table

        """
        row="-".join([data_name, data_date])
        if row not in self.content.index:
            report = pd.concat([self.content, pd.DataFrame(columns=self.content.columns, index=[row])])
            report.loc[row, :]=0
        else:
            report = self.content
        for column in stats.index:
            report.at[row, column] += stats[column] 
        report.at[row, "total_no._missing_values"]=report.loc[row, ["no._missing_value_for_kpi_calculation", "no._missing_value_for_llm_evaluation", "no._missing_value_not_affecting_anything"]].sum()
        report.at[row, "total_no._outlier_values"]=report.loc[row, ["no._outlier_value_for_kpi_calculation", "no._outlier_value_for_llm_evaluation", "no._outlier_value_not_affecting_anything"]].sum()
        report.at[row, "no._flagged_values"]=report.loc[row, ["total_no._missing_values", "total_no._outlier_values"]].sum()
        return report

def saveData(container: dataWashingMachine):
    """
    Export all clean contents, flags and report as csv files

    Parameters
    ----------
        container: Washing machine
            The cleaner to export data

    Notes
    -----
    *For clean contents and flags, all files are exported to processed folder*

    The directory for each file is (root)/data/processed/(branch)/(date)/(name)
        -root: The main project folder file
        -branch: clean for clean data, missing|outlier for each respective contamination type
        -date: The format is (YYYY)/(MM)/(DD)
        -name: listings, calendar, reviews
    """
    decontaminated_data = container.data
    flag = container.flags
    report = container.report.content
    root = Path(__file__).resolve().parents[1]
    for date in decontaminated_data:
        Path(root / "data/processed" / "clean" / date).mkdir(parents=True, exist_ok=True)
        for file in ["listings", "calendar", "reviews"]:
            decontaminated_data.loc[file, date].to_csv(root / "data/processed" / "clean" / date / f"{file}.csv")

    for type in flag:
        for file in flag[type]:
            file_name = file.split("-")
            name, date = file_name[0], "-".join(file_name[1:])
            Path(root / "data/processed" / type / date).mkdir(parents=True, exist_ok=True)
            flag[type][file].to_csv(root / "data/processed" / type / date / f"{name}.csv")
            print(flag[type][file])

    report.to_csv(root / "reports/qa_report.csv" )

def loadConfig(value: str =None)->dict:
    """
    Get the config yaml file and return it
    """
    root = getRoot()

    with open(root / "configs" / "florence.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config if not value else config[value]

def getRoot()->Path:
    """
    Return project root's path
    """
    return Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    config = loadConfig()
    if not config["city"] or not config["snapshots"]:
        pass
    else:
        raw_data = findData()
        print("Data found")
        cleaner = dataWashingMachine(raw_data)
        cleaner.cleanData()
        print("Data cleansed")
        print("---REPORT---")
        print(cleaner.report.content)
        print("---MISSING---")
        for flag in cleaner.flags["missing"]:
            name = flag.split("-")
            print(name[0], end=", ")
            print("/".join(name[1:]))
            print(cleaner.flags["missing"][flag])
        print("---OTULIER---")
        for flag in cleaner.flags["outlier"]:
            name = flag.split("-")
            print(name[0], end=", ")
            print("/".join(name[1:]))
            print(cleaner.flags["outlier"][flag])

    

