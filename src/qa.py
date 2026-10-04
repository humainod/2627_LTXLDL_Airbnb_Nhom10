import yaml
import pandas as pd
import numpy as np

"""
To do:
1) (Done) Create a data locator using id files(maybe it's better to break it up into different functions)
2) Create a set of data filter for each QA criterias
    Should be checking for:
    +) Missing values
    +) Duplicate values
    +) Data types
    +) Outlier
    +) what ever schema drift is bruh
"""
def findData()->pd.DataFrame:

    """
    Find the necessary data from all csv files in raw data to be used for analysis, using id files\n
    Then, check if the amount of data is right\n 
    Might not work depend on file directory

    Returns: 
        pd.DataFrames: Return as a two-dimensional data structure, with each column contains all files from a snapshot 
    """

    with open("configs/florence.yaml", "r") as file:
        config = yaml.safe_load(file)

    raw_data = pd.DataFrame(
        {date: [pd.read_csv(f"data/raw/florence/{date}/{file}") for file in config["files"]] for date in config["snapshots"]},
        index=["listing", "calendar", "reviews"]
        )

    list_id_file = pd.read_csv("data/id/listing_id.csv") 
    rev_id_file = pd.read_csv("data/id/review_id.csv") 

    listing_id = list_id_file[["listing_id", "analysis_area"]]
    review_id = rev_id_file["review_id"]

    print("All index data downloaded:")

    needed_data = raw_data
    print("Preparing for merging data")

    missing_txt = "\x1b[91mMissing\x1b[0m"
    right_txt = "\x1b[96mRight amount\x1b[0m"

    for date in needed_data:
        needed_data.at["listing", date] = pd.merge(needed_data.at["listing", date], listing_id, left_on="id", right_on="listing_id", validate="many_to_one").drop("listing_id", axis=1)
        needed_data.at["calendar", date] = pd.merge(needed_data.at["calendar", date], listing_id, validate="many_to_one")
        needed_data.at["reviews", date] = pd.merge(needed_data.at["reviews", date], review_id, left_on="id", right_on="review_id", validate="many_to_one").drop("review_id", axis=1)
        
        print("-"*20)
        print(f"Found all data about listing on {date}: {len(needed_data.loc['listing', date])==1000}")
        print(f"Found all data about calendar on {date}: {len(needed_data.loc['calendar', date])==365000}")
        print(f"Found all data about reviews on {date}: {len(needed_data.loc['reviews', date])==5000}\n")

        no_entire_apt = sum(needed_data.loc["listing", date]["room_type"]=="Entire home/apt")
        no_priv_room = sum(needed_data.loc["listing", date]["room_type"]=="Private room")
        no_0reviews = sum(needed_data.loc["listing", date]["number_of_reviews"]==0)

        print(f"Checking specific datas:")
        print(f"Entire home: {right_txt if no_entire_apt==875 else missing_txt}")
        print(f"Private room: {right_txt if no_priv_room==125 else missing_txt}")
        print(f"Property with no reviews: {right_txt if (((date=='2025-09-22') & (no_0reviews==50)) | ((date=='2025-12-26') & (no_0reviews==30)) | ((date=='2026-03-27') & (no_0reviews==25)) | ((date=='2026-06-26') & (no_0reviews==25))) else missing_txt}")
        if date=='2026-06-26':
            no_notpriced = sum(needed_data.loc["listing", date]["price"].isna())
            print(f"Property with no price: {right_txt if no_notpriced==20 else missing_txt}")
        
    return needed_data

def checkMissing()

