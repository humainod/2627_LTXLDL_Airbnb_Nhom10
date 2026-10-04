AI: Github Copilot
Usage: Debugging on qa.py, findData()
Detail: Failed to assign dataframe to each cells of needed_data, returning empty cell. Solution: chaning from .loc to .at so that it forces each cell to be assign a dataframe.