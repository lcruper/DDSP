# Libraries -----------------------------------------------------------------------------------------------------¡
from Exact import *
import os

def run(folder_files: str, folder_results: str, t: int, multi: bool = True) -> None:
    files = sorted([file for file in os.listdir(folder_files) if file.startswith("graph_") & file.endswith(".txt")])
    for file in files :
        print(f"RESOLVING: {file}")
        path_to_file = folder_files + "/" + file
        path_to_save_csv = folder_results + "/exact_" + file[:-3] + "csv"
        biobjective(path_to_file, path_to_save_csv, t, multi)
        print(f"FINISHED: {file}")
        print("---------------------------------------------")

if __name__ == "__main__":
    folder_instances = "../instances/"
    folder_results = "../results/"
    timelimit = 1800
    multiobjective = True #If False, it solves the Minimum Dominating Set Problem (distances are not taken into account)

    run(folder_instances, folder_results, timelimit, multiobjective)