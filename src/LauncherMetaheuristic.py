# Libraries -----------------------------------------------------------------------------------------------------
import argparse

from Metaheuristic import *
import os

# Random seed ----------------------------------------------------------------------------------------------------
random.seed(7)

def run(method: str, folder_files: str, folder_results: str, N: int, multi: bool = True) -> None:
    files = sorted([file for file in os.listdir(folder_files) if file.startswith("graph_") & file.endswith(".txt")])
    for file in files :
        print(f"RESOLVING: {file}")
        path_to_file = folder_files + "/" + file
        path_to_save_csv = folder_results + "/" + method + "_" + str(N) + "_" + file[:-3] + "csv"
        biobjective(method, path_to_file, path_to_save_csv, N, multi)
        print(f"FINISHED: {file}")
        print("---------------------------------------------")

# Main -----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    method = "GRA" #GRA, RGA, IRGA
    folder_instances = "../instances/"
    folder_results = "../results/"
    N = 100 # Number of iterations of the GRASP/RGASP/IRGASP
    multiobjective = True #If False, it solves the Minimum Dominating Set Problem (distances are not taken into account)

    run(method, folder_instances, folder_results, N, multiobjective)