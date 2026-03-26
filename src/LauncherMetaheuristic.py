# Libraries -----------------------------------------------------------------------------------------------------
from Metaheuristic import *
import os, re

from results import printResults

# Random seed ----------------------------------------------------------------------------------------------------
random.seed(7)

def run(method: str, folder_files: str, folder_results: str, N: int, alpha: float, beta: float, multi: bool = True) -> None:
    def extract_number(file):
        matches = re.findall(r'(\d+)', file)
        return tuple(map(int, matches)) if matches else (float('inf'),)

    files = sorted([file for file in os.listdir(folder_files) if file.startswith("graph_") & file.endswith(".txt")], key=extract_number)
    for file in files :
        print(f"RESOLVING: {file}")
        path_to_file = folder_files + "/" + file
        if method == "GRA":
            value = alpha
            if alpha == -1:
                value = "rnd"
            path_to_save_csv = folder_results + "/" + method + "_" +  str(N) + "_" + str(value)  + "_" + file[:-3] + "csv"
        elif method == "RGA":
            value = beta
            if beta == -1:
                value = "rnd"
            path_to_save_csv = folder_results + "/" + method + "_" +  str(N) + "_" + str(value) + "_" + file[:-3] + "csv"
        elif method == "IRGA":
            path_to_save_csv = folder_results + "/" + method + "_" +  str(N) + "_" + file[:-3] + "csv"
        biobjective(method, path_to_file, path_to_save_csv, N, alpha, beta, multi)
        print(f"FINISHED: {file}")
        print("---------------------------------------------")

# Main -----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    method = "IRGA" #GRA, RGA, IRGA
    alpha = -1
    beta = -1
    folder_instances = "../instances"
    folder_results = "../results"
    N = 100 # Number of iterations of the GRASP/RGASP/IRGASP
    multiobjective = True #If False, it solves the Minimum Dominating Set Problem (distances are not taken into account)

    run(method, folder_instances, folder_results, N, alpha, beta, multiobjective)
