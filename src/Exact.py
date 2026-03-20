# Libraries -----------------------------------------------------------------------------------------------------
from typing import Tuple

from gurobipy import *
import numpy as np
import pandas as pd
import time

# Auxiliary Functions --------------------------------------------------------------------------------------------

def read_matrix(path_to_file: str) -> np.ndarray:
    M = np.genfromtxt(path_to_file, delimiter=None)
    i_lower = np.tril_indices_from(M, -1)
    M[i_lower] = M.T[i_lower]
    return M

def prune(dist_matrix: np.ndarray, epsilon: float) -> np.ndarray:
    if epsilon == 0:
        return np.zeros_like(dist_matrix)
    if epsilon >= np.max(dist_matrix):
        return dist_matrix.copy()
    pruned_matrix = np.zeros_like(dist_matrix)
    mask = (dist_matrix > 0) & (dist_matrix <= epsilon)
    pruned_matrix[mask] = dist_matrix[mask]
    return pruned_matrix

# Exact -----------------------------------------------------------------------------------------------------------

def monobjective(path_to_file: str, epsilon: float, t: int) -> Tuple[int, float, float, float]:
    D = read_matrix(path_to_file)
    start = time.perf_counter()
    A = prune(D, epsilon)
    n = len(A)

    # Create the optimization model
    model = Model()
    model.Params.OutputFlag = 0 
    model.Params.TimeLimit = t

    x = model.addMVar(n, vtype=GRB.BINARY)
    model.setObjective(np.ones(n) @ x, GRB.MINIMIZE)
    model.addConstr(A @ x >= 1 - x)
    model.optimize()

    f1 = int(model.ObjVal)
    gap = round(model.MIPGap,6)
    s = {i for i in range(n) if x.X[i] > 0.5}
    f2 = 0
    if f1 < n:
        x_indices = set(range(n))
        not_s = set(x_indices) - s
        aux = D[np.ix_(list(s), list(not_s))]
        f2 = np.max(np.min(np.where(aux==0, np.inf, aux), axis=0))

    end = time.perf_counter()
    t = round(end - start, 6)

    model.dispose()
    
    return f1, f2, t, gap

def biobjective(path_to_file: str, path_to_save_csv: str, t: int, multi: bool = True) -> None:
    D = read_matrix(path_to_file)
    results = {"OF1": [], "OF2": [], "Time (s)": [],  "Gap": []}

    ls_epsilon = sorted(set(D.flatten()), reverse=True)[:-1]
    if not multi:
        ls_epsilon = [ls_epsilon[0]]

    for epsilon in ls_epsilon:
        print(f"epsilon: {epsilon}")
        solution = monobjective(path_to_file, epsilon, t)
        results["OF1"].append(solution[0])
        results["OF2"].append(solution[1])
        results["Time (s)"].append(round(solution[2], 2))
        results["Gap"].append(round(solution[3], 2))

    df = pd.DataFrame(results)
    df.to_csv(path_to_save_csv, index=False, sep=";", decimal=",")


