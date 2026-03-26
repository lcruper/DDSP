# Libraries -----------------------------------------------------------------------------------------------------
import time
from typing import Dict, Set, Tuple, List, Any
import random 
import math

import numpy as np
import pandas as pd

# Auxiliary Functions --------------------------------------------------------------------------------------------

def random_01() -> float:
    num = random.uniform(0,1)
    while num == 0 or num == 1:
        num = random.uniform(0,1)
    return num

# ---------------------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------------------

def to_adj_dict(dist_matrix: np.ndarray) -> Dict[int, Tuple[Set[int], int, float]]:
    
    def neighbors(dist_matrix, node: int) -> Tuple[Set[int], int, float]:
        mask = dist_matrix[node] > 0
        ls_neighbors = set(np.nonzero(mask)[0]) | {node} 
        return ls_neighbors, len(ls_neighbors), dist_matrix[node].max()

    return {node: neighbors(dist_matrix, node) for node in range(dist_matrix.shape[0])}

def get_special_nodes(adj_dict: Dict[int, Tuple[Set[int], int, float]]) -> Tuple[Set[int], Set[int], Set[int]] :
    support_nodes = set()
    leaf_nodes = set()
    aisle_nodes = set()
    for key, (neighbors, n, _) in adj_dict.items():
        if n == 1 :
            aisle_nodes.add(key)
        elif n == 2:
            support_nodes |= neighbors - {key}
            leaf_nodes.add(key)
    return support_nodes, leaf_nodes, aisle_nodes

# ---------------------------------------------------------------------------------------------------------------

def domino_degree(adj_dict: Dict[int, Tuple[Set[int], int, float]], s: Set[int], node: int) -> int :
    return len(adj_dict[node][0] & s) 

def feasibility_check(adj_dict: Dict[int, Tuple[Set[int], int, float]], s: Set[int]) -> bool:
    if not s:
        return False
    n = len(adj_dict)
    if len(s) == n: 
        return True
    ds = set()
    for node in s:
        ds |= adj_dict[node][0]
        if len(ds) == n: 
            return True
    return False

def redundancy_check(adj_dict: Dict[int, Tuple[Set[int], int, float]], s: Set[int], special_nodes: Tuple[Set[int], Set[int], Set[int]]) -> Set[int]:
    if not s:
        return set()
    reduced_set = s.copy()
    for node in (s - (special_nodes[0] | special_nodes[2])): 
        new_dominating_set = reduced_set - {node}
        if feasibility_check(adj_dict, new_dominating_set):
            return redundancy_check(adj_dict, new_dominating_set, special_nodes) 
    return reduced_set

def get_f2(dist_matrix: np.ndarray, s: Set[int]) -> float:
    f2 = 0
    n = len(dist_matrix)
    if len(s) < n:
        x_indices = set(range(n))
        not_s = x_indices - s
        aux = dist_matrix[np.ix_(list(s), list(not_s))]
        f2 = np.max(np.min(np.where(aux==0, np.inf, aux), axis=0))
    return f2

# Heuristic ----------------------------------------------------------------------------------------------------------

def get_initial(adj_dict: Dict[int, Tuple[Set[int], int, float]], special_nodes: Tuple[Set[int], Set[int], Set[int]], sort: bool=False) \
        -> Tuple[Set[int], List[int], Set[int], Set[int]]:
    s_initial = special_nodes[0] | special_nodes[2]
    all_nodes = set(range(len(adj_dict)))
    CL = all_nodes - (s_initial | special_nodes[1])
    if sort:
        CL = sorted(CL, key = lambda node : (adj_dict[node][1], -adj_dict[node][2]), reverse=True)
    if not s_initial:
        ds = set()
    else:
        ds = set.union(*(adj_dict[i][0] for i in s_initial))
    nds = all_nodes - ds
    return s_initial, CL, ds, nds

# Constructive

def node_to_insert_constructive(method_constructive: str, alpha: float, beta: float, adj_dict: Dict[int, Tuple[Set[int], int, float]], CL: List[int], nds: Set[int])\
        -> int:

    def greedy(adj_dict: Dict[int, Tuple[Set[int], int, float]], node: int, nds: Set[int]) -> int:
        return len(adj_dict[node][0] & nds) 
    
    if method_constructive == "GRA" :
        CL_copy = set()
        gmin = 0x4f4f4f4f
        gmax = -0x4f4f4f4f
        for node in CL:
            gr = greedy(adj_dict, node, nds)
            CL_copy.add((node, gr))
            if gr > gmax:
                gmax = gr
            if gr < gmin:
                gmin = gr
        threshold = gmax - alpha*(gmax-gmin) 
        CL_copy = sorted(CL_copy, key = lambda node : node[1], reverse=True)
        RCL = []
        for node in CL_copy :
            if node[1] < threshold : 
                break
            RCL.append(node[0])
        return random.choice(RCL)
    
    elif method_constructive == "RGA" :
        RCL = random.sample(tuple(CL), math.ceil(beta*len(CL)))
        return max(RCL, key = lambda node : (greedy(adj_dict, node, nds), -adj_dict[node][2]))  
    
    elif method_constructive == "IRGA":
        random_number = random.randint(1, len(CL))
        RCL = CL[:random_number]
        threshold = adj_dict[RCL[-1]][1]
        for node in CL[random_number:] :
            if adj_dict[node][1] == threshold :
                RCL.append(node)
            else:
                break
        random.shuffle(RCL)
        return max(RCL, key = lambda node : (greedy(adj_dict, node, nds), -adj_dict[node][2]))

def constructive(method_constructive: str, alpha: float, beta: float, adj_dict: Dict[int, Tuple[Set[int], int, float]], initial: Tuple[Set[int], List[int], Set[int], Set[int]])\
    -> Tuple[int, Set[int]] :
    s = initial[0].copy()
    CL = initial[1].copy()
    nds = initial[3].copy()
    while len(nds) > 0 :
        if method_constructive == "GRA" and alpha == -1 :
            alpha = random_01()
        elif method_constructive == "RGA" and beta == -1 :
            beta = random_01()
        node = node_to_insert_constructive(method_constructive, alpha, beta, adj_dict, CL, nds)
        s.add(node)
        CL.remove(node)  
        nds -= adj_dict[node][0]
    return len(s), s

# Local search

def node_to_remove_local_search(ls_nodes_remove: Set[int]) -> int :
    return random.choice(tuple(ls_nodes_remove))
    
def node_to_insert_local_search(adj_dict: Dict[int, Tuple[Set[int], int, float]], s: Set[int], ls_nodes_insert: Set[int], nodes_remove: Set[int]) \
        -> Tuple[int, Any] :
    nds = {node for node in set.union(*(adj_dict[i][0] for i in nodes_remove)) if domino_degree(adj_dict, s, node) == 0}
    candidates = set.intersection(*(adj_dict[i][0] for i in nds), ls_nodes_insert)
    if not candidates :
        return -1, -1      
    node = random.choice(tuple(candidates))
    new_s = s | {node}
    return node, new_s

def local_search(cond: int, adj_dict: Dict[int, Tuple[Set[int], int, float]], special_nodes: Tuple[Set[int], Set[int], Set[int]], s: Set[int], redundancy: bool=True) \
        -> Tuple[int, Set[int]]:
    if redundancy:
        s = redundancy_check(adj_dict, s, special_nodes)
    i = 0
    support_nodes, leaf_nodes, aisle_nodes = special_nodes
    all_nodes = set(range(len(adj_dict)))
    nodes_remove =  s - (support_nodes | aisle_nodes) 
    nodes_insert = all_nodes - (s | leaf_nodes) 
    if not nodes_insert:
        return len(s), s
    while i < cond :
        if len(nodes_remove) < 2:
            break
        node1 = node_to_remove_local_search(nodes_remove)
        node2 = node_to_remove_local_search(nodes_remove - {node1})
        node_insert, new_s = node_to_insert_local_search(adj_dict, s - {node1, node2}, nodes_insert, {node1, node2}) 
        if node_insert == -1 :
            i += 1
            continue
        s = new_s
        nodes_remove -= {node1, node2}
        nodes_remove.add(node_insert)
        nodes_insert.remove(node_insert)
        nodes_insert |= {node1, node2}
        i = 0
    s = redundancy_check(adj_dict, s, special_nodes)
    return len(s), s

# Algorithm

def monoobjective(method: str, path_to_file: str, epsilon: float, N: int, alpha: float, beta: float) \
    -> Tuple[int, float, float]:
    dist_matrix = prune(read_matrix(path_to_file), epsilon)
    adj_dict = to_adj_dict(dist_matrix)
    special_nodes = get_special_nodes(adj_dict)

    start = time.perf_counter()
    if method == "IRGA":
        initial = get_initial(adj_dict, special_nodes, True)
    else:
        initial = get_initial(adj_dict, special_nodes, False)

    initial_solutions = []
    for _ in range(N):
        _, initial_solution_s  = constructive(method, alpha, beta, adj_dict, initial)
        initial_solutions.append(initial_solution_s)
    solutions = []
    for i in range(N):
        solution_OF1, solution_s = local_search(150, adj_dict, special_nodes, initial_solutions[i])
        solution_OF2 = get_f2(dist_matrix, solution_s)
        solutions.append((solution_OF1, solution_OF2))
    end = time.perf_counter()
    best_solution = min(solutions, key = lambda x : (x[0], x[1]))
    OF1 = best_solution[0]
    OF2 = best_solution[1]
    t = end-start

    return OF1, OF2, t

def biobjective(method: str, path_to_file: str, path_to_save_csv: str, N: int, alpha: float, beta: float, multi: bool) -> None:
    D = read_matrix(path_to_file)
    results = {"OF1": [], "OF2": [], "Time (s)": []}

    ls_epsilon = sorted(set(D.flatten()), reverse=True)[:-1]
    if not multi:
        ls_epsilon = [ls_epsilon[0]]

    for epsilon in ls_epsilon:
        print(f"epsilon: {epsilon}")
        OF1, OF2, t = monoobjective(method, path_to_file, epsilon, N, alpha, beta)
        results["OF1"].append(OF1)
        results["OF2"].append(OF2)
        results["Time (s)"].append(np.round(t,2))
    df = pd.DataFrame(results)
    df.to_csv(path_to_save_csv, index=False, sep=";", decimal=",")







