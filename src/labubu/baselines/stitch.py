import numpy as np
from collections import defaultdict
from itertools import product
import pandas as pd
import time

def bipartite_max_mapping(P, P_bar):

    def compute_weight(node_a, node_b):
        """
        Compute the weight between two nodes based on the number of common elements.
        """
        return len(node_a & node_b)

    def build_weight_matrix(P, P_bar):
        """
        Build the weight matrix for the bipartite graph.
        """
        n, m = (len(P), len(P_bar))
        weight_matrix = [[0] * m for _ in range(n)]
        for i, node_p in enumerate(P.values()):
            for j, node_p_bar in enumerate(P_bar.values()):
                weight_matrix[i][j] = compute_weight(set(node_p), set(node_p_bar))
        return weight_matrix

    def max_weight_matching(P, P_bar):
        """
        Find the maximum weight matching for the bipartite graph.
        """
        from scipy.optimize import linear_sum_assignment
        weight_matrix = build_weight_matrix(P, P_bar)
        row_ind, col_ind = linear_sum_assignment(weight_matrix, maximize=True)
        matching = [(i, j, weight_matrix[i][j]) for i, j in zip(row_ind, col_ind)]
        return matching

    def update_p_bar(P, P_bar, matching):
        """
        Update P_bar keys based on the matching results.
        """
        updated_p_bar = {}
        P_keys = list(P.keys())
        P_bar_keys = list(P_bar.keys())
        for i, j, _ in matching:
            updated_p_bar[P_keys[i]] = P_bar[P_bar_keys[j]]
        return updated_p_bar
    result = max_weight_matching(P, P_bar)
    updated_p_bar = update_p_bar(P, P_bar, result)
    return updated_p_bar

def find_connected_components(qubits, pairs):
    """
    根据pairs构建图，找到量子比特的连通分量。
    pairs为[(q1, q2), (q3, q4), ...]
    """
    adjacency = {q: set() for q in qubits}
    for q1, q2 in pairs:
        adjacency[q1].add(q2)
        adjacency[q2].add(q1)
    visited = set()
    components = []
    for q in qubits:
        if q not in visited:
            stack = [q]
            comp = []
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    comp.append(node)
                    for neigh in adjacency[node]:
                        if neigh not in visited:
                            stack.append(neigh)
            components.append(comp)
    return components

def first_fit_partition(qubits, pairs, num_qpus, max_qubits_per_qpu):
    """
    首先将由pairs定义的连通分量作为整体分配给QPU。
    然后将剩余的独立量子比特分配到QPU中。
    """
    qubits = list(qubits)
    components = find_connected_components(qubits, pairs)
    bins = defaultdict(list)
    for comp in components:
        assigned = False
        for qpu, bin_qubits in bins.items():
            if len(bin_qubits) + len(comp) <= max_qubits_per_qpu:
                bin_qubits.extend(comp)
                assigned = True
                break
        if not assigned:
            if len(bins) < num_qpus and len(comp) <= max_qubits_per_qpu:
                bins[len(bins)] = comp[:]
            else:
                return None
    remaining_qubits = set(qubits) - {q for bin_qubits in bins.values() for q in bin_qubits}
    for q in remaining_qubits:
        assigned = False
        for qpu, bin_qubits in bins.items():
            if len(bin_qubits) < max_qubits_per_qpu:
                bin_qubits.append(q)
                assigned = True
                break
        if not assigned:
            if len(bins) < num_qpus:
                bins[len(bins)] = [q]
            else:
                print(f'Unable to assign qubit {q} to any QPU. Current bins: {bins}')
                return None
    return bins

def find_qubit_partition(circuit, num_qpus, max_qubits_per_qpu):
    """
    对每一层尝试寻找一个有效的分区方案。
    确保每个量子比特都被分配到某个 QPU。
    """
    L, Q = circuit.shape
    partitions = {}
    for start in range(L):
        for end in range(start, L):
            qubits = set(range(Q))
            pairs = []
            seen_pairs = set()
            for layer in range(start, end + 1):
                for q in range(Q):
                    target = circuit[layer, q]
                    if 0 <= target < Q and target != q:
                        pair = tuple(sorted((q, target)))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            pairs.append(pair)
            partition = first_fit_partition(qubits, pairs, num_qpus, max_qubits_per_qpu=max_qubits_per_qpu)
            if partition is not None:
                partitions[start, end] = partition
    return partitions

def compute_stitching_cost(partition1, partition2):
    """
    计算两个相邻层分区间的Stitching成本。
    成本定义为与上一层相比需要更换QPU的量子比特数量。
    """
    qubit_to_qpu_1 = {}
    for qpu, qubits in partition1.items():
        for q in qubits:
            qubit_to_qpu_1[q] = qpu
    qubit_to_qpu_2 = {}
    for qpu, qubits in partition2.items():
        for q in qubits:
            qubit_to_qpu_2[q] = qpu
    all_qubits = set(qubit_to_qpu_1.keys()).union(set(qubit_to_qpu_2.keys()))
    cost = 0
    for q in all_qubits:
        if qubit_to_qpu_1.get(q, -1) != qubit_to_qpu_2.get(q, -1):
            cost += 1
    return cost

def dynamic_programming_partitions(circuit, num_qpus, max_qubits_per_qpu):
    """
    利用动态规划寻找从第0层到最后一层的最优分区序列，使Stitching成本最小。
    这里的示例简单地对每层计算分区，然后相邻层之间计算Stitching成本。
    """
    partitions = find_qubit_partition(circuit, num_qpus, max_qubits_per_qpu)
    if not partitions:
        print('No valid partitions found.')
        return
    return dynamic_programming_stitching(len(circuit), partitions)
import math

def dynamic_programming_stitching(m, partitions):
    dp = [[math.inf] * m for _ in range(m)]
    dp_map = {}
    for j in range(0, m):
        if (0, j) in partitions:
            dp[0][j] = 0
            dp_map[0, j] = partitions[0, j]
    for j in range(0, m):
        for i in range(0, j + 1):
            if (i, j) in partitions:
                best_cost = math.inf
                best_i_prime = -1
                for i_prime in range(0, i):
                    if dp[i_prime][i - 1] < math.inf:
                        cost = dp[i_prime][i - 1]
                        if cost < best_cost:
                            best_cost = cost
                            best_i_prime = i_prime
                if best_cost != math.inf:
                    dp_map[i, j] = bipartite_max_mapping(dp_map[best_i_prime, i - 1], partitions[i, j])
                    stiching_cost = compute_stitching_cost(dp_map[best_i_prime, i - 1], dp_map[i, j])
                    best_cost += stiching_cost
                if best_cost < dp[i][j]:
                    dp[i][j] = best_cost
    final_cost = math.inf
    final_i = None
    for i in range(0, m):
        if dp[i][m - 1] < final_cost:
            final_cost = dp[i][m - 1]
            final_i = i
    print(final_i)
    print(dp_map[final_i, m - 1])
    print('final_cost:', final_cost)
    qpu_in_use = len(dp_map[final_i, m - 1])
    while final_i != 0:
        final_cost_temp = math.inf
        m = final_i
        for i in range(0, m):
            if dp[i][m - 1] < final_cost_temp:
                final_cost_temp = dp[i][m - 1]
                final_i = i
        if qpu_in_use < len(dp_map[final_i, m - 1]):
            qpu_in_use = len(dp_map[final_i, m - 1])
    return (final_cost, qpu_in_use)
