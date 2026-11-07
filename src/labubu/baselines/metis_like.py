from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random
import math

@dataclass
class CoarseGraph:
    adjacency: List[List[int]]
    eweights: List[List[int]]
    vweights: List[int]
    fine2coarse: List[int]
    coarse2fine: List[List[int]]

def _build_edge_weight_map(adjacency: List[List[int]], eweights: List[List[int]]) -> List[Dict[int, int]]:
    n = len(adjacency)
    maps: List[Dict[int, int]] = [dict() for _ in range(n)]
    for u in range(n):
        for v, w in zip(adjacency[u], eweights[u]):
            maps[u][v] = maps[u].get(v, 0) + int(w)
    return maps

def _heavy_edge_matching(adjacency: List[List[int]], eweights: List[List[int]], vweights: List[int], rng: random.Random) -> CoarseGraph:
    """
    Coarsen by heavy-edge matching.
    Produces a coarse graph where each coarse vertex is either a matched pair or singleton.
    """
    n = len(adjacency)
    order = list(range(n))
    rng.shuffle(order)
    matched = [-1] * n
    taken = [False] * n
    wmap = _build_edge_weight_map(adjacency, eweights)
    for u in order:
        if taken[u]:
            continue
        best_v = -1
        best_w = -1
        for v in adjacency[u]:
            if taken[v]:
                continue
            w = wmap[u].get(v, 0)
            if w > best_w:
                best_w = w
                best_v = v
        if best_v != -1:
            taken[u] = True
            taken[best_v] = True
            matched[u] = best_v
            matched[best_v] = u
    fine2coarse = [-1] * n
    coarse2fine: List[List[int]] = []
    for u in range(n):
        if fine2coarse[u] != -1:
            continue
        v = matched[u]
        if v != -1 and fine2coarse[v] == -1 and (v != u):
            cid = len(coarse2fine)
            fine2coarse[u] = cid
            fine2coarse[v] = cid
            coarse2fine.append([u, v])
        else:
            cid = len(coarse2fine)
            fine2coarse[u] = cid
            coarse2fine.append([u])
    cn = len(coarse2fine)
    cvw = [0] * cn
    for cid, fine_list in enumerate(coarse2fine):
        cvw[cid] = sum((vweights[x] for x in fine_list))
    coarse_maps: List[Dict[int, int]] = [dict() for _ in range(cn)]
    for u in range(n):
        cu = fine2coarse[u]
        for v, w in zip(adjacency[u], eweights[u]):
            cv = fine2coarse[v]
            if cu == cv:
                continue
            coarse_maps[cu][cv] = coarse_maps[cu].get(cv, 0) + int(w)
    cadj: List[List[int]] = [[] for _ in range(cn)]
    cew: List[List[int]] = [[] for _ in range(cn)]
    for cu in range(cn):
        items = list(coarse_maps[cu].items())
        items.sort(key=lambda x: x[0])
        cadj[cu] = [cv for cv, _ in items]
        cew[cu] = [w for _, w in items]
    return CoarseGraph(adjacency=cadj, eweights=cew, vweights=cvw, fine2coarse=fine2coarse, coarse2fine=coarse2fine)

def _coarsen_multilevel(adjacency: List[List[int]], eweights: List[List[int]], vweights: List[int], rng: random.Random, min_coarse_n: int=30) -> List[CoarseGraph]:
    """
    Build a multilevel hierarchy: level[0] is first coarsening step (from original),
    last level is the coarsest graph.
    """
    levels: List[CoarseGraph] = []
    cur_adj, cur_ew, cur_vw = (adjacency, eweights, vweights)
    while len(cur_adj) > min_coarse_n:
        cg = _heavy_edge_matching(cur_adj, cur_ew, cur_vw, rng)
        levels.append(cg)
        cur_adj, cur_ew, cur_vw = (cg.adjacency, cg.eweights, cg.vweights)
        if len(cur_adj) >= len(cg.fine2coarse) * 0.9:
            break
    return levels

def _initial_kway_greedy(adjacency: List[List[int]], eweights: List[List[int]], vweights: List[int], nparts: int, rng: random.Random) -> List[int]:
    """
    Greedy balanced growing partition on the coarsest graph.
    Heuristic: pick seeds, then expand each part by adding best gain vertex while respecting balance.
    """
    n = len(adjacency)
    if nparts <= 1:
        return [0] * n
    total_w = sum(vweights)
    target = total_w / nparts
    max_w = math.ceil(target * 1.05) if target > 0 else 0
    parts = [-1] * n
    part_w = [0] * nparts
    vol = [sum(eweights[u]) for u in range(n)]
    seeds = sorted(range(n), key=lambda u: vol[u], reverse=True)[:nparts]
    for p, u in enumerate(seeds):
        parts[u] = p
        part_w[p] += vweights[u]
    unassigned = [u for u in range(n) if parts[u] == -1]
    adj_map = _build_edge_weight_map(adjacency, eweights)
    for u in unassigned:
        pass
    rng.shuffle(unassigned)
    for u in unassigned:
        best_p = None
        best_score = -1
        for p in range(nparts):
            if part_w[p] + vweights[u] > max_w:
                continue
            score = 0
            for v in adjacency[u]:
                if parts[v] == p:
                    score += adj_map[u].get(v, 0)
            if score > best_score:
                best_score = score
                best_p = p
        if best_p is None:
            best_p = min(range(nparts), key=lambda p: part_w[p])
        parts[u] = best_p
        part_w[best_p] += vweights[u]
    return parts

def _compute_gain_to_move(u: int, cur_p: int, new_p: int, adjacency: List[List[int]], eweights: List[List[int]], parts: List[int]) -> int:
    """
    Gain = (edge weight from u to current part) - (edge weight from u to new part)
    Moving u decreases cut if it reduces external edges; this simplified gain works well enough.
    """
    w_to_cur = 0
    w_to_new = 0
    for v, w in zip(adjacency[u], eweights[u]):
        pv = parts[v]
        if pv == cur_p:
            w_to_cur += w
        elif pv == new_p:
            w_to_new += w
    return w_to_cur - w_to_new

def _refine_boundary_greedy(adjacency: List[List[int]], eweights: List[List[int]], vweights: List[int], nparts: int, parts: List[int], rng: random.Random, passes: int=3) -> List[int]:
    """
    Simple k-way refinement: try moving boundary vertices to improve cut while keeping balance.
    """
    n = len(adjacency)
    total_w = sum(vweights)
    target = total_w / nparts
    max_w = math.ceil(target * 1.05) if target > 0 else 0
    min_w = math.floor(target * 0.95) if target > 0 else 0
    part_w = [0] * nparts
    for u in range(n):
        part_w[parts[u]] += vweights[u]
    for _ in range(passes):
        order = list(range(n))
        rng.shuffle(order)
        moved = 0
        for u in order:
            cur_p = parts[u]
            neigh_parts = set()
            for v in adjacency[u]:
                pv = parts[v]
                if pv != cur_p:
                    neigh_parts.add(pv)
            if not neigh_parts:
                continue
            best_p = cur_p
            best_gain = 0
            for p in neigh_parts:
                if part_w[p] + vweights[u] > max_w:
                    continue
                if part_w[cur_p] - vweights[u] < min_w:
                    continue
                gain = _compute_gain_to_move(u, cur_p, p, adjacency, eweights, parts)
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            if best_p != cur_p:
                parts[u] = best_p
                part_w[cur_p] -= vweights[u]
                part_w[best_p] += vweights[u]
                moved += 1
        if moved == 0:
            break
    return parts

def metis_like_part_graph(adjacency: List[List[int]], eweights: List[List[int]], nparts: int, vweights: Optional[List[int]]=None, seed: int=0) -> List[int]:
    """
    Public API: returns parts assignment for original graph.
    """
    n = len(adjacency)
    if n == 0:
        return []
    if nparts <= 1:
        return [0] * n
    if vweights is None:
        vweights = [1] * n
    rng = random.Random(seed)
    levels = _coarsen_multilevel(adjacency, eweights, vweights, rng, min_coarse_n=max(30, 4 * nparts))
    cur_adj, cur_ew, cur_vw = (adjacency, eweights, vweights)
    for cg in levels:
        cur_adj, cur_ew, cur_vw = (cg.adjacency, cg.eweights, cg.vweights)
    parts_coarse = _initial_kway_greedy(cur_adj, cur_ew, cur_vw, nparts, rng)
    parts_coarse = _refine_boundary_greedy(cur_adj, cur_ew, cur_vw, nparts, parts_coarse, rng, passes=5)
    for cg in reversed(levels):
        fine_n = len(cg.fine2coarse)
        parts_fine = [0] * fine_n
        for u in range(fine_n):
            parts_fine[u] = parts_coarse[cg.fine2coarse[u]]
        parts_coarse = parts_fine
    parts_final = _refine_boundary_greedy(adjacency, eweights, vweights, nparts, parts_coarse, rng, passes=8)
    return parts_final
