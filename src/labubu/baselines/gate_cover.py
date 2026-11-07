import math
from dataclasses import dataclass
import sys
from typing import Dict, List, Tuple, Optional, Set, Any
from .metis_like import metis_like_part_graph
import numpy as np

def analyze_qasm(qasm_filelocation: str) -> Tuple[np.ndarray, int]:
    """Shared QASM reader; retains actual CX control/target orientation."""
    from ..circuit import read_qasm
    layers = read_qasm(qasm_filelocation, directed=True)
    return layers, layers.shape[1]

class Circuit:

    def __init__(self, filelocation: str):
        self.layers, self.num_qubits = analyze_qasm(filelocation)

    def count_2q_gates(self) -> int:
        count = 0
        for t in range(self.layers.shape[0]):
            row = self.layers[t]
            for q in range(self.num_qubits):
                a = int(row[q])
                if 0 <= a < 10000:
                    count += 1
        return count

def qpu_count(Q: int, L: int) -> int:
    """User requirement: P = floor(Q/L) + 1 (and at least 1)."""
    if L <= 0:
        raise ValueError('L must be > 0.')
    return max(1, math.ceil(Q / L))

def build_comm_graph(layers: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Build the communication graph G(V,E,w) from directed CNOT encoding.
      - V: qubits
      - edge (u,v): if any CNOT between u and v
      - weight w(u,v): number of CNOTs between u and v

    Returns:
      adjacency: List[List[int]]
      eweights:  List[List[int]] aligned with adjacency
    """
    if layers.size == 0:
        return ([], [])
    T, Q = layers.shape
    edge_w: Dict[Tuple[int, int], int] = {}
    for t in range(T):
        row = layers[t]
        for ctrl in range(Q):
            tgt = int(row[ctrl])
            if 0 <= tgt < 10000:
                u, v = (ctrl, tgt)
                if u > v:
                    u, v = (v, u)
                edge_w[u, v] = edge_w.get((u, v), 0) + 1
    adjacency = [[] for _ in range(Q)]
    eweights = [[] for _ in range(Q)]
    for (u, v), w in edge_w.items():
        adjacency[u].append(v)
        eweights[u].append(w)
        adjacency[v].append(u)
        eweights[v].append(w)
    return (adjacency, eweights)

def metis_initial_mapping(layers: np.ndarray, L: int) -> Dict[int, int]:
    """
    Produce an initial qubit->QPU mapping using METIS multilevel k-way partitioning
    if pymetis is available. Otherwise falls back to a simple greedy packing.

    Uses P = floor(Q/L)+1 QPUs.

    Capacity repair:
      If any partition exceeds L, move extra qubits into any non-full QPU (lowest-fill first).
    """
    if layers.size == 0:
        return {}
    Q = layers.shape[1]
    P = qpu_count(Q, L)
    adjacency, eweights = build_comm_graph(layers)

    def fallback_mapping() -> Dict[int, int]:
        m: Dict[int, int] = {}
        counts = [0] * P
        cur = 0
        for q in range(Q):
            while cur < P and counts[cur] >= L:
                cur += 1
            if cur >= P:
                cur = P - 1
            m[q] = cur
            counts[cur] += 1
        return m
    parts = metis_like_part_graph(adjacency=adjacency, eweights=eweights, nparts=P, seed=0)
    parts = list(parts)
    qpu_qubits: List[List[int]] = [[] for _ in range(P)]
    for q, p in enumerate(parts):
        p = int(p)
        if p < 0 or p >= P:
            p = p % P
        qpu_qubits[p].append(q)

    def pick_nonfull_qpu() -> Optional[int]:
        loads = [(len(qpu_qubits[p]), p) for p in range(P)]
        loads.sort()
        for load, p in loads:
            if load < L:
                return p
        return None
    for p in range(P):
        while len(qpu_qubits[p]) > L:
            q = qpu_qubits[p].pop()
            dst = pick_nonfull_qpu()
            if dst is None:
                qpu_qubits[p].append(q)
                break
            qpu_qubits[dst].append(q)
            parts[q] = dst
    return {q: int(parts[q]) for q in range(Q)}

@dataclass(frozen=True)
class Gate:
    gid: int
    layer: int
    control: int
    target: int

@dataclass
class TeleCandidate:
    kind: str
    migrations: List[Tuple[int, int, int]]
    ncov_est: int
    n_epr: int

    def cost(self) -> float:
        return self.n_epr / max(1, self.ncov_est)

def _extract_gates(layers: np.ndarray) -> List[Gate]:
    """Flatten layers into an ordered list of directed CNOT gates."""
    if layers.size == 0:
        return []
    T, Q = layers.shape
    gates: List[Gate] = []
    gid = 0
    for t in range(T):
        row = layers[t]
        for ctrl in range(Q):
            tgt = int(row[ctrl])
            if tgt == -1:
                continue
            if 0 <= tgt < 10000:
                if not 0 <= tgt < Q:
                    raise ValueError(f'Invalid target index at layer {t}: layers[{t}][{ctrl}]={tgt}, Q={Q}.')
                back = int(row[tgt])
                if back != ctrl + 10000:
                    raise ValueError(f'Inconsistent encoding at layer {t}: layers[{t}][{ctrl}]={tgt} but layers[{t}][{tgt}]={back} (expected {ctrl + 10000}).')
                gates.append(Gate(gid=gid, layer=t, control=ctrl, target=tgt))
                gid += 1
    return gates

def _initial_mapping(Q: int, L: int) -> Tuple[Dict[int, int], List[Set[int]]]:
    """Sequential packing into P=floor(Q/L)+1 QPUs."""
    P = qpu_count(Q, L)
    mapping: Dict[int, int] = {}
    qpu_qubits: List[Set[int]] = [set() for _ in range(P)]
    counts = [0] * P
    p = 0
    for q in range(Q):
        while p < P and counts[p] >= L:
            p += 1
        if p >= P:
            p = P - 1
        mapping[q] = p
        qpu_qubits[p].add(q)
        counts[p] += 1
    return (mapping, qpu_qubits)

def _is_local(g: Gate, mapping: Dict[int, int]) -> bool:
    return mapping[g.control] == mapping[g.target]

def _apply_migrations(mapping: Dict[int, int], qpu_qubits: List[Set[int]], migrations: List[Tuple[int, int, int]]) -> None:
    for q, src, dst in migrations:
        if mapping.get(q) != src:
            raise RuntimeError(f'Migration mismatch: qubit {q} expected in QPU {src}, found {mapping.get(q)}.')
        qpu_qubits[src].remove(q)
        qpu_qubits[dst].add(q)
        mapping[q] = dst

def _estimate_ncov_after_migration(g: Gate, gates: List[Gate], idx: int, mapping: Dict[int, int], migrations: List[Tuple[int, int, int]], lookahead_gates: int) -> int:
    """
    ONLY for objective estimation.
    Simulate migrations and count how many subsequent gates become local within the lookahead window.
    """
    sim_map = dict(mapping)
    for q, _, dst in migrations:
        sim_map[q] = dst
    ncov = 1
    end = min(len(gates), idx + 1 + lookahead_gates)
    for j in range(idx + 1, end):
        gj = gates[j]
        if sim_map[gj.control] == sim_map[gj.target]:
            ncov += 1
    return ncov

def _teledata_candidates_for_gate(g: Gate, gates: List[Gate], idx: int, mapping: Dict[int, int], qpu_qubits: List[Set[int]], L: int, lookahead_gates: int) -> List[TeleCandidate]:
    """
    Generate feasible TeleData candidates for this remote gate.
    Fully-connected: 1 EPR per teleported qubit.
    Capacity constraint: dst QPU must not exceed L.
    """
    src_c = mapping[g.control]
    src_t = mapping[g.target]
    P = len(qpu_qubits)

    def capacity_ok(dst_qpu: int, extra: int) -> bool:
        return len(qpu_qubits[dst_qpu]) + extra <= L
    cands: List[TeleCandidate] = []
    if capacity_ok(src_t, 1):
        mig = [(g.control, src_c, src_t)]
        ncov = _estimate_ncov_after_migration(g, gates, idx, mapping, mig, lookahead_gates)
        cands.append(TeleCandidate(kind='TeleData', migrations=mig, ncov_est=ncov, n_epr=1))
    if capacity_ok(src_c, 1):
        mig = [(g.target, src_t, src_c)]
        ncov = _estimate_ncov_after_migration(g, gates, idx, mapping, mig, lookahead_gates)
        cands.append(TeleCandidate(kind='TeleData', migrations=mig, ncov_est=ncov, n_epr=1))
    for k in range(P):
        if k == src_c or k == src_t:
            continue
        if capacity_ok(k, 2):
            mig = [(g.control, src_c, k), (g.target, src_t, k)]
            ncov = _estimate_ncov_after_migration(g, gates, idx, mapping, mig, lookahead_gates)
            cands.append(TeleCandidate(kind='TeleData', migrations=mig, ncov_est=ncov, n_epr=2))
    return cands

def _telegate_candidate_for_gate() -> TeleCandidate:
    """
    Per-gate TeleGate (no multi-gate reuse), so we don't double-count future gates.
      - ncov_est = 1
      - n_epr = 1
    """
    return TeleCandidate(kind='TeleGate', migrations=[], ncov_est=1, n_epr=1)

def gate_cover(layers: np.ndarray, L: int, *, lookahead_gates: int=16, mapping0: Optional[Dict[int, int]]=None) -> Dict[str, Any]:
    """
    Gate-by-gate scheduling.

    IMPORTANT:
      - We DO NOT pre-mark future gates as covered.
      - Future gates ONLY contribute to objective estimation (ncov_est).
      - Every gate is visited in order; its local/remote nature is evaluated under CURRENT mapping.
    """
    if layers.size == 0:
        return {'D': [], 'final_mapping': {}, 'gates': []}
    gates = _extract_gates(layers)
    Q = layers.shape[1]
    P = qpu_count(Q, L)
    if mapping0 is None:
        mapping, qpu_qubits = _initial_mapping(Q, L)
    else:
        mapping = dict(mapping0)
        qpu_qubits = [set() for _ in range(P)]
        for q in range(Q):
            if q not in mapping:
                raise ValueError(f'mapping0 missing qubit {q}.')
            p = int(mapping[q])
            if not 0 <= p < P:
                raise ValueError(f'mapping0[{q}]={p} invalid, P={P}.')
            qpu_qubits[p].add(q)
        for p in range(P):
            if len(qpu_qubits[p]) > L:
                raise ValueError(f'Initial mapping violates capacity: QPU {p} has {len(qpu_qubits[p])}>{L}.')
    D: List[Dict[str, Any]] = []
    for idx, g in enumerate(gates):
        if _is_local(g, mapping):
            D.append({'op': 'LOCAL', 'gid': g.gid, 'layer': g.layer, 'control': g.control, 'target': g.target})
            continue
        td_cands = _teledata_candidates_for_gate(g, gates, idx, mapping, qpu_qubits, L, lookahead_gates)
        tg_cand = _telegate_candidate_for_gate()
        best = min(td_cands + [tg_cand], key=lambda c: c.cost())
        if best.kind == 'TeleData':
            _apply_migrations(mapping, qpu_qubits, best.migrations)
            D.append({'op': 'TELEDATA', 'gid_trigger': g.gid, 'layer_trigger': g.layer, 'migrations': best.migrations, 'n_epr': best.n_epr, 'ncov_est': best.ncov_est, 'cost': best.cost()})
        else:
            D.append({'op': 'TELEGATE', 'gid_trigger': g.gid, 'layer_trigger': g.layer, 'control': g.control, 'target': g.target, 'n_epr': 1, 'ncov_est': 1, 'cost': 1.0})
    return {'D': D, 'final_mapping': mapping, 'gates': gates}
