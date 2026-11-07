import math
from dataclasses import dataclass
import os
from typing import Dict, List, Tuple, Optional, Set, Any
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
        """Count CNOTs by counting control entries (<10000)."""
        count = 0
        for t in range(self.layers.shape[0]):
            row = self.layers[t]
            for q in range(self.num_qubits):
                a = int(row[q])
                if 0 <= a < 10000:
                    count += 1
        return count

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
    if L <= 0:
        raise ValueError('L must be > 0.')
    P = Q // L + 1
    mapping: Dict[int, int] = {}
    qpu_qubits: List[Set[int]] = [set() for _ in range(P)]
    for q in range(Q):
        p = q // L
        mapping[q] = p
        qpu_qubits[p].add(q)
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
    Under the user's requested semantics ("do not put future covered gates into covered/D"),
    we DO NOT implement multi-gate TeleGate reuse here, otherwise you'd double-count future gates.
    So TeleGate is per-remote-gate:
      - ncov_est = 1
      - n_epr = 1
    """
    return TeleCandidate(kind='TeleGate', migrations=[], ncov_est=1, n_epr=1)

def gate_cover(layers: np.ndarray, L: int, *, lookahead_gates: int=16, mapping0: Optional[Dict[int, int]]=None) -> Dict[str, Any]:
    """
    Gate-by-gate scheduling.

    IMPORTANT CHANGE (per user):
      - We DO NOT pre-mark future gates as covered.
      - Future gates ONLY contribute to objective estimation (ncov_est).
      - Every gate will be visited in order; its local/remote nature is evaluated under CURRENT mapping.

    New (per user):
      - Return mapping_by_layer: a T x Q list-of-lists where
            mapping_by_layer[t][q] = QPU id of logical qubit q
        after finishing all gates in layer t (i.e., end-of-layer snapshot).
    """
    if layers.size == 0:
        return {'D': [], 'final_mapping': {}, 'gates': [], 'mapping_events': [], 'mapping_by_layer': []}
    gates = _extract_gates(layers)
    T = layers.shape[0]
    Q = layers.shape[1]
    if mapping0 is None:
        mapping, qpu_qubits = _initial_mapping(Q, L)
    else:
        mapping = dict(mapping0)
        P = Q // L + 1
        qpu_qubits = [set() for _ in range(P)]
        for q in range(Q):
            if q not in mapping:
                raise ValueError(f'mapping0 missing qubit {q}.')
            p = mapping[q]
            if not 0 <= p < P:
                raise ValueError(f'mapping0[{q}]={p} invalid, P={P}.')
            qpu_qubits[p].add(q)
        for p in range(P):
            if len(qpu_qubits[p]) > L:
                raise ValueError(f'Initial mapping violates capacity: QPU {p} has {len(qpu_qubits[p])}>{L}.')
    D: List[Dict[str, Any]] = []
    mapping_events: List[Dict[str, Any]] = []
    mapping_by_layer: List[Optional[List[int]]] = [None] * T

    def _snapshot_vector() -> List[int]:
        return [mapping[q] for q in range(Q)]
    if len(gates) == 0:
        v = _snapshot_vector()
        for t in range(T):
            mapping_by_layer[t] = v[:]
        return {'D': D, 'final_mapping': mapping, 'gates': gates, 'mapping_events': mapping_events, 'mapping_by_layer': mapping_by_layer}
    last_layer_done = -1
    for idx, g in enumerate(gates):
        if g.layer > last_layer_done + 1:
            v = _snapshot_vector()
            for t in range(last_layer_done + 1, min(g.layer, T)):
                mapping_by_layer[t] = v[:]
            last_layer_done = g.layer - 1
        if _is_local(g, mapping):
            D.append({'op': 'LOCAL', 'gid': g.gid, 'layer': g.layer, 'control': g.control, 'target': g.target})
        else:
            td_cands = _teledata_candidates_for_gate(g, gates, idx, mapping, qpu_qubits, L, lookahead_gates)
            tg_cand = _telegate_candidate_for_gate()
            best = min(td_cands + [tg_cand], key=lambda c: c.cost())
            if best.kind == 'TeleData':
                _apply_migrations(mapping, qpu_qubits, best.migrations)
                D.append({'op': 'TELEDATA', 'gid_trigger': g.gid, 'layer_trigger': g.layer, 'migrations': best.migrations, 'n_epr': best.n_epr, 'ncov_est': best.ncov_est, 'cost': best.cost()})
                mapping_events.append({'gid_trigger': g.gid, 'layer_trigger': g.layer, 'migrations': best.migrations, 'mapping': mapping.copy()})
            else:
                D.append({'op': 'TELEGATE', 'gid_trigger': g.gid, 'layer_trigger': g.layer, 'control': g.control, 'target': g.target, 'n_epr': 1, 'ncov_est': 1, 'cost': 1.0})
        next_layer = gates[idx + 1].layer if idx + 1 < len(gates) else None
        if next_layer is None or next_layer != g.layer:
            if 0 <= g.layer < T:
                mapping_by_layer[g.layer] = _snapshot_vector()
                last_layer_done = max(last_layer_done, g.layer)
    if last_layer_done < T - 1:
        v = _snapshot_vector()
        for t in range(last_layer_done + 1, T):
            mapping_by_layer[t] = v[:]
    for t in range(T):
        if mapping_by_layer[t] is None:
            mapping_by_layer[t] = _snapshot_vector()
    return {'D': D, 'final_mapping': mapping, 'gates': gates, 'mapping_events': mapping_events, 'mapping_by_layer': mapping_by_layer}
