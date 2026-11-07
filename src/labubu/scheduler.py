"""CPG, bounded perturbation, capacity repair and checkpoint selection."""
from collections import Counter
from numbers import Integral
import random
import numpy as np


def _integer_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'iu':
        raise ValueError(f'{name} must contain integers; implicit truncation is not allowed')
    return np.array(array, dtype=int, copy=True)


def entanglement_cost(layers, mapping):
    remote = sum(int(mapping[t, q] != mapping[t, partner])
                 for t, row in enumerate(layers)
                 for q, partner in enumerate(row) if partner > q)
    migrations = int(np.count_nonzero(mapping[1:] != mapping[:-1]))
    return {"total": remote + migrations, "telegate": remote, "teledata": migrations}


def repair(proposal, current, capacities, max_repair=500):
    """Bounded rollback, with a fresh repair budget for each layer update."""
    candidate = list(proposal)
    for _ in range(max_repair):
        counts = Counter(candidate)
        excess = {i: counts[i] - cap for i, cap in enumerate(capacities)
                  if counts[i] > cap}
        if not excess:
            return candidate
        for q, target in enumerate(candidate):
            if excess.get(target, 0) > 0 and current[q] != target:
                excess[target] -= 1
                candidate[q] = int(current[q])
        if all(count <= capacities[i] for i, count in Counter(candidate).items()):
            return candidate
    return list(current)


def initialize(layers, capacities, idle):
    """Layer-local initialization with qubit-order packing and anchor backfill."""
    effective = [cap - idle for cap in capacities]
    width = layers.shape[1]
    if any(cap < 1 for cap in effective) or width > sum(effective):
        raise ValueError("Insufficient capacity after reserving idle slots")
    mapping = np.full(layers.shape, -1, dtype=int)
    for t, row in enumerate(layers):
        usage = [0] * len(capacities)
        temp_layer = row.copy()
        current_qpu = 0
        anchor = -1
        for q in range(width):
            if current_qpu < len(capacities) and usage[current_qpu] == effective[current_qpu]:
                current_qpu += 1
            partner = int(temp_layer[q])
            if partner == -2:
                continue
            if partner == -1:
                target = anchor if anchor != -1 else current_qpu
                if target >= len(capacities):
                    raise ValueError("Not enough QPUs for layer-local initialization")
                mapping[t, q] = target
                usage[target] += 1
                anchor = -1
            else:
                if current_qpu >= len(capacities):
                    raise ValueError("Not enough QPUs for layer-local initialization")
                if usage[current_qpu] >= effective[current_qpu] - 1:
                    anchor = current_qpu
                    current_qpu += 1
                if current_qpu >= len(capacities) or effective[current_qpu] - usage[current_qpu] < 2:
                    raise ValueError("Not enough pair capacity for layer-local initialization")
                mapping[t, q] = current_qpu
                mapping[t, partner] = current_qpu
                usage[current_qpu] += 2
                temp_layer[partner] = -2
    return mapping


def sweep(layers, mapping, capacities, rng, perturb=False, max_repair=500):
    depth, width = layers.shape
    for t in range(depth):
        proposal = []
        for q in range(width):
            candidates = []
            if t + 1 < depth:
                candidates.append(int(mapping[t + 1, q]))
            if t > 0:
                candidates.append(int(mapping[t - 1, q]))
            if layers[t, q] != -1:
                candidates.append(int(mapping[t, layers[t, q]]))
            if not candidates:
                candidates = [int(mapping[t, q])]
            if perturb:
                target = rng.choice(candidates)
            else:
                counts = Counter(candidates)
                winners = list(dict.fromkeys(x for x in candidates if counts[x] == max(counts.values())))
                target = rng.choice(winners)
            proposal.append(target)
        mapping[t] = repair(proposal, mapping[t], capacities, max_repair)


def optimize(layers, capacities, iterations=1000, interval=50, seed=0,
             idle=1, max_repair=500, cpg_only=False, initial_mapping=None):
    layers = _integer_array(layers, 'Interaction matrix')
    capacities = list(capacities)
    if layers.ndim != 2 or min(layers.shape) < 1:
        raise ValueError("Expected a nonempty depth-by-width interaction matrix")
    if not capacities or any(not isinstance(c, Integral) or isinstance(c, bool) or c < 1 for c in capacities):
        raise ValueError("Capacities must be positive integers")
    for name, value in [('iterations', iterations), ('interval', interval),
                        ('max_repair', max_repair), ('idle', idle)]:
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise ValueError(f'{name} must be an integer')
    if iterations < 1 or interval < 1 or max_repair < 1 or idle < 0:
        raise ValueError("Invalid iteration, repair, interval or idle parameter")
    for row in layers:
        for q, p in enumerate(row):
            if p != -1 and (p < 0 or p >= len(row) or p == q or row[p] != q):
                raise ValueError("Interaction matrix must contain symmetric disjoint pairs")
    mapping = initialize(layers, capacities, idle) if initial_mapping is None else _integer_array(initial_mapping, 'Initial mapping')
    if mapping.shape != layers.shape or np.any(mapping < 0) or np.any(mapping >= len(capacities)):
        raise ValueError("Invalid initial mapping")
    if any(any(n > capacities[i] for i, n in Counter(row).items()) for row in mapping):
        raise ValueError("Initial mapping exceeds capacity")
    rng = random.Random(seed)
    best, best_cost, checkpoints = mapping.copy(), float("inf"), []
    for iteration in range(1, iterations + 1):
        # Zero-based epoch schedule: BP at epochs
        # 100, 150, ..., 850 for iterations=1000, interval=50.
        epoch = iteration - 1
        perturb = (not cpg_only and 100 <= epoch < iterations - 100
                   and epoch % interval == 0)
        if perturb:
            cost = entanglement_cost(layers, mapping)["total"]
            checkpoints.append({"before_iteration": iteration, "cost": cost})
            if cost < best_cost:
                best, best_cost = mapping.copy(), cost
        sweep(layers, mapping, capacities, rng, perturb, max_repair)
    cost = entanglement_cost(layers, mapping)["total"]
    checkpoints.append({"after_iteration": iterations, "cost": cost})
    if cost < best_cost:
        best = mapping.copy()
    return {**entanglement_cost(layers, best), "mapping": best.tolist(),
            "seed": seed, "checkpoints": checkpoints}
