# Method

## Objective

An interaction matrix has shape `(depth, width)`. Each row encodes disjoint CX pairs:
`G[t,q]=p` and `G[t,p]=q`; a value of `-1` indicates no CX on that qubit in that layer.
A mapping matrix of the same shape stores QPU indices.

The total entanglement cost is the number of CX pairs mapped to different QPUs plus
the number of qubits changing QPU between adjacent layers. Each pair or migration
is counted once. Per-layer occupancy must not exceed QPU capacity.

## Initialization and updates

Initialization packs qubits in index order, placing each layer's CX partners together.
Anchor backfill uses remaining space for unpaired qubits. One slot per QPU is reserved
at initialization; optimization may use that space.

Each qubit's candidate list contains its next-layer QPU, previous-layer QPU and
current-layer CX partner's QPU, when defined. Multiplicity is retained. CPG chooses
a majority value with random tie breaking. BP samples an entry uniformly from this
list. An empty list falls back to the qubit's current QPU.

Decisions are buffered within a layer and committed after bounded capacity repair.
Repair rolls back migrations into overloaded QPUs, with an independent budget for
each layer update. A layer falls back to its current mapping if the budget is exhausted.
Layers are processed in temporal order.

## Iteration schedule and solution selection

For 1000 iterations, zero-based epochs 0–99 and 900–999 use CPG. BP replaces CPG at
epochs 100,150,...,850 for interval 50. Budgets at or below 200 use only CPG.
`--cpg-only` disables BP for the entire run.

The algorithm evaluates cost immediately before BP and after the final iteration,
then selects the lowest-cost recorded mapping. All returned cost components are
computed from that mapping. Layer-synchronous decisions do not guarantee global
cost monotonicity, so this selection is over recorded checkpoints rather than every
intermediate state. Among independent runs, the lowest returned cost is selected;
ties follow seed order.

## Steane interaction expansion

Each input qubit expands into seven data qubits and one syndrome ancilla. The
expansion uses transversal CX interactions and syndrome interactions at a configurable
period (default 5). Named Clifford gates are required at the input interface.
This is an interaction-cost model, not state, noise or logical-error simulation.
