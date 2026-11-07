# Experiments

The benchmark grid is QFT/RD × widths {30,40,50,60,70,80,90,100} × capacities
{5,15,25}: 48 parameter points. The experiment configuration is:

| Parameter | Value |
|---|---|
| Independent runs | 18 |
| Optimizer seeds | 0–17 |
| Iterations | 1000 |
| BP interval | 50 |
| Maximum repair rounds per layer update | 500 |
| Reserved initialization slots per QPU | 1 |
| QPU counts | `benchmarks/manifest.json` |

Seeds refer to optimization, not benchmark generation. Each point reports the
minimum entanglement cost across runs. `summary.csv` also contains mean, sample
standard deviation and range. All per-seed mappings and checkpoints are available.
The fixed inputs define the workload; parameter counts alone do not identify a circuit.
RD30/50/70/90 differ slightly in depth/CX count from the paper's Table I.

## Comparison and plotting

`results/baselines.csv` contains the fixed comparison costs for Telegate-SA,
Telegate-RD, Teledata-ZS, GC and GC+LABUBU. The grid runner executes LABUBU; it does
not execute these baseline algorithms. In particular, the baseline helper modules
are not an end-to-end GC-initialized LABUBU experiment interface.

Relative improvement is `(baseline_cost - LABUBU_cost) / baseline_cost × 100`.
`plot_fig6.py` displays this value on the left axis and LABUBU entanglement cost on
the right axis. The baseline series labelled GC+LABUBU* retains its source-table
label; the table alone does not specify the full initialization/configuration pipeline.

## Execution

Run commands from the repository root. The default Python backend supports process
parallelism across seeds. The optional native backend can be built with the command
in the README. `verify_native.py` compares full mappings, checkpoints and costs
against Python. `verify_results.py` checks the 16 input hashes and all 864 stored
mappings, capacity constraints, costs, checkpoint minima and seed coverage.

`outputs/` is ignored by Git. Cached runs carry fingerprints of the input, parameters,
runner and optimizer sources; native runs additionally fingerprint backend files.

## Scope

The included experiment runner and result records cover the 48-point unencoded
comparison. Steane expansion is available as an algorithm input utility; the project
does not provide executable experiment drivers for every encoded figure, repair
statistic, solver comparison or ablation in the paper. The baseline table supplies
comparison data independently of the auxiliary baseline APIs.
