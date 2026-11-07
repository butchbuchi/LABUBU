# Data and interfaces

## Benchmarks

`benchmarks/qft/` and `benchmarks/rd/` contain eight fixed QASM inputs each.
`manifest.json` records their SHA256 values and the QPU count at each benchmark/capacity
combination. The fixed QASM files are the executable benchmark definition.
Benchmark-generation scripts and generation seeds are not part of this distribution.
The input reader supports both standard QASM headers and header-stripped files.

## Results

- `results/baselines.csv`: circuit type, width, capacity, baseline name and cost.
- `results/fig6/runs/*.json`: per-seed summary, mapping, checkpoint values and costs.
- `results/fig6/results.json`: all per-seed summaries.
- `results/fig6/summary.csv`: 48-point cost summaries.
- `results/fig6/baseline_comparison.csv`: relative improvements against the baseline table.
- `results/fig6/fig06.*`: benchmark comparison in PNG, PDF and SVG formats.

The runner writes the same result organization under its `--output` directory.
Newly computed records also include input/configuration/source fingerprints.
Execution timings describe individual runs and are not a cross-platform runtime claim.

## Baseline utilities

`gate_cover.py` and `gate_cover_randominit.py` expose gate-cover utilities;
`metis_like.py` provides graph partitioning helpers and `stitch.py` provides partition
stitching. These modules are separate auxiliary interfaces, not a unified runner for
all reported baselines. SA is represented by comparison data, not an implementation.

The scheduler uses symmetric CX encoding. GateCover uses directed encoding:
control stores the target index; target stores the control index plus 10000.
Its reader preserves actual control/target orientation. Do not interchange these
matrix representations. GateCover's directed encoding requires width below 10000.

The main JSON output contains total, Telegate and Teledata cost, selected mapping,
seed, checkpoint values, run totals, configuration and input SHA256. Integer inputs
and valid symmetric interaction matrices are required by the scheduler API.
