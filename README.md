# LABUBU

Research code for **Labubu: Layer-Buffered Bundled Optimization for Efficient Remote
Gate Scheduling in Distributed Quantum Computing**, by Xu Xu, Yu Liu, Yingling Mao,
and Yuanyuan Yang, *IEEE Transactions on Networking*, vol. 34, pp. 4763–4778, 2026.
[Paper](https://doi.org/10.1109/TON.2026.3685555).

LABUBU minimizes entanglement consumption under a hybrid Telegate/Teledata model.
It combines coordinate-wise pruned greedy mapping (CPG), bounded perturbation (BP),
and layer-buffered capacity repair. The project provides the scheduler, QFT/RD
benchmark inputs, a Steane interaction expansion, parallel experiment execution,
and the result tables and plotting scripts for the 48-point benchmark grid.

## Installation

Python 3.10 or later is required. From the repository root:

```bash
python -m pip install -e '.[baselines,figures]'
```

NumPy is sufficient for the scheduler; the optional dependencies support baseline
utilities and experiment analysis. `requirements-reproduction.txt` specifies a
Python 3.12 environment for the experiments.

## Run a circuit

```bash
labubu examples/qft10.qasm --capacity 5 --qpus 3 --runs 18 --workers 8 --output outputs/qft10.json
labubu examples/clifford4.qasm --capacity 25 --qpus 2 --steane --iterations 100 --output outputs/encoded.json
```

QPU count is explicit. Initialization places each CX pair together and reserves one
slot per QPU; pair packing can require more QPUs than `ceil(width/capacity)`.
`--workers 0` uses available CPUs; `--workers 1` runs serially. Independent runs
use seeds `seed + run_index`. A run updates layers in sequence.

The reader supports a single-register OpenQASM 2 subset with one statement per line:
CX and single-qubit gates. Measurements, barriers and custom gate definitions require
preprocessing. Single-qubit gates advance local clocks. Steane mode accepts named
Clifford gates (`h`, `s`, `sdg`, Pauli gates, `id`, `cx`); parameterized Clifford gates
must first be decomposed into this gate set.

## Benchmark experiments

The grid contains QFT/RD circuits of widths 30–100 in steps of 10 and QPU capacities
5, 15 and 25. Each point uses 18 seeds, 1000 iterations, BP interval 50, repair budget
500 and one reserved initialization slot per QPU.

```bash
python experiments/verify_results.py
python experiments/plot_fig6.py
python experiments/fig6.py --runs 18 --workers 8
python experiments/plot_fig6.py --results-dir outputs/fig6
```

`results/fig6/` contains the experiment records; new command output is written to
`outputs/fig6/`. The runner computes LABUBU and evaluates it against the fixed baseline
table in `results/baselines.csv`. Baseline helper modules have separate interfaces;
there is no unified runner for generating all methods in that table.

For a smaller experiment:

```bash
python experiments/fig6.py --types RD --widths 30 --capacities 15 --runs 2 --iterations 10 --workers 2
```

Cached runs are identified by input, configuration and source hashes. Use `--force`
to recompute. Changes to the iteration budget or seed count change the experiment
configuration. See [experiment settings](docs/EXPERIMENTS.md) and
[data formats](docs/DATA.md).

## Native backend

Python is the default backend. The optional C++ backend uses the same candidate
ordering and Python-compatible random-choice sequence. On Linux:

```bash
g++ -O3 -std=c++17 -shared -fPIC experiments/native_engine.cpp -o experiments/liblabubu_native.so
python experiments/verify_native.py
python experiments/fig6.py --backend native --runs 18 --workers 8
```

The build command targets Linux; use Python on other platforms.

## Project structure

| Path | Contents |
|---|---|
| `src/labubu/` | Scheduler, cost model, QASM reader, Steane interaction expansion and CLI |
| `src/labubu/baselines/` | Gate-cover and partition-stitching utilities |
| `benchmarks/` | 16 fixed QFT/RD inputs, SHA256 values and QPU configurations |
| `examples/` | Small QFT and Clifford examples |
| `experiments/` | Grid execution, validation, plotting and native backend |
| `results/` | Per-seed experiment records, summary tables, baseline costs and figures |
| `docs/` | Method, experiment configuration and data documentation |
| `tests/` | Algorithm and interface regression tests |
| `.github/workflows/` | Linux/Windows test configuration |

```bash
python -m unittest discover -s tests -v
```

The model counts remote CX operations and inter-layer qubit migrations. The Steane
extension represents interaction load; it does not simulate states or physical noise.
The experiment scope and checkpoint selection are specified in [method](docs/METHOD.md).

## Citation

Citation metadata is provided in `CITATION.cff`. Please cite the paper when using
LABUBU in research.

## License

This project is licensed under the [MIT License](LICENSE).
Third-party dependencies retain their respective licenses.
