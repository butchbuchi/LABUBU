import argparse
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import hashlib
from pathlib import Path
from . import read_qasm, optimize
from .encoding import FT_circuit_func
from .parallel import run_seed


def available_cpus():
    """Respect CPU affinity and Linux cgroup v2 CPU quota when available."""
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, count)



def main():
    parser = argparse.ArgumentParser(description="Run LABUBU on a pre-decomposed single-register QASM circuit")
    parser.add_argument("circuit", type=Path)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--qpus", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0, help="Worker processes; 0=automatic, 1=serial")
    parser.add_argument("--idle", type=int, default=1)
    parser.add_argument("--steane", action="store_true")
    parser.add_argument("--qec-period", type=int, default=5)
    parser.add_argument("--cpg-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    if args.qpus < 1 or args.runs < 1 or args.qec_period < 1:
        parser.error("qpus, runs and qec-period must be positive")
    layers = read_qasm(args.circuit, clifford_only=args.steane)
    if args.steane:
        layers = FT_circuit_func(layers, args.qec_period)
    workers = min(args.runs, args.workers or available_cpus())
    run = partial(optimize, layers, [args.capacity] * args.qpus,
                  args.iterations, args.interval, idle=args.idle,
                  cpg_only=args.cpg_only)
    seeds = range(args.seed, args.seed + args.runs)
    if workers == 1:
        results = [run(seed=seed) for seed in seeds]
    else:
        # Spawn supports Windows and avoids inheriting native-library state.
        # map preserves seed order, including deterministic tie selection.
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(run_seed, ((run, seed) for seed in seeds)))
    result = min(results, key=lambda r: r["total"])
    result["runs"] = [{k: r[k] for k in ["seed", "total", "telegate", "teledata"]} for r in results]
    result["execution"] = {"workers": workers, "start_method": "spawn" if workers > 1 else "serial"}
    result['input_sha256'] = hashlib.sha256(args.circuit.read_bytes()).hexdigest()
    result["parameters"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    result["circuit"] = {"depth": len(layers), "width": layers.shape[1], "cnots": int((layers >= 0).sum() // 2)}
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
