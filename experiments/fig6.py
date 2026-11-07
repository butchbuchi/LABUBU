"""Run fixed Fig.6 grid with independent seeds and content-addressed caches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Supports a source checkout as well as an installed package.
sys.path.insert(0, str(ROOT / 'src'))
from labubu import read_qasm, optimize, entanglement_cost


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def job(task):
    typ, width, capacity, qpus, seed, options = task
    circuit = ROOT / 'benchmarks' / typ.lower() / f'{typ.lower()}{width}.qasm'
    config = dict(type=typ, width=width, capacity=capacity, qpus=qpus, seed=seed,
                  iterations=options['iterations'], interval=50, idle=1, max_repair=500,
                  initialization='layer_local', backend=options['backend'])
    identity = dict(config=config, input_sha256=sha(circuit),
                    source_sha256={n: sha(ROOT / n) for n in [
                        'src/labubu/scheduler.py', 'src/labubu/circuit.py',
                        'experiments/fig6.py']})
    if options['backend'] == 'native':
        identity['native_sha256'] = {n: sha(ROOT/'experiments'/n) for n in
                                   ['native_engine.cpp', 'native_engine.py', 'liblabubu_native.so']}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = Path(options['output']) / 'runs'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f'{typ}{width}_L{capacity}_seed{seed}.json'
    if output.exists() and not options['force']:
        saved = json.loads(output.read_text())
        if saved.get('fingerprint') == fingerprint:
            return saved
    layers = read_qasm(circuit)
    start = time.monotonic()
    if options['backend'] == 'native':
        from native_engine import optimize_native
        from labubu.scheduler import initialize
        result = optimize_native(layers, [capacity]*qpus,
                                 initialize(layers, [capacity]*qpus, 1),
                                 iterations=options['iterations'], seed=seed)
    else:
        result = optimize(layers, [capacity]*qpus, iterations=options['iterations'], seed=seed)
    mapping = np.array(result['mapping'])
    cost = entanglement_cost(layers, mapping)
    if any(cost[k] != result[k] for k in cost):
        raise RuntimeError('Returned cost and mapping disagree')
    if any(np.max(np.bincount(row, minlength=qpus)) > capacity for row in mapping):
        raise RuntimeError('Capacity violation')
    if result['total'] != min(c['cost'] for c in result['checkpoints']):
        raise RuntimeError('Checkpoint mismatch')
    summary = dict(config, total=result['total'], telegate=result['telegate'],
                   teledata=result['teledata'], seconds=time.monotonic()-start, valid=True)
    record = dict(fingerprint=fingerprint, identity=identity, summary=summary, result=result)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(record))
    temporary.replace(output)
    return record


def summarize(records, output):
    import pandas as pd
    baselines = pd.read_csv(ROOT/'results/baselines.csv')
    rows, gains = [], []
    groups = {}
    for record in records:
        summary = record['summary']
        groups.setdefault((summary['type'], summary['width'], summary['capacity']), []).append(summary)
    for (typ, width, capacity), runs in sorted(groups.items()):
        costs = [r['total'] for r in runs]
        row = dict(type=typ, width=width, capacity=capacity, runs=len(runs),
                   min_cost=min(costs), max_cost=max(costs), mean_cost=float(np.mean(costs)),
                   std_cost=float(np.std(costs, ddof=1)) if len(costs)>1 else 0.)
        rows.append(row)
        sub = baselines[(baselines.type==typ)&(baselines.width==width)&(baselines.capacity==capacity)]
        for baseline in sub.itertuples():
            gain = 100*(baseline.benchmark_cost-min(costs))/baseline.benchmark_cost
            gains.append(dict(row, benchmark=baseline.benchmark,
                              baseline_cost=baseline.benchmark_cost, gain_pct=gain))
    pd.DataFrame(rows).to_csv(output/'summary.csv', index=False)
    pd.DataFrame(gains).to_csv(output/'baseline_comparison.csv', index=False)
    (output/'results.json').write_text(json.dumps([r['summary'] for r in records], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--runs',type=int,default=18)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--iterations',type=int,default=1000)
    parser.add_argument('--widths',nargs='+',type=int,default=list(range(30,101,10)))
    parser.add_argument('--capacities',nargs='+',type=int,default=[5,15,25])
    parser.add_argument('--types',nargs='+',choices=['QFT','RD'],default=['QFT','RD'])
    parser.add_argument('--backend',choices=['python','native'],default='python')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/fig6')
    parser.add_argument('--force',action='store_true')
    args = parser.parse_args()
    if args.workers < 0 or args.runs < 1 or args.iterations < 1:
        parser.error('workers must be nonnegative; runs and iterations must be positive')
    from labubu.__main__ import available_cpus
    workers = min(args.workers or available_cpus(), args.runs*len(args.widths)*len(args.capacities)*len(args.types))
    manifest = json.loads((ROOT/'benchmarks/manifest.json').read_text())
    options = dict(iterations=args.iterations,backend=args.backend,force=args.force,output=str(args.output.resolve()))
    tasks=[]
    for typ in dict.fromkeys(args.types):
        for width in dict.fromkeys(args.widths):
            for capacity in dict.fromkeys(args.capacities):
                key=f'{typ}{width}_L{capacity}'
                if key not in manifest['configurations']:
                    parser.error(f'Unsupported grid point: {key}')
                qpus=manifest['configurations'][key]['qpus']
                tasks.extend((typ,width,capacity,qpus,seed,options) for seed in range(args.seed,args.seed+args.runs))
    if workers == 1:
        records=[job(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            records=[]
            for i, record in enumerate(pool.map(job,tasks),1):
                records.append(record)
                print(f'{i}/{len(tasks)} runs complete',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    summarize(records,args.output)
    print(f'Validated {len(records)} runs; results: {args.output}')

if __name__=='__main__':
    main()
