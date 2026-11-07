"""Check optional native backend against Python on deterministic random cases."""
from pathlib import Path
import sys
import random
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from labubu import optimize
from labubu.scheduler import initialize
from native_engine import optimize_native
rng=random.Random(17)
for seed in range(30):
    g=np.full((7,8),-1,dtype=int)
    for row in g:
        order=list(range(8));rng.shuffle(order)
        for i in range(0,6,2):
            a,b=order[i:i+2];row[a]=b;row[b]=a
    caps=[5]*4;initial=initialize(g,caps,1)
    expected=optimize(g,caps,iterations=300,seed=seed,initial_mapping=initial)
    actual=optimize_native(g,caps,initial,iterations=300,seed=seed)
    if actual!=expected:
        raise RuntimeError(f'Native/Python mismatch for seed {seed}')
print('PASS: native/Python full mappings, checkpoints and costs identical in 30 cases.')
