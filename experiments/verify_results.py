"""Independently validate the 864 bundled mappings, costs, seeds and input hashes."""
from pathlib import Path
import hashlib
import json
import sys
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from labubu import read_qasm

manifest=json.loads((ROOT/'benchmarks/manifest.json').read_text())
for name,digest in manifest['inputs_sha256'].items():
    raw = (ROOT/name).read_bytes()
    # Git may convert text line endings between Windows and Linux.
    # Accept only byte-identical input or an LF/CRLF equivalent.
    lf = raw.replace(b'\r\n', b'\n')
    variants = (raw, lf, lf.replace(b'\n', b'\r\n'))
    if not any(hashlib.sha256(value).hexdigest() == digest for value in variants):
        raise RuntimeError(f'Input hash mismatch: {name}')
groups={};cache={};total=0
for path in sorted((ROOT/'results/fig6/runs').glob('*.json')):
    data=json.loads(path.read_text());s=data['summary'];r=data['result']
    typ,w,cap,seed=s['type'],s['width'],s['capacity'],s['seed']
    key=typ,w
    if key not in cache:
        g=read_qasm(ROOT/'benchmarks'/typ.lower()/f'{typ.lower()}{w}.qasm')
        t,q=np.nonzero(g>np.arange(w))
        cache[key]=g,t,q,g[t,q]
    g,t,q,p=cache[key];m=np.asarray(r['mapping'])
    if m.shape!=g.shape or np.any(m<0) or np.any(m>=s['qpus']):
        raise RuntimeError(f'Invalid mapping: {path.name}')
    telegate=int(np.count_nonzero(m[t,q]!=m[t,p]))
    teledata=int(np.count_nonzero(m[1:]!=m[:-1]))
    if (r['telegate'],r['teledata'],r['total'])!=(telegate,teledata,telegate+teledata):
        raise RuntimeError(f'Cost mismatch: {path.name}')
    if s['total']!=r['total'] or min(c['cost'] for c in r['checkpoints'])!=r['total']:
        raise RuntimeError(f'Summary/checkpoint mismatch: {path.name}')
    if any(max(Counter(row).values())>cap for row in m):
        raise RuntimeError(f'Capacity violation: {path.name}')
    groups.setdefault((typ,w,cap),[]).append(seed);total+=1
if len(groups)!=48 or total!=864 or any(sorted(v)!=list(range(18)) for v in groups.values()):
    raise RuntimeError('Grid/seeds incomplete')
print('PASS: 16 input hashes; 48 points; 864 mappings/costs/capacity checks; seeds 0–17.')
