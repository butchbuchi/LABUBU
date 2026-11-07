"""ctypes backend preserving Python random.Random.choice trajectories."""
from pathlib import Path
import ctypes as C
import random
import numpy as np
P=Path(__file__).resolve().parent
lib=C.CDLL(str(P/'liblabubu_native.so'))
f=lib.labubu_run
PI=C.POINTER(C.c_int32);PU=C.POINTER(C.c_uint32);PL=C.POINTER(C.c_int64)
f.argtypes=[C.c_int]*6+[PI,PI,PI,PU,C.c_int,PI,PL,PI];f.restype=C.c_int

def optimize_native(layers,capacities,initial_mapping,iterations=1000,interval=50,seed=0,max_repair=500):
 g=np.ascontiguousarray(layers,dtype=np.int32);initial=np.ascontiguousarray(initial_mapping,dtype=np.int32);caps=np.ascontiguousarray(capacities,dtype=np.int32)
 if g.ndim!=2 or min(g.shape)<1 or initial.shape!=g.shape or caps.ndim!=1 or len(caps)<1:
  raise ValueError('Invalid interaction matrix, initial mapping or capacity shape')
 if iterations<1 or interval<1 or max_repair<1 or np.any(caps<1) or np.any(initial<0) or np.any(initial>=len(caps)):
  raise ValueError('Invalid iteration parameters, capacities or QPU IDs')
 if np.any(g < -1) or np.any(g >= g.shape[1]):
  raise ValueError('Invalid interaction partner')
 if not all(np.all(np.bincount(row,minlength=len(caps))<=caps) for row in initial):
  raise ValueError('Initial mapping exceeds capacity')
 state=random.Random(seed).getstate()[1];mt=np.array(state[:-1],dtype=np.uint32)
 out=np.empty_like(g);costs=np.empty(iterations+1,dtype=np.int64);markers=np.empty(iterations+1,dtype=np.int32)
 n=f(*g.shape,len(caps),iterations,interval,max_repair,g.ctypes.data_as(PI),initial.ctypes.data_as(PI),caps.ctypes.data_as(PI),mt.ctypes.data_as(PU),state[-1],out.ctypes.data_as(PI),costs.ctypes.data_as(PL),markers.ctypes.data_as(PI))
 if n<1:raise RuntimeError('Native execution failed')
 tg=sum(int(out[t,q]!=out[t,p]) for t,row in enumerate(g) for q,p in enumerate(row) if p>q)
 td=int(np.count_nonzero(out[1:]!=out[:-1]));checks=[{('before_iteration' if marker>0 else 'after_iteration'):abs(int(marker)),'cost':int(value)} for marker,value in zip(markers[:n],costs[:n])]
 return {'total':tg+td,'telegate':tg,'teledata':td,'mapping':out.tolist(),'seed':seed,'checkpoints':checks}
