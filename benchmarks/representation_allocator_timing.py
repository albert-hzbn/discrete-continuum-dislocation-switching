"""Sections 92.2 and 92.3 post hoc: allocator sweep costs. --cpu (92.3) measures process CPU time, one run per configuration;
otherwise (92.2) wall time, three interleaved repeats, minimum kept."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json,sys,time
from pathlib import Path
import numpy as np
from bmvr.representation_allocator import GlideEnsemble,Run
from representation_allocator import ensemble,G,TAU
from representation_allocator_control import NC,OUT

if __name__=='__main__':
    x,y,s,ids=ensemble(92);probe=GlideEnsemble(G,b=1.,tau=TAU);b0=.5*TAU/float(np.abs(probe.stress(probe.deposit(x,y,s.astype(float)))).max());M=1/(b0*TAU)
    cpu='--cpu' in sys.argv;clock=time.process_time if cpu else time.time
    configs=[('D',64),('C',64)]+[('H',n) for n in NC];times={f'{m}{n if m=="H" else ""}':[] for m,n in configs}
    for rep in range(1 if cpu else 3):
        for m,n in (configs if rep%2==0 else configs[::-1]):
            e=GlideEnsemble(G,b=b0,M=M,tau=TAU,n_c=n);r=Run(e,x,y,s,ids,m);dt=.25*e.h/(M*b0*TAU);start=clock()
            for _ in range(200):r.step(dt)
            times[f'{m}{n if m=="H" else ""}'].append((clock()-start)/200);print(rep,m,n,times[f'{m}{n if m=="H" else ""}'][-1],flush=True)
    best={k:min(v) for k,v in times.items()};h=[best[f'H{n}'] for n in NC]
    mono=all(b>=.9*a for a,b in zip(h,h[1:]))
    (OUT/('cpu-time-post-hoc.json' if cpu else 'timing-post-hoc.json')).write_text(json.dumps(dict(times=times,minimum=best,E1_cost_post_hoc=mono),indent=1)+'\n');print(best,mono)
