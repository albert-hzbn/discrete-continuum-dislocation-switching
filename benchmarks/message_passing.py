"""Section 61.2: message-passing strip decomposition, strong and weak scaling, bitwise check against serial."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform
from pathlib import Path
from time import perf_counter
import numpy as np
from bmvr.grid import Grid,circle
from bmvr.energy import DoubleWell
from bmvr.solvers import ac_step
from bmvr.adaptivity import focused_grid
from bmvr import decomposition as D

P_LIST=(1,2,4,8)
H=.5;DT=.02


def serial(grid,v,steps):
    for _ in range(steps):v=ac_step(grid,v,DoubleWell(),DT)
    return v


def measure(grid,parts,per_cell):
    est=per_cell*grid.size/parts;warmup=max(20,int(.5/est));steps=max(30,int(.6/est))
    v0=circle(grid,14.);best=None
    for _ in range(3):
        t,field=D.run(grid,v0,parts,steps,warmup,DT);best=t if best is None else min(best,t)
    ok=bool(np.array_equal(field,serial(grid,v0,warmup+steps)))
    return dict(parts=parts,cells=grid.size,warmup=warmup,steps=steps,seconds_per_step=best,bitwise=ok)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    g=Grid.uniform(256.,512);v=circle(g,14.);start=perf_counter();serial(g,v,40);per_cell=(perf_counter()-start)/40/g.size
    uniform=Grid.uniform(512.,1024);adaptive=focused_grid(512.,H,((-25,25),(-25,25)))
    strong={name:[measure(grid,p,per_cell) for p in P_LIST] for name,grid in (('uniform',uniform),('adaptive',adaptive))}
    weak=[]
    for p in P_LIST:
        n=int(round(512*np.sqrt(p)));weak.append(dict(n=n,**measure(Grid.uniform(n*H,n),p,per_cell)))
    for rows in strong.values():
        for r in rows:r['speedup']=rows[0]['seconds_per_step']/r['seconds_per_step'];r['efficiency']=r['speedup']/r['parts']
    for r in weak:r['efficiency']=(weak[0]['seconds_per_step']/weak[0]['cells'])/(r['seconds_per_step']/(r['cells']/r['parts']))
    ratio={str(p):strong['uniform'][i]['seconds_per_step']/strong['adaptive'][i]['seconds_per_step'] for i,p in enumerate(P_LIST)}
    gates=dict(D1_bitwise=all(r['bitwise'] for rows in list(strong.values())+[weak] for r in rows))
    report=dict(scope='Section 61.2 message-passing strips on one node (pipes between processes, private memory, one thread each); not a network measurement.',
        serial_ns_per_cell=per_cell*1e9,strong=strong,weak=weak,adaptive_over_uniform=ratio,gates=gates,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),cpu_count=os.cpu_count()))
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item())+'\n')
    print(gates,{k:[(r['parts'],round(r['speedup'],2)) for r in v] for k,v in strong.items()},[(r['parts'],round(r['efficiency'],2)) for r in weak],ratio,flush=True)


if __name__=='__main__':main()
