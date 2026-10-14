"""Section 67: where adaptive resolution stops paying. Speedup of the focused grid against uniform PF as the active
region grows, at equal accuracy (case 1 physics, NumPy solver, one thread, best of three)."""
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
from bmvr.adaptivity import focused_grid,transfer

LENGTH,H,DT,STEPS=128.,.5,.02,200
RADII=(6.,12.,18.,24.,32.,40.,48.,56.)


def timed(grid,v):
    x=v.copy();start=perf_counter()
    while perf_counter()-start<.5:x=ac_step(grid,x,DoubleWell(),DT)  # sustained warm-up (Section 55)
    best=None
    for _ in range(3):
        x=v.copy();start=perf_counter()
        for _ in range(STEPS):x=ac_step(grid,x,DoubleWell(),DT)
        t=perf_counter()-start;best=t if best is None or t<best else best
    return best,x


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);uniform=Grid.uniform(LENGTH,int(LENGTH/H));rows=[]
    for R in RADII:
        b=min(R+10.,LENGTH/2);adaptive=focused_grid(LENGTH,H,((-b,b),(-b,b)))
        tu,vu=timed(uniform,circle(uniform,R));ta,va=timed(adaptive,circle(adaptive,R))
        mapped=transfer(adaptive,uniform,va);err=float(np.sqrt(uniform.integrate((mapped-vu)**2)/uniform.integrate(vu**2)))
        g=2*b/LENGTH;rows.append(dict(radius=R,band_half_width=b,fine_axis_fraction=g,active_fraction=g*g,predicted_cell_ratio=1/(g+(1-g)/4)**2,uniform_cells=uniform.size,adaptive_cells=adaptive.size,
            cell_ratio=uniform.size/adaptive.size,speedup=tu/ta,relative_l2=err,uniform_seconds=tu,adaptive_seconds=ta))
        print(R,rows[-1]['active_fraction'],round(rows[-1]['cell_ratio'],2),round(rows[-1]['speedup'],2),err,flush=True)
    efficiency=[r['speedup']/r['cell_ratio'] for r in rows]
    gates=dict(C1_equal_accuracy=all(r['relative_l2']<=1e-6 for r in rows),C2_monotone_crossover=all(rows[i]['speedup']>=rows[i+1]['speedup']-.05 for i in range(len(rows)-1)) and rows[-1]['speedup']<=1.2,
        C3_speedup_tracks_cells=all(.5<=e<=1.2 for e in efficiency))
    report=dict(scope='Section 67 crossover of adaptive advantage with active fraction; declared gates.',rows=rows,efficiency=efficiency,gates=gates,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),threads=1))
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item())+'\n');print(gates,flush=True)


if __name__=='__main__':main()
