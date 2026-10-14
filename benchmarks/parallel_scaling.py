"""Section 55: measured strong and weak thread scaling of uniform and adaptive grids."""
import argparse,json,os,platform
from pathlib import Path
from time import perf_counter
import numpy as np
import numba
import scipy
from bmvr.grid import Grid,circle
from bmvr.energy import DoubleWell
from bmvr.adaptivity import focused_grid,transfer
from bmvr.solvers import ac_step as numpy_ac_step
from bmvr import parallel

ROOT=Path(__file__).resolve().parents[1]
THREADS=(1,2,4,8)
H=.5
RADIUS=14.
STEPS=200
REPEATS=5
DT=.02


def per_step(step,state,steps=None,repeats=REPEATS,warmup=.5,target=.6):
    # macOS places short bursts on efficiency cores: sustain load before timing and size
    # each repeat to about `target` seconds so every configuration is timed after promotion.
    v=state.copy();start=perf_counter();count=0
    while perf_counter()-start<warmup:v=step(v);count+=1
    steps=steps or max(20,int(count*target/warmup))
    times=[]
    for _ in range(repeats):
        v=state.copy();start=perf_counter()
        for _ in range(steps):v=step(v)
        times.append((perf_counter()-start)/steps)
    return float(np.median(times)),float(min(times)),float(max(times))


def ac_grids(length):
    return dict(uniform=Grid.uniform(length,int(round(length/H))),adaptive=focused_grid(length,H,((-25,25),(-25,25))))


def stepper(grid):
    # Allocation-free explicit step alternating between two buffers.
    model=DoubleWell();buf=[np.empty(grid.shape),np.empty(grid.shape)];k=[0]
    def step(x):
        k[0]^=1;return parallel.ac_step(grid,x,model,DT,out=buf[k[0]])
    return step


def ac_series(grid,threads):
    v=circle(grid,RADIUS);rows={}
    for p in threads:
        parallel.set_threads(p);rows[str(p)]=dict(zip(('median','min','max'),per_step(stepper(grid),v)))
    return rows


def ch_series(n,threads):
    grid=Grid.uniform(float(n),n,periodic=True);rng=np.random.default_rng(1701);c=.05*rng.standard_normal(grid.shape);lam=grid.spectrum();rows={}
    for p in threads:
        rows[str(p)]=dict(zip(('median','min','max'),per_step(lambda x:parallel.ch_step(grid,x,.5,symbol=lam,workers=p),c)))
    return rows


def efficiency(rows,cells=None):
    t1=rows['1']['median'];out={}
    for p,r in rows.items():
        k=int(p);work=1. if cells is None else cells[p]/cells['1']/k
        out[p]=dict(speedup=t1/r['median'],efficiency=t1/r['median']/k) if cells is None else dict(efficiency=work*t1/r['median'])
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);model=DoubleWell()
    # One-time JIT compilation is measured and excluded from per-step times (brief section 17).
    g=Grid.uniform(8.,16);start=perf_counter();parallel.ac_step(g,circle(g,2.),model,DT);compile_s=perf_counter()-start
    # Kernel equivalence on the study grids.
    L=512.;grids=ac_grids(L);equivalence={}
    for name,grid in grids.items():
        v=circle(grid,RADIUS);ref=numpy_ac_step(grid,v,model,DT);res=[]
        for p in THREADS:parallel.set_threads(p);res.append(parallel.ac_step(grid,v,model,DT))
        equivalence[name]=dict(max_difference_to_numpy=float(max(abs(r-ref).max() for r in res)),thread_invariant=all(np.array_equal(res[0],r) for r in res))
    # Single-thread NumPy reference timing of the original solver, for context only.
    numpy_rows={name:dict(zip(('median','min','max'),per_step(lambda x,g=grid:numpy_ac_step(g,x,model,DT),circle(grid,RADIUS),repeats=3))) for name,grid in grids.items()}
    strong={name:ac_series(grid,THREADS) for name,grid in grids.items()}
    ratio={p:strong['uniform'][p]['median']/strong['adaptive'][p]['median'] for p in strong['uniform']}
    # Accuracy of the two discretizations over the timed interval (t=4): same physics, h=0.5 in the active region.
    parallel.set_threads(8);fields={}
    for name,grid in grids.items():
        v=circle(grid,RADIUS)
        for _ in range(STEPS):v=parallel.ac_step(grid,v,model,DT)
        fields[name]=(grid,v)
    gu,vu=fields['uniform'];ga,va=fields['adaptive'];mapped=transfer(ga,gu,va)
    accuracy=dict(relative_l2=float(np.sqrt(gu.integrate((mapped-vu)**2)/gu.integrate(vu**2))),linf=float(abs(mapped-vu).max()))
    weak_n={'1':512,'2':724,'4':1024,'8':1448};weak={}
    for p in THREADS:
        n=weak_n[str(p)];grid=Grid.uniform(n*H,n);parallel.set_threads(p)
        weak[str(p)]=dict(n=n,cells=n*n,**dict(zip(('median','min','max'),per_step(stepper(grid),circle(grid,RADIUS)))))
    ch_strong=ch_series(1024,THREADS)
    ch_weak_n={'1':512,'2':720,'4':1024,'8':1440};ch_weak={}
    for p in THREADS:
        n=ch_weak_n[str(p)];r=ch_series(n,(p,))[str(p)];ch_weak[str(p)]=dict(n=n,cells=n*n,**r)
    report=dict(scope='Section 55: thread scaling of the unchanged AC finite-volume step (uniform PF and focused adaptive grid) and the spectral CH step. Identical arithmetic across thread counts; not a distributed-memory measurement.',
        threads=THREADS,h=H,radius=RADIUS,dt=DT,timing='0.5 s sustained warm-up, then 5 repeats of about 0.6 s each; median per step',repeats=REPEATS,compile_seconds=compile_s,equivalence=equivalence,
        grids={k:dict(shape=list(g.shape),cells=g.size) for k,g in grids.items()},numpy_single_thread=numpy_rows,
        ac_strong=strong,ac_strong_scaling={k:efficiency(v) for k,v in strong.items()},adaptive_over_uniform_speedup=ratio,accuracy_t4=accuracy,
        ac_weak=weak,ac_weak_efficiency=efficiency(weak,{p:w['cells'] for p,w in weak.items()}),
        ch_strong=ch_strong,ch_strong_scaling=efficiency(ch_strong),ch_weak=ch_weak,ch_weak_efficiency=efficiency(ch_weak,{p:w['cells'] for p,w in ch_weak.items()}),
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,platform=platform.platform(),
            cpu_count=os.cpu_count(),threading_layer=numba.threading_layer()))
    (out/'report.json').write_text(json.dumps(report,indent=1)+'\n')
    print(json.dumps(dict(strong=report['ac_strong_scaling'],ratio=ratio,weak=report['ac_weak_efficiency'],ch=report['ch_strong_scaling'],chw=report['ch_weak_efficiency'],acc=accuracy),indent=0),flush=True)


if __name__=='__main__':main()
