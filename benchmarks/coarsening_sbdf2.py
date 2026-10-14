"""Section 79: late CH coarsening exponents with the second-order SBDF2 integrator; declared gates X2, X3 (X1 in the report)."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from bmvr.grid import Grid
from bmvr.energy import DoubleWell
from bmvr.solvers import ch_sbdf2
from bmvr.diagnostics import structure_factor
from bmvr.coarsening import initial_coarsening_field,coarsening_windows

END=4096.;L=256.;H=.5;SEEDS=[1701+1000*k for k in range(16)];DTS=(.25,.125)


def member(args):
    out,seed,dt=args;f=Path(out)/f'member-{dt:g}-{seed}.npy'
    if f.exists():return str(f)
    n=round(L/H);grid=Grid.uniform(L,n,periodic=True);model=DoubleWell();sym=grid.spectrum();c=initial_coarsening_field(L,n,seed);prev=None
    every=round(16/dt);rows=[]
    def record(t):
        e=model.value(grid,c);_,ell=structure_factor(grid,c);rows.append([t,ell,2*np.sqrt(2)/3*grid.area/e])
    record(0.)
    for step in range(1,round(END/dt)+1):
        c,prev=ch_sbdf2(grid,c,prev,dt,symbol=sym),c
        if step%every==0:record(step*dt)
    np.save(f,np.array(rows));return str(f)


def assemble(out):
    data={dt:np.array([np.load(out/f'member-{dt:g}-{s}.npy') for s in SEEDS]) for dt in DTS};t=data[DTS[0]][0][:,0]
    windows={dt:[coarsening_windows(t,v[:,1],v[:,2],L) for v in arr] for dt,arr in data.items()}
    def exponent(dt,start,kind):
        vals=[next(w for w in ws if w['start']==start)[kind+'_exponent'] for ws in windows[dt]];return float(np.mean(vals)),float(np.std(vals,ddof=1)/np.sqrt(len(vals)))
    ex={f'{dt:g}':{f'{s}':{k:exponent(dt,s,k) for k in ('spectral','energy')} for s in (512,1024,2048)} for dt in DTS}
    diffs={f'{s}':{k:ex['0.125'][f'{s}'][k][0]-ex['0.25'][f'{s}'][k][0] for k in ('spectral','energy')} for s in (512,1024,2048)}
    m,se=ex['0.125']['2048']['energy']
    gates=dict(X2_exponent_stability=all(abs(v)<=.02 for d in diffs.values() for v in d.values()),X3_energy_one_third=bool(abs(m-1/3)<=2*se))
    (out/'report.json').write_text(json.dumps(dict(scope='Section 79 SBDF2 coarsening ensemble.',exponents=ex,differences=diffs,gates=gates,seeds=SEEDS),indent=1)+'\n');print(gates,diffs,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);jobs=[(str(out),s,dt) for dt in DTS[::-1] for s in SEEDS]
    with Pool(args.workers) as pool:
        for f in pool.imap_unordered(member,jobs):print('done',f,flush=True)
    assemble(out)
