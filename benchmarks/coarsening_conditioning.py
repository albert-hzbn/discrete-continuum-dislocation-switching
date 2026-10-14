"""Section 51: conditioning of late CH coarsening and ensemble-statistics accuracy."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bmvr.grid import Grid
from bmvr.energy import DoubleWell
from bmvr.solvers import ch_step
from bmvr.diagnostics import structure_factor
from bmvr.coarsening import initial_coarsening_field,coarsening_windows,periodic_cell_resample

END=4096.
SEEDS=[1701+1000*k for k in range(8)]
DISCRETIZATIONS=dict(A=(.5,.25),B=(.25,.25),C=(.5,.125))


def lengths(grid,model,field):
    energy=model.value(grid,field);_,ell=structure_factor(grid,field)
    return ell,2*np.sqrt(2)/3*grid.area/energy


def trajectory(length,h,dt,seed,perturbation=0.,partner=False):
    """Run to END; optionally a second copy with a zero-mean unit-RMS initial perturbation of size eps."""
    n=round(length/h);grid=Grid.uniform(length,n,periodic=True);model=DoubleWell();symbol=grid.spectrum()
    field=initial_coarsening_field(length,n,seed);other=None
    if partner:
        xi=np.random.default_rng(seed+99).normal(size=field.shape);xi-=xi.mean();xi/=np.sqrt(np.mean(xi**2))
        other=field+perturbation*xi
    steps=round(END/dt);every=round(16/dt);rows=[];start=perf_counter()
    def record(t):
        ls,le=lengths(grid,model,field);row=[t,ls,le]
        if other is not None:
            lo,leo=lengths(grid,model,other);d=field-other
            row+=[float(np.sqrt(np.mean(d**2))),float(np.max(abs(d))),abs(lo/ls-1),abs(leo/le-1)]
        rows.append(row)
    record(0.)
    for step in range(1,steps+1):
        field=ch_step(grid,field,dt,symbol=symbol)
        if other is not None:other=ch_step(grid,other,dt,symbol=symbol)
        if step%every==0:record(step*dt)
    return np.array(rows),perf_counter()-start


def conditioning(out):
    result={}
    for eps in (1e-10,1e-6):
        rows,seconds=trajectory(128.,.5,.5,1701,eps,partner=True)
        np.save(out/f'conditioning-{eps:g}.npy',rows)
        result[f'{eps:g}']=dict(final_l2=float(rows[-1,3]),final_linf=float(rows[-1,4]),max_spectral_length_difference=float(rows[1:,5].max()),
            max_energy_length_difference=float(rows[1:,6].max()),time_l2_exceeds_1em3=next((float(r[0]) for r in rows if r[3]>1e-3),None),
            time_l2_exceeds_0p3=next((float(r[0]) for r in rows if r[3]>.3),None),solver_seconds=seconds)
    return result


def ensemble_member(out,label,seed):
    h,dt=DISCRETIZATIONS[label];rows,seconds=trajectory(256.,h,dt,seed)
    np.save(out/f'ensemble-{label}-{seed}.npy',rows);(out/f'ensemble-{label}-{seed}.json').write_text(json.dumps(dict(solver_seconds=seconds))+'\n')


def assemble(out,conditioning_result):
    data={label:np.array([np.load(out/f'ensemble-{label}-{seed}.npy') for seed in SEEDS]) for label in DISCRETIZATIONS}
    t=data['A'][0][:,0];late=(t>=256)
    means={label:v.mean(axis=0) for label,v in data.items()}
    def rel(a,b,col):return float(np.max(abs(means[a][late,col]/means[b][late,col]-1)))
    windows={label:[coarsening_windows(t,v[:,1],v[:,2],256.) for v in arr] for label,arr in data.items()}
    def exponent(label,start,kind):
        values=[next(w for w in ws if w['start']==start)[kind+'_exponent'] for ws in windows[label]]
        return float(np.mean(values)),float(np.std(values,ddof=1)/np.sqrt(len(values)))
    starts=(512,1024,2048)
    exponents={label:{f'{s}':{k:exponent(label,s,k) for k in ('spectral','energy')} for s in starts} for label in DISCRETIZATIONS}
    gates={'mesh_lengths':max(rel('A','B',1),rel('A','B',2))<=.025,'time_lengths':max(rel('A','C',1),rel('A','C',2))<=.025,
        'exponent_stability':all(abs(exponents['A'][f'{s}'][k][0]-exponents[o][f'{s}'][k][0])<=.02 for s in starts for k in ('spectral','energy') for o in ('B','C')),
        'scale_separation':all(w['scale_separated'] for ws in windows.values() for member in ws for w in member if w['start']>=512)}
    mean_b,se_b=exponents['B']['2048']['energy'];gates['interpretation_energy_one_third']=abs(mean_b-1/3)<=2*se_b
    report=dict(scope='Section 51 CH conditioning and ensemble statistics; same model, initial construction and stabilized scheme as Section 24.',
        conditioning=conditioning_result,seeds=SEEDS,discretizations={k:dict(h=v[0],dt=v[1]) for k,v in DISCRETIZATIONS.items()},
        mean_length_differences=dict(mesh_spectral=rel('A','B',1),mesh_energy=rel('A','B',2),time_spectral=rel('A','C',1),time_energy=rel('A','C',2)),
        standard_error_final=dict(spectral=float(data['B'][:,-1,1].std(ddof=1)/np.sqrt(8)/means['B'][-1,1]),energy=float(data['B'][:,-1,2].std(ddof=1)/np.sqrt(8)/means['B'][-1,2])),
        exponents=exponents,gates=gates,
        solver_seconds={label:[json.loads((out/f'ensemble-{label}-{seed}.json').read_text())['solver_seconds'] for seed in SEEDS] for label in DISCRETIZATIONS},
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),threads=1))
    (out/'report.json').write_text(json.dumps(report,indent=1)+'\n');print(json.dumps(dict(gates=gates,diffs=report['mean_length_differences'],B=exponents['B'])),flush=True)


POINTWISE=dict(time=[(.5,.5),(.5,.25),(.5,.125),(.5,.0625)],space=[(1.,.0625),(.5,.0625),(.25,.0625)])


def pointwise_member(out,h,dt):
    """Section 51.3: one box-128 seed-1701 trajectory; saves the final field."""
    length=128.;n=round(length/h);grid=Grid.uniform(length,n,periodic=True);symbol=grid.spectrum();model=DoubleWell()
    field=initial_coarsening_field(length,n,1701);start=perf_counter()
    for _ in range(round(END/dt)):field=ch_step(grid,field,dt,symbol=symbol)
    ls,le=lengths(grid,model,field)
    np.savez(out/f'pointwise-{h:g}-{dt:g}.npz',field=field,ls=ls,le=le,seconds=perf_counter()-start)


def pointwise_assemble(out):
    def load(h,dt):
        with np.load(out/f'pointwise-{h:g}-{dt:g}.npz') as d:return d['field'],float(d['ls']),float(d['le'])
    def restrict(field,n):
        # Spectral resampling to the coarser cell centers, the same construction as the initial fields.
        return periodic_cell_resample(field,128.,n)
    result={}
    for name,series in POINTWISE.items():
        diffs=[]
        for (h1,d1),(h2,d2) in zip(series,series[1:]):
            a,la,ea=load(h1,d1);b,lb,eb=load(h2,d2)
            if b.shape!=a.shape:b=restrict(b,a.shape[0])
            diffs.append(dict(pair=[(h1,d1),(h2,d2)],l2=float(np.sqrt(np.mean((a-b)**2))),spectral_length=abs(la/lb-1),energy_length=abs(ea/eb-1)))
        orders=[float(np.log2(x['l2']/y['l2'])) for x,y in zip(diffs,diffs[1:])]
        result[name]=dict(differences=diffs,observed_orders=orders)
    t=result['time'];result['gates']=dict(time_first_order=all(x['l2']>y['l2'] for x,y in zip(t['differences'],t['differences'][1:])) and all(.7<=o<=1.3 for o in t['observed_orders']),
        space_decreasing=all(x['l2']>y['l2'] for x,y in zip(result['space']['differences'],result['space']['differences'][1:])))
    last=t['differences'][-1];order=t['observed_orders'][-1] if t['observed_orders'] else None
    if order and order>0:
        # Successive difference ~ error of the coarser member for a convergent sequence; extrapolate to 0.05.
        result['dt_for_final_l2_0p05']=float(.0625*(0.05/last['l2'])**(1/order))
    (out/'pointwise.json').write_text(json.dumps(result,indent=1)+'\n');print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--conditioning',action='store_true');parser.add_argument('--member',nargs=2,metavar=('LABEL','SEED'))
    parser.add_argument('--pointwise',nargs=2,type=float,metavar=('H','DT'));parser.add_argument('--pointwise-assemble',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.pointwise:pointwise_member(out,*args.pointwise);return
    if args.pointwise_assemble:pointwise_assemble(out);return
    if args.conditioning:(out/'conditioning.json').write_text(json.dumps(conditioning(out),indent=1)+'\n');return
    if args.member:ensemble_member(out,args.member[0],int(args.member[1]));return
    assemble(out,json.loads((out/'conditioning.json').read_text()))


if __name__=='__main__':main()
