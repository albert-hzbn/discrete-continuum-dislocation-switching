"""Section 66: dynamic crack branching with explicit elastodynamics and AT2 damage; declared gates."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform
from pathlib import Path
from time import perf_counter
import numpy as np
from bmvr.dynamic_fracture import DynamicFracture2D,ThermalDynamicFracture2D
from bmvr.fracture2d import FractureParameters

N,DT,END=200,.001,1.5
LOADS=(.03,.06,.09)
LINES=(.65,.75,.85)
WIDE_LINES=(.9,1.,1.1,1.2,1.3,1.4)  # Section 66.4 post hoc 2 x 1 domain
C_R=.93


def run(sigma,out,width=1,thermal=False,n=N,end_time=None):
    cls=ThermalDynamicFracture2D if thermal else DynamicFracture2D
    model=cls(n,FractureParameters(ell=.015),traction=sigma,nx=n*width);s=model.initial();start=perf_counter()
    lines,end,stop=(LINES,END,model.base.width-model.base.h) if width==1 else (WIDE_LINES,3.,model.base.width-.05)
    end=end if end_time is None else end_time
    Hof=lambda e:e['kinetic']+e['elastic']+(e['heat'] if thermal else e['fracture'])-e['work_potential']  # Section 81: fracture energy becomes heat
    e0=model.energies(s);H0=Hof(e0);rows=[];snaps={};min_inc=0.;k=0;minT=np.inf
    while s['time']<end-1e-12:
        prev=s['q'];s,info=model.step(s,DT);k+=1;min_inc=min(min_inc,float((s['q']-prev).min()))
        if k%10==0:
            e=model.energies(s);H=Hof(e)
            if thermal:minT=min(minT,float(s['T'].min()))
            rows.append(dict(time=s['time'],tip=model.centerline_tip(s['q']),H=H,mechanical=e['kinetic']+e['elastic'],fracture=e['fracture'],
                branches={f'{x:g}':model.branches(s['q'],x) for x in lines},q_max=float(s['q'].max()),q_min=float(s['q'].min()),
                T_max=float(s['T'].max()) if thermal else None))
        if k%100==0:
            snaps[f"{s['time']:.3f}"]=s['q'].astype(np.float32);print(sigma,f"t={s['time']:.3f}",f"tip={rows[-1]['tip']:.3f}",rows[-1]['branches'],flush=True)
        if rows and rows[-1]['tip']>=stop:break
    np.savez_compressed(out/f'sigma-{sigma:g}.npz',**snaps,final=s['q'].astype(np.float32),**({'T':s['T'].astype(np.float32)} if thermal else {}))
    t=np.array([r['time'] for r in rows]);tip=np.array([r['tip'] for r in rows])
    speed=np.array([(tip[i+5]-tip[i])/(t[i+5]-t[i]) for i in range(len(t)-5)]) if len(t)>5 else np.zeros(0)
    peak=max(r['mechanical'] for r in rows);drift=max(abs(r['H']-H0) for r in rows)/peak
    reached=[r for r in rows if r['tip']<stop]
    max_bands={x:max(r['branches'][x] for r in reached) if reached else 0 for x in (f'{v:g}' for v in lines)}
    result=dict(sigma=sigma,width=width,thermal=thermal,n=n,min_T=minT if thermal else None,steps=k,end_time=s['time'],max_speed=float(speed.max()) if len(speed) else 0.,energy_drift_fraction=drift,
        min_increment=min_inc,q_bounds=[min(r['q_min'] for r in rows),max(r['q_max'] for r in rows)],max_bands_before_edge=max_bands,
        first_branch_time=next((r['time'] for r in rows if any(v>=2 for v in r['branches'].values())),None),rows=rows,seconds=perf_counter()-start)
    if width>1:
        early=[r for r in rows if r['tip']<=1.7];result['K4_branch_away_from_edge']=any(v>=2 for r in early for v in r['branches'].values())
        result['first_early_branch']=next((dict(time=r['time'],tip=r['tip'],branches=r['branches']) for r in early if any(v>=2 for v in r['branches'].values())),None)
    (out/f'sigma-{sigma:g}.json').write_text(json.dumps(result,indent=1,default=lambda o:o.item())+'\n');print(sigma,'done',max_bands,result['max_speed'],drift,flush=True)


def assemble(out):
    r={s:json.loads((out/f'sigma-{s:g}.json').read_text()) for s in LOADS}
    bands=lambda s:max(r[s]['max_bands_before_edge'].values())
    gates=dict(K1_bounds=all(x['min_increment']>=-1e-12 and x['q_bounds'][0]>=0 and x['q_bounds'][1]<=1 and x['max_speed']<=C_R for x in r.values()),
        K2_energy=all(x['energy_drift_fraction']<=.05 for x in r.values()),K3_branching=bands(.09)>=2 and bands(.03)<=1)
    report=dict(scope='Section 66 dynamic branching; synthetic nondimensional parameters; declared gates.',n=N,dt=DT,c_R=C_R,
        runs={s:{k:v for k,v in x.items() if k!='rows'} for s,x in r.items()},gates=gates,environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()))
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item())+'\n');print(gates,{s:(bands(s),r[s]['max_speed'],r[s]['energy_drift_fraction']) for s in LOADS},flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--sigma',type=float)
    parser.add_argument('--width',type=int,default=1,help='domain width in heights (Section 66.4 post hoc uses 2)')
    parser.add_argument('--thermal',action='store_true',help='Section 81: fracture heat and thermal softening')
    parser.add_argument('--n',type=int,default=N,help='cells per unit height (Section 83 refinement: 150, 300)')
    parser.add_argument('--end',type=float,help='end time (Section 83 amendment: n = 300 stops at 1.5)');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    run(args.sigma,out,args.width,args.thermal,args.n,args.end) if args.sigma else assemble(out)


if __name__=='__main__':main()
