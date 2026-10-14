"""Section 90: dynamic crack branching in the finite-strain thermochemical model (AT1, brittle regime); gates H1-H4.
--erode (Section 90.2, post hoc): Section 68 reactive erosion of certificate-refused fully broken cells."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from bmvr.nodal_damage import NodalThermalFracture2D
from bmvr.thermofracture import ThermalFractureParameters
from bmvr.thermoplastic import ThermalPlasticParameters

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'thermochemical-branching-2026-09-28'
GRIP,ELL,MOB,DT,LINES=.09,.03,5000.,2e-4,(.4,.5,.6,.7)


def build(n):
    model=NodalThermalFracture2D(n,degree=2,preconditioned=True,transport=ThermalPlasticParameters(yield_stress=(100.,100.)),
        fracture=ThermalFractureParameters(ell=ELL,mobility=MOB,elastic_split='volumetric-deviatoric',crack_density='AT1'))
    m=model.mechanics.mesh;x,y=m.coordinates.T;bottom=np.flatnonzero(y==0);top=np.flatnonzero(y==1);pin=bottom[np.argmin(abs(x[bottom]-.5))]
    model.fixed=np.r_[2*bottom+1,2*top+1,2*pin];u=np.zeros(2*m.node_count);u[0::2]=-.25*GRIP*(x-.5);u[1::2]=GRIP*y;model.values=u[model.fixed].copy()
    state=model.initial(u=u,equilibrate=True,c=.3,eta=.3,T=1.,damage=lambda x,y:.95*np.exp(-((y-.5)/ELL)**2)*(1-np.tanh((x-.15)/ELL))/2)
    return model,state


def probe(q,n):
    g=q.reshape(n+1,n+1);xs=np.linspace(0,1,n+1);ys=xs;on=[xs[i] for i in range(n+1) if g[i,n//2]>=.8]
    def bands(xv):
        c=g[int(round(xv*n)),:]>=.8;return int(np.sum(c[1:]&~c[:-1])+c[0])
    return (max(on) if on else 0.),{f'{v:g}':bands(v) for v in LINES},float(g[:,abs(ys-.5)>.25].max())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,required=True);parser.add_argument('--erode',action='store_true');parser.add_argument('--band',type=float,help='Section 90.4: delete cells whose four damage nodes are all >= band');args=parser.parse_args();n=args.n
    tag=f'n{n}'+('-erode' if args.erode else '')+(f'-band{args.band:g}' if args.band else '');erosions=[];deletions=[]
    OUT.mkdir(parents=True,exist_ok=True);model,state=build(n);t=0.;rows=[];snaps={};first=model.rate(state)['diagnostics'];start=time.time();failure=None
    scale=max(first['elastic_energy']+first['chemical_energy']+first['gradient_energy']+2*abs(first['partition_energy'])+first['fracture_energy']+first['damage_binding_energy'],1e-12)
    checks=dict(positive_T=True,positive_J=True,damage_bounds=True,irreversible=True,midpoint=True);drift=0.;min_inc=0.;k=0;halvings=0
    while t<2.-1e-12:
        prev=state
        for attempt in range(9):
            model.mechanics.refused_cells=None;err=None
            for pieces in (1,2,4):
                try:
                    s=prev
                    for _ in range(pieces):s,d=model.step(s,DT/pieces)
                    err=None;break
                except (ValueError,RuntimeError) as e:err=e
            if err is None:break
            if not (args.erode and isinstance(err,ValueError)):break
            prev,event=model.erode_refused(prev)
            if event is None:break
            erosions.append(dict(time=t,cells=[int(c) for c in event['cells']],energy_moved=float(event['energy_moved']),ledger_error=float(event['ledger_error'])))
        if err is not None:failure=dict(time=t,error=str(err));break
        halvings+=pieces>1;state=s;t+=DT;prev_damage=prev.damage;k+=1;min_inc=min(min_inc,float((state.damage-prev_damage).min()))
        if args.band:
            band=(state.damage[model.damage_cells]>=args.band).all(axis=1)
            if band.any():
                active=model.mechanics.active if getattr(model.mechanics,'active',None) is not None else np.ones(len(band),bool)
                if (band&active).any():
                    state,ev=model.remove_cells(state,band)
                    if ev:deletions.append(dict(time=t,cells=len(ev['cells']),energy_moved=ev['energy_moved'],ledger_error=ev['ledger_error']))
        drift=max(drift,abs(d['total_energy']-first['total_energy'])/scale)
        checks['positive_T']&=d['min_T']>0;checks['positive_J']&=d['min_J']>0;checks['midpoint']&=d['midpoint_residual']<=1e-12
        checks['damage_bounds']&=bool(state.damage.min()>=0 and state.damage.max()<=1);checks['irreversible']&=min_inc>=-1e-14
        if k%25==0:
            tip,b,off=probe(state.damage,n);rows.append(dict(time=t,tip=tip,branches=b,off_line_max=off,E=d['total_energy'],T_max=d.get('max_T')))
            if k%250==0:snaps[f'{t:.3f}']=state.damage.astype(np.float32);print(tag,len(erosions),f't={t:.3f} tip={tip:.3f}',b,f'off={off:.2e} drift={drift:.1e} {time.time()-start:.0f}s',flush=True)
            if tip>=.85:break
    tip,b,off=probe(state.damage,n);np.savez_compressed(OUT/f'{tag}.npz',**snaps,final=state.damage.astype(np.float32))
    ts=np.array([r['time'] for r in rows]);tips=np.array([r['tip'] for r in rows]);sp=np.array([(tips[i+8]-tips[i])/(ts[i+8]-ts[i]) for i in range(len(ts)-8)]) if len(ts)>8 else np.zeros(0)
    early=[r for r in rows if r['tip']<.8];fb=next((r for r in early if any(v>=2 for v in r['branches'].values())),None)
    res=dict(n=n,erode=args.erode,band=args.band,erosions=erosions,eroded_cells=sum(len(e['cells']) for e in erosions),deletions=deletions,deleted_cells=sum(e['cells'] for e in deletions),steps=k,end_time=t,failure=failure,halvings=halvings,energy_drift=drift,checks=checks,H2_off_line_max=max((r['off_line_max'] for r in early),default=0.),
        first_branch=None if fb is None else dict(time=fb['time'],tip=fb['tip'],line=min(float(x) for x,v in fb['branches'].items() if v>=2),branches=fb['branches']),
        max_speed=float(sp.max()) if len(sp) else 0.,seconds=time.time()-start,rows=rows)
    (OUT/f'{tag}.json').write_text(json.dumps(res,indent=1,default=float)+'\n');print(n,'done',{k2:v for k2,v in res.items() if k2!='rows'},flush=True)


if __name__=='__main__':main()
