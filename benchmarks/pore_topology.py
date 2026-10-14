"""Section 58: pore collapse, coalescence and nucleation events, with declared gates."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform
from pathlib import Path
from time import perf_counter
import numpy as np
from bmvr.pore_topology import PoreModel,GAMMA,A
from bmvr.diagnostics import interface_radii

ROOT=Path(__file__).resolve().parents[1]
K_EX=.5


def collapse(R0=5.,t_max=800.,matrix=.005):
    # The declared matrix n=0 is on the bound: cells where the unrelaxed profile grows cannot supply vacancies,
    # and the first run failed there (run-first-attempt.log). n=0.005 is still far below gamma/R0=0.047.
    # The second attempt failed when the coalescence neck starved (n=0); PoreModel.step now has the availability limiter.
    m=PoreModel();n=np.full(m.grid.shape,matrix);p=m.pore((0.,0.),R0);E=m.energy(n,p);I0=m.inventory(n,p)
    t=0.;rows=[];increase=0.;every=round(1/m.dt)
    k=0;gone=None
    while t<t_max:
        n,p,E1,used=m.step(n,p,E);increase=max(increase,E1-E);E=E1;t+=used;k+=1
        if k%every==0:
            _,count=m.components(p);rows.append(dict(time=t,count=count,area=m.grid.integrate(p),energy=E,drift=abs(m.inventory(n,p)-I0)/max(I0,1.)))
            if count==0 and gone is None:gone=t
            if gone is not None and t>gone+5:break
    tc=R0**2/(2*K_EX/A*GAMMA)
    return dict(R0=R0,matrix=matrix,collapse_time=gone,interface_limited_estimate=tc,ratio=None if gone is None else gone/tc,max_energy_increase=increase,
        max_inventory_drift=max(r['drift'] for r in rows),counts=sorted({r['count'] for r in rows}),rows=rows)


def merged_shape(m,p):
    angles=np.linspace(0,2*np.pi,180,endpoint=False);r=interface_radii(m.grid,p,angles,level=.5)
    basis=np.stack((np.ones_like(angles),np.cos(2*angles),np.sin(2*angles)),1);(a0,a2,b2),*_=np.linalg.lstsq(basis,r,rcond=None);return float(a0),float(a2)


def coalescence(h=.5,t_max=300.,stop_at_merge=False):
    m=PoreModel(h=h);n=np.full(m.grid.shape,GAMMA/8);p=np.maximum(m.pore((-9.5,0.),8.),m.pore((9.5,0.),8.));E=m.energy(n,p);I0=m.inventory(n,p)
    t=0.;rows=[];increase=0.;merge=None;k=0;every=max(1,round(.5/m.dt));limited=0
    _,c0=m.components(p)
    while t<t_max:
        n,p,E1,used=m.step(n,p,E);increase=max(increase,E1-E);E=E1;t+=used;k+=1;limited+=m.limited
        if k%every==0:
            _,count=m.components(p);row=dict(time=t,count=count,drift=abs(m.inventory(n,p)-I0)/I0,energy=E,limited_cell_steps=limited)
            if count==1:
                if merge is None:merge=t
                a0,a2=merged_shape(m,p);row.update(a0=a0,a2_over_a0=a2/a0)
            rows.append(row)
            if stop_at_merge and merge is not None:break
    shape=[r['a2_over_a0'] for r in rows if 'a2_over_a0' in r]
    return dict(h=h,initial_count=c0,merge_time=merge,limited_cell_steps=limited,counts=sorted({r['count'] for r in rows}),max_energy_increase=increase,max_inventory_drift=max(r['drift'] for r in rows),
        shape_monotone=bool(len(shape)>1 and np.all(np.diff(shape)<=1e-12)),shape_first=shape[0] if shape else None,shape_last=shape[-1] if shape else None,rows=rows,
        final_p=p if not stop_at_merge else None)


def nucleation(supersaturation,seed=5801,theta=.6,nu=.001,tau=1.,t_end=40.,exclusion=None):
    m=PoreModel(length=128.);rng=np.random.default_rng(seed);n=np.full(m.grid.shape,supersaturation);p=np.zeros(m.grid.shape)
    E=m.energy(n,p);Q=0.;I0=m.inventory(n,p);events=[];refusals={};history=[];increase=0.;steps=round(tau/m.dt)
    for k in range(round(t_end/tau)):
        for _ in range(rng.poisson(nu*m.grid.area*tau)):
            site=rng.uniform(-m.length/2,m.length/2,2);nn,pp,rec=m.insert(n,p,site,exclusion=exclusion)
            if nn is None:refusals[rec['refused']]=refusals.get(rec['refused'],0)+1;continue
            dF=m.energy(nn,pp)-E
            if rng.uniform()>=min(1.,np.exp(-dF/theta)):refusals['thinned']=refusals.get('thinned',0)+1;continue
            before=m.inventory(n,p);ledger_before=E+Q
            n,p=nn,pp;E=m.energy(n,p);Q-=dF
            labels,_=m.components(p);idx=np.unravel_index(np.argmin(m.distance(site)),p.shape);lab=labels[idx]
            events.append(dict(time=k*tau,**rec,delta_F=float(dF),inventory_error=abs(m.inventory(n,p)-before)/before,
                ledger_error=abs(E+Q-ledger_before)/max(abs(ledger_before),1e-30),min_n=float(n.min()),site_index=[int(i) for i in idx],
                area_at_insertion=float(m.grid.integrate(np.where(labels==lab,p,0.))) if lab else 0.))
        for _ in range(steps):
            n,p,E1,_used=m.step(n,p,E);increase=max(increase,E1-E);E=E1
        labels,count=m.components(p)
        for ev in events:
            lab=labels[tuple(ev['site_index'])];area=float(m.grid.integrate(np.where(labels==lab,p,0.))) if lab else 0.
            ev.setdefault('area_history',[]).append(area)
            if ev['time']==k*tau and 'area_after_interval' not in ev:ev['area_after_interval']=area
        history.append(dict(time=(k+1)*tau,count=count,mean_n=float(m.grid.integrate(n)/m.grid.area),pore_area=float(m.grid.integrate(p)),drift=abs(m.inventory(n,p)-I0)/max(I0,1.)))
    return dict(supersaturation=supersaturation,seed=seed,theta=theta,nu=nu,events=events,refusals=refusals,history=history,max_energy_increase=increase,final_p=p,
        max_inventory_drift=max(h['drift'] for h in history))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--post-hoc-nucleation',action='store_true');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);start=perf_counter()
    import sys
    if '--post-hoc-nucleation' in sys.argv:
        # Post hoc (Section 58 outcome): exclusion zone of 4/a around existing pores; per-embryo areas over later intervals.
        ph=nucleation(.06,exclusion=4/A);ph.pop('final_p')
        # Isolated embryo at the rule's first insertion state, followed for 60 time units.
        m=PoreModel(length=128.);n=np.full(m.grid.shape,.06);p=np.zeros(m.grid.shape);n,p,rec=m.insert(n,p,(0.,0.));E=m.energy(n,p);iso=[];t=0.
        for k in range(1,1201):
            n,p,E,used=m.step(n,p,E);t+=used
            if k%20==0:lab,_=m.components(p);iso.append(dict(time=t,component_area=float(m.grid.integrate(np.where(lab>0,p,0.))),total_pore=float(m.grid.integrate(p))))
        ph['isolated_embryo']=dict(insertion=rec,history=iso)
        (out/'nucleation-post-hoc.json').write_text(json.dumps(ph,indent=1,default=lambda o:o.item())+'\n')
        print(len(ph['events']),ph['refusals'],[(e['time'],round(e['area_at_insertion'],2),[round(a,2) for a in e['area_history'][:6]]) for e in ph['events']],flush=True);return
    col=collapse();co=coalescence();cof=coalescence(h=.25,stop_at_merge=True);nuc=nucleation(.06);ctrl=nucleation(0.)
    np.savez_compressed(out/'fields.npz',coalescence_final=co.pop('final_p'),nucleation_final=nuc.pop('final_p'));ctrl.pop('final_p');cof.pop('final_p')
    energy_ok=lambda r:r['max_energy_increase']<=1e-11
    gates=dict(T1_topology=col['collapse_time'] is not None and co['initial_count']==2 and co['merge_time'] is not None,
        T2_conservation=all(r['max_inventory_drift']<=1e-10 and energy_ok(r) for r in (col,co)),T3_shape=co['shape_monotone'],
        N1_event_inventory=bool(nuc['events']) and all(e['inventory_error']<=1e-12 and e['min_n']>=0 for e in nuc['events']),
        N2_ledger=bool(nuc['events']) and all(e['ledger_error']<=1e-12 for e in nuc['events']),
        N3_growth=bool(nuc['events']) and all(e['area_after_interval']>e['area_at_insertion'] for e in nuc['events']),N4_control=not ctrl['events'])
    report=dict(scope='Section 58 pore topology on the Section 52 model with singly degenerate exchange; synthetic nucleation closure.',
        collapse=col,coalescence=co,coalescence_h025=cof,nucleation=nuc,control=ctrl,gates=gates,study_seconds=perf_counter()-start,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),threads=1))
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item())+'\n')
    print(gates,'collapse',col['collapse_time'],col['ratio'],'merge',co['merge_time'],cof['merge_time'],'events',len(nuc['events']),nuc['refusals'],flush=True)


if __name__=='__main__':main()
