"""Section 59: case 9 with local crystal plasticity on uniform and adaptive periodic grids, with declared gates."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform
from pathlib import Path
from time import perf_counter
import numpy as np
from bmvr.grid import Grid,circle
from bmvr.adaptivity import focused_grid,adapt,transfer
from bmvr.defects import Screws
from bmvr.local_plasticity import AntiplaneElasticity,energy_forces

ROOT=Path(__file__).resolve().parents[1]
LENGTH,H=64.,.5
PARAMS=dict(imposed=.15,chemical=.3,interface=.2,c0=.2,hardening=.5,gradient=1.)
L_GAMMA=1.
CACHE={}


WEIGHTED={'on':True}


def elasticity(grid):
    key=(grid.xe.tobytes(),grid.ye.tobytes(),WEIGHTED['on'])
    if key not in CACHE:CACHE[key]=AntiplaneElasticity(grid,weighted=WEIGHTED['on'])
    return CACHE[key]


def state(grid,v,c,g,s,o):return energy_forces(grid,elasticity(grid),v,c,g,s,o,**PARAMS)


def homogeneous(dt,end=8.):
    grid=Grid.uniform(LENGTH,32,periodic=True);v=-np.ones(grid.shape);c=np.full(grid.shape,PARAMS['c0']);g=np.zeros(grid.shape)
    s=Screws(np.zeros((0,2)),np.zeros(0),core=1.);o=s.positions.copy();steps=round(end/dt);recurrence=0.
    for _ in range(steps):
        cur=state(grid,v,c,g,s,o);g=g-dt*L_GAMMA*cur['g_gamma']
        recurrence=recurrence+dt*L_GAMMA*(PARAMS['imposed']-(1+PARAMS['hardening'])*recurrence)
    inf=PARAMS['imposed']/(1+PARAMS['hardening']);exact=inf*(1-np.exp(-L_GAMMA*(1+PARAMS['hardening'])*end))
    return dict(dt=dt,gamma=float(g.mean()),spread=float(np.ptp(g)),recurrence=float(recurrence),recurrence_error=float(abs(g.mean()-recurrence)),continuous=float(exact),continuous_error=float(abs(g.mean()-exact)))


def coupled(variant,end=8.):
    out=dict(variant=variant,timings=dict(solver_s=0.,adaptation_s=0.,factorization_s=0.))
    grid=focused_grid(LENGTH,H,((-24,24),(-24,24)),periodic=True) if variant=='adaptive' else Grid.uniform(LENGTH,128,periodic=True)
    v,c,g=circle(grid,8),np.full(grid.shape,.2),np.zeros(grid.shape)
    s=Screws(np.array([[-10.,0],[10.,0]]),np.array([1.,-1.]),core=1.);o=s.positions.copy();inventory=grid.integrate(c)
    start=perf_counter();cur=state(grid,v,c,g,s,o);out['timings']['factorization_s']+=perf_counter()-start
    rows=[dict(time=0.,energy=cur['energy'],mass=inventory,positions=s.positions.tolist(),plastic_screws=0.,mean_gamma=0.,mean_stress=cur['mean_stress'],dofs=3*grid.size+2)]
    t,steps,events,increase=0.,0,[],0.
    while t<end-1e-12:
        start=perf_counter();dv,dc,dg=-cur['gv'],grid.lap(cur['mu']),-L_GAMMA*cur['g_gamma'];vel=s.mobility*cur['force'];used=min(.02,end-t)
        for _ in range(24):
            nv,nc,ng,ns=v+used*dv,c+used*dc,g+used*dg,s.moved(s.positions+used*vel);cand=state(grid,nv,nc,ng,ns,o)
            if nc.min()>0 and cand['energy']<=cur['energy']+1e-10:break
            used*=.5
        else:raise RuntimeError('Coupled step failed its energy/bounds check.')
        increase=max(increase,cand['energy']-cur['energy']);v,c,g,s,cur=nv,nc,ng,ns,cand;t+=used;steps+=1
        out['timings']['solver_s']+=perf_counter()-start
        if variant=='adaptive' and steps%100==0:
            start=perf_counter();energy=lambda gr,f:state(gr,f[0],f[1],f[2],s,o)['energy']
            grid,fields,record=adapt(grid,[v,c,g],H,LENGTH,energy,points=s.positions);v,c,g=fields;events.append(record);cur=state(grid,v,c,g,s,o)
            out['timings']['adaptation_s']+=perf_counter()-start
        if steps%20==0 or t>=end-1e-12:
            rows.append(dict(time=t,energy=cur['energy'],mass=grid.integrate(c),positions=s.positions.tolist(),plastic_screws=cur['plastic_screws'],
                mean_gamma=cur['mean_gamma'],mean_stress=cur['mean_stress'],dofs=3*grid.size+2))
    tilde=g-grid.integrate(g)/grid.area;near=np.min([np.hypot(grid.X-x,grid.Y-y) for x,y in s.positions],axis=0)<=8
    out.update(steps=steps,max_energy_increase=increase,inventory_drift=max(abs(r['mass']-inventory) for r in rows)/inventory,burgers=float(s.burgers.sum()),
        transfers=events,accepted_transfers=sum(e['changed'] for e in events),rows=rows,final_positions=s.positions.tolist(),
        locality=dict(max_tilde=float(abs(tilde).max()),max_gamma=float(abs(g).max()),near_fraction=float(grid.integrate(np.where(near,abs(tilde),0.))/grid.integrate(abs(tilde)))),
        grid=dict(xe=grid.xe.tolist(),ye=grid.ye.tolist()),fields=dict(v=v,c=c,gamma=g))
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--declared-interpolation',action='store_true');args=parser.parse_args()
    WEIGHTED['on']=not args.declared_interpolation
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    hom={f'{dt:g}':homogeneous(dt) for dt in (.02,.01)}
    runs={k:coupled(k) for k in ('uniform','adaptive')};u,a=runs['uniform'],runs['adaptive']
    gu=Grid(np.array(u['grid']['xe']),np.array(u['grid']['ye']),periodic=True);ga=Grid(np.array(a['grid']['xe']),np.array(a['grid']['ye']),periodic=True)
    diff={k:float(np.sqrt(gu.integrate((transfer(ga,gu,a['fields'][k])-u['fields'][k])**2)/gu.integrate(u['fields'][k]**2))) for k in ('v','c','gamma')}
    position=float(np.max(np.abs(np.array(a['final_positions'])-np.array(u['final_positions']))))
    np.savez_compressed(out/'fields.npz',**{f'{r}_{k}':runs[r]['fields'][k] for r in runs for k in ('v','c','gamma')},
        uniform_xe=gu.xe,uniform_ye=gu.ye,adaptive_xe=ga.xe,adaptive_ye=ga.ye)
    for r in runs.values():r.pop('fields')
    ratio=u['timings']['solver_s']/(a['timings']['solver_s']+a['timings']['adaptation_s'])
    first,last=u['rows'][0],u['rows'][-1]
    gates=dict(P1_consistency=all(r['max_energy_increase']<=1e-10 and r['inventory_drift']<=1e-10 and r['burgers']==0 for r in runs.values()),
        P2_homogeneous=all(h['recurrence_error']<=1e-12 for h in hom.values()) and 1.8<=hom['0.02']['continuous_error']/hom['0.01']['continuous_error']<=2.2,
        P3_local=all(r['locality']['max_tilde']>=.25*r['locality']['max_gamma'] and r['locality']['near_fraction']>=.5 for r in runs.values()),
        P4_relaxation=abs(last['mean_stress'])<abs(first['mean_stress']) and abs(last['plastic_screws'])>0 and abs(last['mean_gamma'])>0,
        P5_adaptive=max(diff.values())<=.01 and position<=.01 and ratio>1)
    report=dict(scope='Section 59: case 9 with a local slip field; declared gates.',parameters=dict(PARAMS,L_gamma=L_GAMMA,length=LENGTH,h=H),homogeneous=hom,runs=runs,
        adaptive_vs_uniform=dict(relative_l2=diff,max_position_difference=position,solver_ratio_including_adaptation=ratio,
            dof_ratio=u['rows'][-1]['dofs']/a['rows'][-1]['dofs']),gates=gates,environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),threads=1))
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item() if hasattr(o,'item') else str(o))+'\n')
    print(gates,diff,position,ratio,flush=True)


if __name__=='__main__':main()
