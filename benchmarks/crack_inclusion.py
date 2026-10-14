"""Section 62: crack arrest, deflection and branching at a tough inclusion (Section 36 AT2 model)."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform
from pathlib import Path
from time import perf_counter
import numpy as np
from bmvr.fracture2d import Fracture2D,FractureParameters

CENTER,RADIUS,GM,ELL=(.6,.5),.12,.001,.06
LOADS=np.linspace(0,.12,49)
RUNS={'k1-n64':(1.,64),'k1.25-n64':(1.25,64),'k4-n64':(4.,64),'k4-n96':(4.,96)}
# Post hoc (Section 62 outcome): the declared kappa=4 runs arrested at the inclusion and then failed by a new crack
# along the loaded edge. These runs toughen the edge strips (y<0.06, y>0.94) tenfold to suppress that failure.
POST_HOC={'k4-n64-tough-edges':(4.,64),'k2-n64-tough-edges':(2.,64)}


def toughness(kappa,tough_edges=False):
    def g(x,y):
        value=np.where(np.hypot(x-CENTER[0],y-CENTER[1])<RADIUS,kappa*GM,GM)
        return np.where((y<.06)|(y>.94),10*GM,value) if tough_edges else value
    return g


def measures(model,q):
    x,y=model.coordinates.T;grid=q.reshape(model.n+1,model.n+1)   # grid[i,j] at x_i,y_j
    broken=q>=.8;inside=np.hypot(x-CENTER[0],y-CENTER[1])<RADIUS
    col=grid[int(round(CENTER[0]/model.h)),:];runs=np.flatnonzero(np.diff(np.r_[0,(col>=.8).astype(int),0])==1)
    center=grid[:,model.n//2];along=np.flatnonzero(center>=.8)
    return dict(tip=float(x[broken].max()) if broken.any() else 0.,inside_mean=float(q[inside].mean()),branches_at_center=int(len(runs)),
        centerline_front=float(along[-1]*model.h) if len(along) else 0.,centerline_through_inclusion=bool((center[int((CENTER[0]-RADIUS)/model.h)+1:int((CENTER[0]+RADIUS)/model.h)]>=.8).all()))


def run(name,out):
    kappa,n=(RUNS|POST_HOC)[name];model=Fracture2D(n,parameters=FractureParameters(ell=ELL),toughness=toughness(kappa,name in POST_HOC))
    u=np.zeros(2*model.node_count);q=model.seed();rows=[];snaps={};start=perf_counter();failure=None
    for i,load in enumerate(LOADS):
        before=q.copy()
        try:u,q,state=model.solve_load(u,q,float(load))
        except RuntimeError as error:failure=dict(load=float(load),index=i,message=str(error));break
        rows.append(dict(load=float(load),reaction=state['reaction'],energy=state['energy'],fracture_energy=state['fracture_energy'],min_increment=float((q-before).min()),
            q_min=float(q.min()),q_max=float(q.max()),max_fixed_load_increase=float(max(0,np.max(np.diff(state['fixed_load_energy'])))),
            mechanical_residual=state['mechanical_residual'],damage_residual=state['damage_residual'],iterations=state['alternate_iterations'],**measures(model,q)))
        if i%4==0 or i==len(LOADS)-1:snaps[f'{load:.4f}']=q.copy()
        print(name,i,f"tip={rows[-1]['tip']:.3f} inside={rows[-1]['inside_mean']:.3f} branches={rows[-1]['branches_at_center']}",flush=True)
    np.savez_compressed(out/f'{name}.npz',**snaps)
    checks=dict(irreversibility=min(r['min_increment'] for r in rows)>=-1e-12,bounds=min(r['q_min'] for r in rows)>=0 and max(r['q_max'] for r in rows)<=1,
        descent=max(r['max_fixed_load_increase'] for r in rows)<1e-11,stationarity=max(max(r['mechanical_residual'],r['damage_residual']) for r in rows)<=1e-6)
    result=dict(name=name,kappa=kappa,n=n,rows=rows,failure=failure,checks=checks,seconds=perf_counter()-start)
    (out/f'{name}.json').write_text(json.dumps(result,indent=1,default=lambda o:o.item())+'\n')


def arrest(rows,edge=CENTER[0]-RADIUS):
    arrived=[r for r in rows if r['tip']>=edge-ELL]
    if not arrived:return None
    first=arrived[0];parked=[r for r in arrived if r['tip']<=edge+ELL]
    last=parked[-1] if parked else first
    return dict(arrival_load=first['load'],leave_load=last['load'],interval_fraction=(last['load']-first['load'])/max(first['load'],1e-30),
        reaction_rises=last['reaction']>first['reaction'])


def assemble(out):
    r={k:json.loads((out/f'{k}.json').read_text()) for k in RUNS}
    fin=lambda k:r[k]['rows'][-1];arr={k:arrest(r[k]['rows']) for k in RUNS}
    gates=dict(B1_control=r['k1-n64']['failure'] is None and fin('k1-n64')['centerline_front']>=1-1e-9,
        B2_arrest=arr['k4-n64'] is not None and arr['k4-n64']['interval_fraction']>=.1 and arr['k4-n64']['reaction_rises'],
        B3_deflection=fin('k4-n64')['inside_mean']<.3 and fin('k4-n64')['tip']>=CENTER[0]+RADIUS,
        B4_penetration=any(row['centerline_through_inclusion'] for row in r['k1.25-n64']['rows']),
        B5_numerics=all(all(x['checks'].values()) for x in r.values()))
    report=dict(scope='Section 62: tough inclusion ahead of a seeded notch, Section 36 AT2 model; declared gates.',center=CENTER,radius=RADIUS,matrix_toughness=GM,ell=ELL,
        runs={k:{a:b for a,b in v.items() if a!='rows'} for k,v in r.items()},final={k:fin(k) for k in RUNS},arrest=arr,gates=gates,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),threads=1))
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item())+'\n');print(gates,arr,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--run',choices=list(RUNS)+list(POST_HOC));args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    run(args.run,out) if args.run else assemble(out)


if __name__=='__main__':main()
