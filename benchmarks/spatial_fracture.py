"""Fixed-width spatial AT2 profiles, crack loading, unloading and sensitivities."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,hashlib,json,platform,subprocess,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bmvr import _resource as resource
import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
from bmvr.fracture2d import Fracture2D,FractureParameters,bound_quadratic
from response_cache import array_storage


def json_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False,default=json_scalar)+'\n')


def front(model,q):
    profile=q.reshape(model.n+1,model.n+1)[:,model.n//2];where=np.flatnonzero(profile>=.8)
    if not len(where):return 0.
    i=int(where[-1])
    if i==model.n:return model.length
    return float((i+(profile[i]-.8)/(profile[i]-profile[i+1]))*model.h)


def worker(n,steps,residual,heterogeneous,output,iterations=2000):
    output.mkdir(parents=True,exist_ok=True);start=perf_counter()
    toughness=(lambda x,y:.001*(1-.3*(1+np.tanh((x-.55)/.05))/2)) if heterogeneous else None
    model=Fracture2D(n,parameters=FractureParameters(residual=residual),toughness=toughness)
    u=np.zeros(2*model.node_count);q=model.seed();initial=model.evaluate(u,q)
    history=[];snapshots=[];displacements=[];last_load=0.;last_reaction=0.;work=0.
    timings={'initialization_s':perf_counter()-start,'solver_s':0.,'progress_io_s':0.,'compilation_s':0.}
    loads=np.r_[np.linspace(0,.075,steps+1),.05,.025,0.]
    for index,load in enumerate(loads):
        before=q.copy();began=perf_counter()
        try:u,q,state=model.solve_load(u,q,float(load),max_iterations=iterations)
        except RuntimeError as error:
            dump(output/'failure.json',{'load_index':index,'load':float(load),'iteration_cap':iterations,'message':str(error),'completed_loads':history})
            np.savez_compressed(output/'last-accepted.npz',u=u,q=q)
            raise
        timings['solver_s']+=perf_counter()-began
        work+=(state['reaction']+last_reaction)*(load-last_load)/2;last_load=load;last_reaction=state['reaction']
        row={'load':float(load),'reaction':state['reaction'],'energy':state['energy'],'elastic_energy':state['elastic_energy'],
            'fracture_energy':state['fracture_energy'],'work_trapezoid':float(work),'work_minus_energy_change':float(work-state['energy']+initial['energy']),
            'centerline_front':front(model,q),'minimum_damage_increment':float(np.min(q-before)),
            'damage_min':float(q.min()),'damage_max':float(q.max()),'max_fixed_load_energy_increase':float(max(0,np.max(np.diff(state['fixed_load_energy'])))),
            'mechanical_residual':state['mechanical_residual'],'damage_residual':state['damage_residual'],
            'alternate_iterations':state['alternate_iterations'],'mechanical_iterations':state['mechanical_iterations'],
            'damage_iterations':state['damage_iterations']}
        history.append(row);snapshots.append(q.copy());displacements.append(u.copy())
        began=perf_counter();dump(output/'progress.json',{'completed_loads':len(history),'total_loads':len(loads),'latest':row})
        timings['progress_io_s']+=perf_counter()-began
        if index%5==0:print(f'n={n} load {index}/{len(loads)-1}: reaction={row["reaction"]:.5f}, front={row["centerline_front"]:.3f}',flush=True)
    report={'n':n,'ramp_steps':steps,'residual_stiffness':residual,'heterogeneous':heterogeneous,'iteration_cap':iterations,'physical_parameters':{'length':1.,'ell':.06,'lambda':2.,'mu':1.,'toughness':.001,'seed_length':.3},
        'history':history,'initial_energy':initial['energy'],'initial_front':front(model,model.seed()),
        'peak_reaction':max(r['reaction'] for r in history),'final_fracture_energy':history[-1]['fracture_energy'],
        'final_front':history[-1]['centerline_front'],'maximum_alternate_iterations':max(r['alternate_iterations'] for r in history),
        'persistent_array_bytes':array_storage(model,u,q,state),'primary_dofs':3*model.node_count,'timings':timings,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'threads':1}}
    report['checks']={'irreversibility':min(r['minimum_damage_increment'] for r in history)>=-1e-12,
        'damage_bounds':min(r['damage_min'] for r in history)>=0 and max(r['damage_max'] for r in history)<=1,
        'fixed_load_energy_descent':max(r['max_fixed_load_energy_increase'] for r in history)<1e-11,
        'mechanical_stationarity':max(r['mechanical_residual'] for r in history)<=1e-6,
        'damage_stationarity':max(r['damage_residual'] for r in history)<=1e-6,
        'crack_extension':report['final_front']>report['initial_front']+.1,
        'post_peak_softening':history[steps]['reaction']<.8*report['peak_reaction'],
        'unloading_no_healing':float(np.min(snapshots[-1]-snapshots[steps]))>=-1e-12,
        'zero_load_reaction':abs(history[-1]['reaction'])<1e-9}
    began=perf_counter();np.savez_compressed(output/'fields.npz',damage=snapshots,displacement=displacements,load=loads,
        coordinates=model.coordinates,toughness=model.toughness)
    model.save(output/'restart.npz',u,q,float(loads[-1]),len(loads)-1);timings['array_io_s']=perf_counter()-began
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;report['peak_rss_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    began=perf_counter();dump(output/'report.json',report);dump(output/'output-timing.json',{'report_io_s':perf_counter()-began})


def profiles(output):
    rows=[]
    for n in (16,32,64):
        model=Fracture2D(n,parameters=FractureParameters(ell=.1));x=model.coordinates[:,0]
        lower=np.zeros(model.node_count);lower[x==.5]=1.;u=np.zeros(2*model.node_count)
        A,rhs=model.damage_system(u);q,status=bound_quadratic(A,rhs,lower)
        exact=np.cosh((.5-abs(x-.5))/.1)/np.cosh(5)
        energy=model.evaluate(u,q)['fracture_energy'];target=model.p.toughness*np.tanh(5)
        rows.append({'n':n,'h_over_ell':model.h/.1,'profile_rms':float(np.sqrt(np.mean((q-exact)**2))),
            'energy':energy,'analytical_energy':target,'energy_relative_error':abs(energy/target-1),'projected_residual':status['projected_residual']})
        np.savez_compressed(output/f'profile-{n}.npz',coordinates=model.coordinates,q=q,exact=exact)
    model=Fracture2D(24);u,q,state=model.solve_load(np.zeros(2*model.node_count),np.zeros(model.node_count),.02,compression=True)
    compression={'max_damage':float(q.max()),'mechanical_residual':state['mechanical_residual'],'damage_residual':state['damage_residual']}
    dump(output/'profiles.json',{'profiles':rows,'compression':compression,'checks':{
        'profile_refinement':all(a['profile_rms']>b['profile_rms'] for a,b in zip(rows,rows[1:])),
        'energy_refinement':all(a['energy_relative_error']>b['energy_relative_error'] for a,b in zip(rows,rows[1:])),
        'finest_energy_below_one_percent':rows[-1]['energy_relative_error']<.01,'all_compressive_no_damage':compression['max_damage']<1e-12}})


def compare(output,run,reference,qualify=True):
    a=json.loads((output/run/'report.json').read_text());b=json.loads((output/reference/'report.json').read_text())
    fine=Fracture2D(b['n']);points=fine.quadrature_points.reshape(-1,2)
    with np.load(output/run/'fields.npz') as first,np.load(output/reference/'fields.npz') as second:
        values=[]
        for data,n in ((first,a['n']),(second,b['n'])):
            grid=np.linspace(0,1,n+1);values.append(RegularGridInterpolator((grid,grid),data['damage'][-1].reshape(n+1,n+1))(points))
        damage_error=float(np.linalg.norm(values[0]-values[1])/np.linalg.norm(values[1]))
    peak=abs(a['peak_reaction']/b['peak_reaction']-1);energy=abs(a['final_fracture_energy']/b['final_fracture_energy']-1)
    return {'run':run,'reference':reference,'damage_relative_l2':damage_error,'peak_reaction_relative':peak,
        'fracture_energy_relative':energy,'front_absolute':abs(a['final_front']-b['final_front']),
        'gates':{'peak_reaction_5pct':peak<.05,'fracture_energy_5pct':energy<.05} if qualify else {}}


def collect(output):
    reports={p.parent.name:json.loads(p.read_text()) for p in sorted(output.glob('*/report.json'))}
    for name,r in reports.items():
        r['timings'].update(json.loads((output/name/'output-timing.json').read_text()));r.update(json.loads((output/name/'process.json').read_text()))
    comparisons=[compare(output,'mesh-24','mesh-48'),compare(output,'mesh-32','mesh-48'),compare(output,'mesh-32','load-halfstep'),
        compare(output,'residual-small','mesh-32',False),compare(output,'heterogeneous','mesh-32',False)]
    if 'mesh-64' in reports:comparisons.append(compare(output,'mesh-48','mesh-64'))
    source=[Path(__file__),ROOT/'src/bmvr/fracture2d.py',ROOT/'unified_multiscale_defect_framework.md']
    summary={'reports':reports,'comparisons':comparisons,'profiles':json.loads((output/'profiles.json').read_text()),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        'scope':'Conventional plane-strain spectral AT2 reference, fixed physical length, local staggered stationarity; no acceleration or global crack-path claim.'}
    dump(output/'summary.json',summary);print(json.dumps({'comparisons':comparisons,'failed_checks':{k:[c for c,v in r['checks'].items() if not v] for k,r in reports.items()}},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/spatial-fracture-complete-2026-09-22')
    p.add_argument('--worker',action='store_true');p.add_argument('--n',type=int,default=24);p.add_argument('--steps',type=int,default=20)
    p.add_argument('--residual',type=float,default=1e-4);p.add_argument('--heterogeneous',action='store_true');p.add_argument('--collect',action='store_true')
    p.add_argument('--iterations',type=int,default=2000);p.add_argument('--finer',action='store_true');args=p.parse_args()
    if args.worker:worker(args.n,args.steps,args.residual,args.heterogeneous,args.output,args.iterations);return
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.collect:
        if not args.finer:profiles(args.output)
        jobs=[(f'mesh-{n}',n,20,1e-4,False) for n in (24,32,48)]+[('load-halfstep',32,40,1e-4,False),('residual-small',32,20,1e-5,False),('heterogeneous',32,20,1e-4,True)]
        if args.finer:jobs=[('mesh-64',64,20,1e-4,False)]
        for name,n,steps,residual,heterogeneous in jobs:
            folder=args.output/name
            if (folder/'report.json').exists():raise FileExistsError('Use a fresh output directory.')
            began=perf_counter();subprocess.run([sys.executable,__file__,'--worker','--n',str(n),'--steps',str(steps),'--residual',str(residual),
                '--output',str(folder),'--iterations',str(args.iterations)]+(['--heterogeneous'] if heterogeneous else []),cwd=ROOT,check=True)
            dump(folder/'process.json',{'launch_to_exit_s':perf_counter()-began});print(name,'finished',flush=True)
    collect(args.output)


if __name__=='__main__':main()
