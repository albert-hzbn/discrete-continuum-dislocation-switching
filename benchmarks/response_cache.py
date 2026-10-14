"""Fair repeated timing of equally optimized full and affine transport paths."""
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
from bmvr.continuum3d import Cube
from bmvr.loops3d import circular_loop
from bmvr.transport_metric import PlanarTransport
from bmvr.response_cache import ScalarTransportOperator,AffineTransportCache


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def array_storage(*objects):
    seen=set();arrays={}
    def visit(item):
        if id(item) in seen:return
        seen.add(id(item))
        if isinstance(item,np.ndarray):
            owner=item
            while isinstance(owner.base,np.ndarray):owner=owner.base
            arrays[id(owner)]=owner.nbytes
        elif isinstance(item,dict):
            for value in item.values():visit(value)
        elif isinstance(item,(list,tuple)):
            for value in item:visit(value)
        elif hasattr(item,'__dict__'):
            visit(item.__dict__)
    for item in objects:visit(item)
    return sum(arrays.values())


def worker(n,kind,output):
    output.mkdir(parents=True,exist_ok=True);start=perf_counter()
    grid=Cube(12,n);x,y,z=grid.coordinates()
    model=PlanarTransport(grid,background=np.eye(3)[...,None,None,None]*.03*np.cos(2*np.pi*x/12)*np.sin(2*np.pi*y/12))
    loops=[circular_loop(radius=1.5,count=8,center=(center,0,0),burgers=(sign*.2,0,0),core_tension=0,
                        glide_normal=np.array([0.,0.,1.])) for center,sign in ((-2,1),(2,-1))]
    gamma=model.gamma_from_loops(loops);operator=ScalarTransportOperator(model)
    cache=AffineTransportCache(operator,gamma) if kind=='affine' else None;c=np.array([])
    state=cache.state(c,project=False) if cache is not None else operator.evaluate(gamma)
    mode=kind;initial_energy=state['energy'];initial_alpha=model.evaluate(gamma)['alpha']
    timings={'initialization_s':perf_counter()-start,'online_s':0.,'compilation_s':0.}
    times=[0.];energies=[state['energy']];counts=[0 if cache else grid.n**3];events=[];residuals=[]
    t=0.;steps=0;max_cache_bytes=0
    while t<4-1e-12:
        began=perf_counter()
        if mode=='affine':
            c,state,event=cache.enrich(c,budget=.05,max_modes=3)
            max_cache_bytes=max(max_cache_bytes,cache.storage_bytes)
            if event['added_modes'] or event['fallback_required']:events.append({'time':t,**event})
            if event['fallback_required']:mode='field'
            else:residuals.append(state['projection']['relative_residual'])
        if mode=='affine':c,state,used,trials=cache.step(c,state,min(.02,4-t))
        else:state,used,trials=operator.step(state,min(.02,4-t))
        timings['online_s']+=perf_counter()-began;t+=used;steps+=1
        times.append(t);energies.append(state['energy']);counts.append(len(c) if mode=='affine' else grid.n**3)
    numerical_total=perf_counter()-start
    timings['other_evolution_s']=numerical_total-timings['initialization_s']-timings['online_s']
    stored=array_storage(model,operator,cache,state,c,loops)
    began=perf_counter();direct=model.evaluate(state['gamma'])
    force_error=float(np.max(np.abs(state['conjugate']-direct['conjugate'])))
    energy_error=abs(state['energy']-direct['energy'])
    timings['full_tensor_validation_s']=perf_counter()-began
    report={'n':n,'kind':kind,'steps':steps,'end_time':t,'final_mode':mode,'final_coordinates':counts[-1],
        'parameters':{'length':12.,'initial_core':.8,'core_coefficient':.3,'field_mobility':10.,'dt':.02,'budget':.05,'mode_cap':3},
        'initial_energy':initial_energy,'final_energy':state['energy'],'max_energy_increase':float(max(0,np.max(np.diff(energies)))),
        'tensor_energy_absolute_error':energy_error,'tensor_conjugate_max_error':force_error,
        'source_change_relative':float(np.linalg.norm(direct['alpha']-initial_alpha)/np.linalg.norm(initial_alpha)),
        'divergence_relative':direct['divergence_relative'],'plastic_trace':float(np.trace(direct['mean_beta'])),
        'maximum_used_residual':float(max(residuals)) if residuals else None,
        'persistent_array_bytes':stored,'maximum_cache_bytes':max_cache_bytes,'response_calls':operator.response_calls,
        'events':events,'timings':timings,'environment':{'python':platform.python_version(),'numpy':np.__version__,
        'scipy':scipy.__version__,'platform':platform.platform(),'threads':1}}
    report['checks']={'energy_descent':report['max_energy_increase']<1e-10,
        'burgers_closure':direct['divergence_relative']<1e-10,'glide_volume':abs(report['plastic_trace'])<1e-10,
        'full_tensor_energy':energy_error<1e-10,'full_tensor_force':force_error<1e-10,
        'kinetic_budget':not residuals or max(residuals)<=.05,'activation_preserves_state_energy':all(e['energy_jump']==0 for e in events)}
    began=perf_counter();np.savez_compressed(output/'fields.npz',gamma=state['gamma'],alpha=direct['alpha'],stress=direct['stress'],time=times,energy=energies,coordinates=counts)
    timings['array_io_s']=perf_counter()-began
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;report['peak_rss_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    began=perf_counter();dump(output/'report.json',report);report_io=perf_counter()-began
    dump(output/'output-timing.json',{'report_io_s':report_io})


def collect(output):
    reports=[]
    for path in sorted(output.glob('run-*/report.json')):
        r=json.loads(path.read_text());r['folder']=path.parent.name
        r['timings'].update(json.loads((path.parent/'output-timing.json').read_text()))
        r.update(json.loads((path.parent/'process.json').read_text()))
        r['timings']['process_measurement_overhead_s']=r['launch_to_exit_s']-sum(r['timings'].values())
        reports.append(r)
    comparisons=[]
    for n in (13,25,33):
        full=[r for r in reports if r['n']==n and r['kind']=='field'];affine=[r for r in reports if r['n']==n and r['kind']=='affine']
        median=lambda rows,key:float(np.median([r['timings'][key] for r in rows]))
        errors={}
        with np.load(output/full[0]['folder']/'fields.npz') as a,np.load(output/affine[0]['folder']/'fields.npz') as b:
            for key in ('gamma','alpha','stress'):
                d=b[key]-a[key];errors[key]={'l1':float(np.mean(np.abs(d))),'l2':float(np.sqrt(np.mean(d*d))),
                    'linf':float(np.max(np.abs(d))),'relative_l2':float(np.linalg.norm(d)/np.linalg.norm(a[key]))}
        er=abs(affine[0]['final_energy']/full[0]['final_energy']-1)
        launch=lambda rows:float(np.median([r['launch_to_exit_s'] for r in rows]))
        comparisons.append({'n':n,'repetitions':min(len(full),len(affine)),'errors':errors,'energy_relative':er,
            'accuracy_met':errors['alpha']['relative_l2']<.01 and er<.001,
            'field_online_s':median(full,'online_s'),'affine_online_s':median(affine,'online_s'),
            'online_ratio':median(full,'online_s')/median(affine,'online_s'),
            'field_launch_s':launch(full),'affine_launch_s':launch(affine),'launch_ratio':launch(full)/launch(affine),
            'field_launch_range':[min(r['launch_to_exit_s'] for r in full),max(r['launch_to_exit_s'] for r in full)],
            'affine_launch_range':[min(r['launch_to_exit_s'] for r in affine),max(r['launch_to_exit_s'] for r in affine)],
            'field_peak_rss':float(np.median([r['peak_rss_bytes'] for r in full])),
            'affine_peak_rss':float(np.median([r['peak_rss_bytes'] for r in affine])),
            'field_persistent_arrays':full[0]['persistent_array_bytes'],'affine_persistent_arrays':affine[0]['persistent_array_bytes'],
            'field_responses':full[0]['response_calls'],'affine_responses':affine[0]['response_calls'],
            'final_affine_coordinates':affine[0]['final_coordinates']})
    source=[ROOT/'src/bmvr/response_cache.py',ROOT/'src/bmvr/transport_metric.py',Path(__file__),ROOT/'unified_multiscale_defect_framework.md']
    dump(output/'summary.json',{'reports':reports,'comparisons':comparisons,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        'scope':'Same homogeneous continuum law; both methods use the exact scalar operator. No memory or novelty benefit presumed.'})
    print([(r['n'],r['online_ratio'],r['launch_ratio'],r['accuracy_met']) for r in comparisons],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/response-cache-optimized-2026-09-13')
    p.add_argument('--worker',choices=('field','affine'));p.add_argument('--n',type=int,default=13);p.add_argument('--collect',action='store_true');args=p.parse_args()
    if args.worker:worker(args.n,args.worker,args.output);return
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.collect:
        configurations=[(n,kind) for n in (13,25,33) for kind in ('field','affine')]
        for repeat in range(3):
            shift=repeat*2;order=configurations[shift:]+configurations[:shift]
            if repeat%2:order=[item for j in range(0,len(order),2) for item in order[j:j+2][::-1]]
            for n,kind in order:
                folder=args.output/f'run-{n}-{kind}-{repeat+1}'
                if (folder/'report.json').exists():raise FileExistsError('Use a fresh output directory.')
                began=perf_counter();subprocess.run([sys.executable,__file__,'--worker',kind,'--n',str(n),'--output',str(folder)],cwd=ROOT,check=True)
                dump(folder/'process.json',{'launch_to_exit_s':perf_counter()-began});print(folder.name,'finished',flush=True)
    collect(args.output)


if __name__=='__main__':main()
