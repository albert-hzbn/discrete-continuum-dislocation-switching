"""Section 33 reciprocal coupling, refinement and matched-law comparison."""
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
from bmvr.coupled_transport import CoupledTransport,CoupledGammaCache
from response_cache import array_storage
from representation_transfer import spectral_restrict
from three_dimensional import volume_project


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def initial(n):
    grid=Cube(12,n);x,y,z=grid.coordinates()
    v=np.tanh((3-np.sqrt(x*x+y*y+(z-.3)**2))/np.sqrt(2))
    c=.2+.04*np.exp(-((x-2)**2+(y-1)**2+z*z)/(2*1.5**2))
    loops=[circular_loop(radius=1.5,count=8,center=(center,0,0),burgers=(sign*.2,0,0),core_tension=0,
        glide_normal=np.array([0.,0.,1.])) for center,sign in ((-2,1),(2,-1))]
    gamma=PlanarTransport(grid).gamma_from_loops(loops)
    mean=np.zeros((3,3));mean[0,2]=mean[2,0]=.06
    return CoupledTransport(grid,mean_strain=mean),(v,c,gamma)


def worker(n,kind,dt,output):
    output.mkdir(parents=True,exist_ok=True);start=perf_counter()
    model,fields=initial(n);g=model.grid;v,c,gamma=fields
    initial_fields=tuple(f.copy() for f in fields)
    cache=CoupledGammaCache(model,gamma) if kind=='affine' else None
    a=np.array([]);state=model.evaluate(*fields);mass=g.integrate(c)
    initial_energy=state['energy'];mode=kind
    timings={'initialization_s':perf_counter()-start,'online_s':0.,'compilation_s':0.}
    times=[0.];energies=[initial_energy];inventories=[mass];counts=[2*n**3+(0 if cache else n**3)]
    events=[];residuals=[];t=0.;steps=0;backtracks=0
    while t<1-1e-12:
        began=perf_counter()
        if mode=='affine':
            a,state,event=cache.enrich(v,c,a,budget=.05,max_modes=4)
            if event['added_modes'] or event['fallback_required']:events.append({'time':t,**event})
            if event['fallback_required']:mode='field'
            else:residuals.append(state['projection']['relative_residual'])
        if mode=='affine':
            v,c,a,state,used,trials=cache.step(v,c,a,min(dt,1-t),current=state)
            gamma=state['gamma']
        else:
            (v,c,gamma),state,used,trials=model.step((v,c,gamma),min(dt,1-t),current=state,
                freeze_gamma=kind=='frozen-gamma',freeze_fields=kind=='frozen-fields')
        timings['online_s']+=perf_counter()-began;t+=used;steps+=1;backtracks+=trials-1
        times.append(t);energies.append(state['energy']);inventories.append(g.integrate(c))
        counts.append(2*n**3+(len(a) if mode=='affine' else n**3))
    timings['other_evolution_s']=perf_counter()-start-sum(timings.values())
    stored=array_storage(model,cache,state,v,c,gamma,a,initial_fields)
    began=perf_counter();readback=model.tensor_readback(v,c,gamma);fresh=model.evaluate(v,c,gamma)
    physical_energy=sum(val for key,val in fresh['components'].items() if key!='mechanical_and_core')+readback['energy']+readback['core_energy']
    energy_error=abs(physical_energy-state['energy'])
    force_error=float(np.max(abs(state['trace_stress']-np.einsum('ii...->...',readback['stress']))))
    timings['tensor_validation_s']=perf_counter()-began
    changes={key:float(np.linalg.norm(f-before)/max(np.linalg.norm(before-bg),1e-30))
        for key,f,before,bg in zip(('phase','population','gamma'),(v,c,gamma),initial_fields,(-1,.2,0))}
    report={'n':n,'kind':kind,'dt':dt,'steps':steps,'backtracks':backtracks,'end_time':t,'final_mode':mode,
        'initial_energy':initial_energy,'final_energy':state['energy'],'max_energy_increase':float(max(0,np.max(np.diff(energies)))),
        'inventory_max_absolute_error':float(max(abs(np.array(inventories)-mass))),
        'population_range':[float(c.min()),float(c.max())],'tensor_energy_absolute_error':energy_error,
        'trace_stress_max_error':force_error,'equilibrium_rms':readback['equilibrium_rms'],
        'divergence_relative':readback['divergence_relative'],'plastic_trace':readback['plastic_trace'],
        'maximum_used_residual':float(max(residuals)) if residuals else None,'events':events,
        'final_gamma_coordinates':len(a) if mode=='affine' else n**3,'final_total_coordinates':counts[-1],
        'persistent_array_bytes':stored,'changes_relative_to_initial':changes,'timings':timings,
        'parameters':{'length':12.,'initial_core':.8,'profile_scale':float(np.sqrt(2)),'core_coefficient':.3,
            'field_mobility':10.,'budget':.05,'max_modes':4,'epsilon_xz':.06},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'platform':platform.platform(),'threads':1}}
    report['checks']={'energy_descent':report['max_energy_increase']<1e-10,
        'inventory':report['inventory_max_absolute_error']<1e-10,'positive_population':bool(0<c.min() and c.max()<1),
        'full_tensor_energy':energy_error<1e-10,'reciprocal_trace_stress':force_error<1e-10,
        'mechanical_equilibrium':readback['equilibrium_rms']<1e-10,'burgers_closure':readback['divergence_relative']<1e-10,
        'plastic_volume':abs(readback['plastic_trace'])<1e-10,'kinetic_budget':not residuals or max(residuals)<=.05,
        'activation_energy':all(e['energy_jump']==0 for e in events)}
    report['checks']={key:bool(value) for key,value in report['checks'].items()}
    began=perf_counter()
    np.savez_compressed(output/'fields.npz',phase=v,population=c,gamma=gamma,alpha=readback['alpha'],stress=readback['stress'],
        time=times,energy=energies,inventory=inventories,coordinates=counts)
    if cache is not None:cache.save(output/'restart.npz',v,c,a,gamma,mode,t,steps)
    timings['array_io_s']=perf_counter()-began
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;report['peak_rss_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    began=perf_counter();dump(output/'report.json',report)
    dump(output/'output-timing.json',{'report_io_s':perf_counter()-began})


def compare(output,first,reference):
    a=json.loads((output/first/'report.json').read_text());b=json.loads((output/reference/'report.json').read_text())
    errors={};averaged={}
    with np.load(output/first/'fields.npz') as data,np.load(output/reference/'fields.npz') as ref:
        for key,bg in (('phase',-1),('population',.2),('gamma',0),('alpha',0),('stress',0)):
            exact=ref[key] if a['n']==b['n'] else spectral_restrict(ref[key],b['n'],a['n'],12)
            delta=data[key]-exact;scale=max(np.linalg.norm(exact-bg),1e-30)
            errors[key]={'l1':float(np.mean(abs(delta))),'l2':float(np.sqrt(np.mean(delta*delta))),
                'linf':float(np.max(abs(delta))),'relative_l2':float(np.linalg.norm(delta)/scale)}
            if a['n']!=b['n']:
                cell=volume_project(ref[key],b['n'],a['n'],12)
                averaged[key]=float(np.linalg.norm(data[key]-cell)/max(np.linalg.norm(cell-bg),1e-30))
    return {'run':first,'reference':reference,'errors':errors,'cell_overlap_relative_l2':averaged,
        'energy_relative':abs(a['final_energy']/b['final_energy']-1),
        'online_ratio_reference_over_run':b['timings']['online_s']/a['timings']['online_s']}


def collect(output):
    reports={p.parent.name:json.loads(p.read_text()) for p in sorted(output.glob('*/report.json'))}
    for name,r in reports.items():
        r['timings'].update(json.loads((output/name/'output-timing.json').read_text()))
        r.update(json.loads((output/name/'process.json').read_text()))
    matched=[compare(output,f'affine-{n}','field-'+str(n)) for n in (13,21,29)]
    time=[compare(output,f'{kind}-21',f'{kind}-21-halfdt') for kind in ('field','affine')]
    space=[compare(output,f'field-{n}','field-41') for n in (13,21,29)]
    ablations=[compare(output,f'{kind}-21','field-21') for kind in ('frozen-gamma','frozen-fields')]
    feedback=[]
    for kind,key in (('frozen-gamma','population'),('frozen-fields','alpha')):
        with np.load(output/f'{kind}-21/fields.npz') as a,np.load(output/'field-21/fields.npz') as b,\
             np.load(output/f'{kind}-21-halfdt/fields.npz') as c,np.load(output/'field-21-halfdt/fields.npz') as d:
            coarse=a[key]-b[key];fine=c[key]-d[key]
            change=float(np.linalg.norm(coarse-fine)/max(np.linalg.norm(fine),1e-30))
            feedback.append({'ablation':kind,'observable':key,'paired_difference_relative_change':change,'persistence_10pct':change<.1})
    source=[ROOT/'src/bmvr/coupled_transport.py',ROOT/'src/bmvr/response_cache.py',Path(__file__),
        ROOT/'src/bmvr/continuum3d.py',ROOT/'unified_multiscale_defect_framework.md']
    summary={'reports':reports,'matched':matched,'time':time,'space':space,'ablations':ablations,'feedback_refinement':feedback,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        'spatial_error_monotonicity':{key:all(a['errors'][key]['l2']>b['errors'][key]['l2'] for a,b in zip(space,space[1:])) for key in space[0]['errors']},
        'scope':'Only gamma is reduced; phase/population remain full grid. One timing per case does not establish speedup. Spatial collocation restriction and cell-overlap diagnostics are both retained.'}
    dump(output/'summary.json',summary)
    print(json.dumps({'matched_alpha_errors':[r['errors']['alpha']['relative_l2'] for r in matched],
        'failed_checks':{k:[c for c,v in r['checks'].items() if not v] for k,r in reports.items()},
        'spatial_error_monotonicity':summary['spatial_error_monotonicity']},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/coupled-transport-2026-09-22')
    parser.add_argument('--worker',choices=('field','affine','frozen-gamma','frozen-fields'))
    parser.add_argument('--n',type=int,default=13);parser.add_argument('--dt',type=float,default=.01)
    parser.add_argument('--collect',action='store_true');parser.add_argument('--feedback-refinement',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.n,args.worker,args.dt,args.output);return
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.collect:
        configurations=[(f'{kind}-{n}',n,kind,.01) for n in (13,21,29) for kind in ('field','affine')]
        configurations += [('field-41',41,'field',.01)]
        configurations += [(f'{kind}-21-halfdt',21,kind,.005) for kind in ('field','affine')]
        configurations += [(f'{kind}-21',21,kind,.01) for kind in ('frozen-gamma','frozen-fields')]
        refined=[(f'{kind}-21-halfdt',21,kind,.005) for kind in ('frozen-gamma','frozen-fields')]
        configurations=refined if args.feedback_refinement else configurations+refined
        for name,n,kind,dt in configurations:
            folder=args.output/name
            if (folder/'report.json').exists():raise FileExistsError('Use a fresh output directory.')
            began=perf_counter();subprocess.run([sys.executable,__file__,'--worker',kind,'--n',str(n),'--dt',str(dt),'--output',str(folder)],cwd=ROOT,check=True)
            dump(folder/'process.json',{'launch_to_exit_s':perf_counter()-began});print(name,'finished',flush=True)
    collect(args.output)


if __name__=='__main__':main()
