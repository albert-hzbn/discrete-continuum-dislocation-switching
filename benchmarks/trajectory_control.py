"""Independent full references for first-order accumulated-error correction."""
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
from bmvr.localized_transport import LocalizedTransport
from bmvr.trajectory_control import TrajectoryControl
from coupled_transport import initial,dump,compare
from response_cache import array_storage


def worker(n,kind,dt,output):
    output.mkdir(parents=True,exist_ok=True);start=perf_counter();model,fields=initial(n);g=model.grid
    control=None if kind=='field' else TrajectoryControl(LocalizedTransport(model,fields,reset_at_cap=True),correct=kind=='corrected')
    state=model.evaluate(*fields) if control is None else control.chart.state(project=False)
    mass=g.integrate(fields[1]);initial_energy=state['energy'];timings={'initialization_s':perf_counter()-start,'online_s':0.,'compilation_s':0.}
    t=0.;times=[0.];energies=[initial_energy];inventories=[mass];events=[];counts=[];indicators=[];steps=0;backtracks=0
    while t<1-1e-12:
        began=perf_counter()
        if control is None:
            fields,state,used,trials=model.step(fields,min(dt,1-t),current=state);counts.append([n**3]*3)
        else:
            state,used,trials,event=control.step(min(dt,1-t));fields=state['fields'];events.append(event)
            counts.append(event['used_coordinates']);indicators.append(list(event['indicators'].values()))
        timings['online_s']+=perf_counter()-began;t+=used;steps+=1;backtracks+=trials-1
        times.append(t);energies.append(state['energy']);inventories.append(g.integrate(fields[1]))
    timings['other_evolution_s']=perf_counter()-start-sum(timings.values())
    stored=array_storage(model,control,state,fields)
    began=perf_counter();tensor=model.tensor_readback(*fields);direct=model.evaluate(*fields)
    total=sum(v for k,v in direct['components'].items() if k!='mechanical_and_core')+tensor['energy']+tensor['core_energy']
    energy_error=abs(state['energy']-total);force_error=max(float(np.max(abs(a-b))) for a,b in zip(state['forces'],direct['forces']))
    timings['tensor_validation_s']=perf_counter()-began
    corrections=[e['correction'] for e in events if e['correction'] is not None]
    report={'n':n,'kind':kind,'dt':dt,'end_time':t,'steps':steps,'backtracks':backtracks,'initial_energy':initial_energy,'final_energy':state['energy'],
        'max_energy_increase':float(max(0,np.max(np.diff(energies)))),'inventory_max_absolute_error':float(max(abs(np.array(inventories)-mass))),
        'tensor_energy_absolute_error':energy_error,'direct_force_max_error':force_error,'equilibrium_rms':tensor['equilibrium_rms'],
        'divergence_relative':tensor['divergence_relative'],'plastic_trace':tensor['plastic_trace'],'corrections':corrections,
        'correction_count':0 if control is None else control.corrections,'rejection_count':0 if control is None else control.rejections,
        'tangent_calls':0 if control is None else control.tangent_calls,'indicator_calls':0 if control is None else control.indicator_calls,
        'maximum_physical_coordinates':int(np.max(np.sum(counts,axis=1))),
        'auxiliary_error_coordinates':0 if control is None else 3*n**3,
        'maximum_total_evolving_coordinates':int(np.max(np.sum(counts,axis=1)))+(0 if control is None else 3*n**3),
        'final_used_coordinates':counts[-1],
        'persistent_array_bytes':stored,'timings':timings,'environment':{'python':platform.python_version(),'numpy':np.__version__,
            'scipy':scipy.__version__,'platform':platform.platform(),'threads':1},
        'parameters':{'length':12.,'initial_core':.8,'profile_scale':float(np.sqrt(2)),'core_coefficient':.3,'field_mobility':10.,
            'epsilon_xz':.06,'kinetic_budget':.05,'mode_cap':32,'field_trigger':.005,'energy_trigger':.0005}}
    report['checks']={'energy_descent':report['max_energy_increase']<1e-10,'inventory':report['inventory_max_absolute_error']<1e-10,
        'positive_population':0<fields[1].min() and fields[1].max()<1,'tensor_energy':energy_error<1e-10,'direct_forces':force_error<1e-10,
        'mechanical_equilibrium':tensor['equilibrium_rms']<1e-10,'burgers_closure':tensor['divergence_relative']<1e-10,
        'plastic_volume':abs(tensor['plastic_trace'])<1e-10,'kinetic_budgets':all(max(e['used_residuals'])<=.05 for e in events),
        'correction_audits':all(c['accepted'] and c['proposed_energy_jump']<1e-10 and abs(c['inventory_jump'])<1e-10 for c in corrections)}
    report['checks']={k:bool(v) for k,v in report['checks'].items()}
    began=perf_counter();error=np.zeros((3,)+g.shape) if control is None else np.array(control.error)
    np.savez_compressed(output/'fields.npz',phase=fields[0],population=fields[1],gamma=fields[2],alpha=tensor['alpha'],stress=tensor['stress'],
        error=error,time=times,energy=energies,inventory=inventories,coordinates=counts,indicators=np.array(indicators))
    if control is not None:control.save(output/'restart.npz')
    timings['array_io_s']=perf_counter()-began
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;report['peak_rss_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    began=perf_counter();dump(output/'report.json',report);dump(output/'output-timing.json',{'report_io_s':perf_counter()-began})


def collect(output):
    reports={p.parent.name:json.loads(p.read_text()) for p in sorted(output.glob('*/report.json'))}
    for name,r in reports.items():
        # Normalize the first saved report's physical-only count label; no trajectory is changed.
        if 'maximum_evolving_coordinates' in r:
            r['maximum_physical_coordinates']=r.pop('maximum_evolving_coordinates')
            r['auxiliary_error_coordinates']=0 if r['kind']=='field' else 3*r['n']**3
            r['maximum_total_evolving_coordinates']=r['maximum_physical_coordinates']+r['auxiliary_error_coordinates']
            dump(output/name/'report.json',r)
        r['timings'].update(json.loads((output/name/'output-timing.json').read_text()));r.update(json.loads((output/name/'process.json').read_text()))
    pairs=[(f'{kind}-{n}',f'field-{n}') for n in (13,21,29) for kind in ('monitor','corrected')]
    pairs.append(('corrected-21-halfdt','field-21-halfdt'));matched=[compare(output,a,b) for a,b in pairs]
    estimator=[]
    for row in matched:
        row['gates']={key:row['errors'][key]['relative_l2']<.01 for key in ('phase','population','alpha')};row['gates']['energy']=row['energy_relative']<.001
        with np.load(output/row['run']/'fields.npz') as a,np.load(output/row['reference']/'fields.npz') as b:
            row['same_accepted_time_grid']=bool(a['time'].shape==b['time'].shape and np.allclose(a['time'],b['time'],rtol=0,atol=1e-12))
            if row['run'].startswith('monitor'):
                discrepancy={key:float(np.linalg.norm(a['error'][i]-(b[key]-a[key]))/max(np.linalg.norm(b[key]-a[key]),1e-30)) for i,key in enumerate(('phase','population','gamma'))}
                estimator.append({'run':row['run'],'error_field_relative_discrepancy':discrepancy,'gates_20pct':{key:val<.2 for key,val in discrepancy.items()}})
    source=[Path(__file__),ROOT/'src/bmvr/trajectory_control.py',ROOT/'src/bmvr/localized_transport.py',ROOT/'src/bmvr/coupled_transport.py',
        ROOT/'src/bmvr/continuum3d.py',ROOT/'unified_multiscale_defect_framework.md']
    result={'reports':reports,'matched':matched,'estimator_validation':estimator,
        'time':[compare(output,'corrected-21','corrected-21-halfdt'),compare(output,'field-21','field-21-halfdt')],
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        'scope':'First-order error estimation and audited numerical corrections. Full references are independent; no certified bound or efficiency claim.'}
    dump(output/'summary.json',result)
    print(json.dumps({'comparisons':[{'run':r['run'],'gates':r['gates'],'energy_pct':r['energy_relative']*100,
        'population_pct':r['errors']['population']['relative_l2']*100} for r in matched],'estimator_validation':estimator},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/trajectory-control-2026-09-22')
    p.add_argument('--worker',choices=('field','monitor','corrected'));p.add_argument('--n',type=int,default=13)
    p.add_argument('--dt',type=float,default=.01);p.add_argument('--collect',action='store_true');args=p.parse_args()
    if args.worker:worker(args.n,args.worker,args.dt,args.output);return
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.collect:
        jobs=[(f'{kind}-{n}',n,kind,.01) for n in (13,21,29) for kind in ('field','monitor','corrected')]
        jobs += [(f'{kind}-21-halfdt',21,kind,.005) for kind in ('field','corrected')]
        for name,n,kind,dt in jobs:
            folder=args.output/name
            if (folder/'report.json').exists():raise FileExistsError('Use a fresh output directory.')
            began=perf_counter();subprocess.run([sys.executable,__file__,'--worker',kind,'--n',str(n),'--dt',str(dt),'--output',str(folder)],cwd=ROOT,check=True)
            dump(folder/'process.json',{'launch_to_exit_s':perf_counter()-began});print(name,'finished',flush=True)
    collect(args.output)


if __name__=='__main__':main()
