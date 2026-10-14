"""All-block localized reduction, accuracy gates and full-promotion control."""
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
from coupled_transport import initial,dump,compare
from response_cache import array_storage


def worker(n,kind,dt,budget,cap,output,reset=False):
    output.mkdir(parents=True,exist_ok=True);start=perf_counter();model,fields=initial(n);g=model.grid
    chart=LocalizedTransport(model,fields,budgets=(budget,)*3,caps=(cap,)*3,reset_at_cap=reset) if kind=='localized' else None
    state=model.evaluate(*fields) if chart is None else chart.state(project=False)
    initial_energy=state['energy'];mass=g.integrate(fields[1]);timings={'initialization_s':perf_counter()-start,'online_s':0.,'compilation_s':0.}
    times=[0.];energies=[initial_energy];inventories=[mass];counts=[[n**3]*3 if chart is None else [0]*3]
    events=[];residuals=[];t=0.;steps=0;backtracks=0
    while t<1-1e-12:
        began=perf_counter()
        if chart is not None:
            state,added=chart.activate();events.extend(added)
            residuals.append([p['relative_residual'] for p in state['projections']])
            state,used,trials=chart.step(min(dt,1-t),current=state);fields=state['fields']
        else:fields,state,used,trials=model.step(fields,min(dt,1-t),current=state)
        timings['online_s']+=perf_counter()-began;t+=used;steps+=1;backtracks+=trials-1
        times.append(t);energies.append(state['energy']);inventories.append(g.integrate(fields[1]))
        counts.append([n**3]*3 if chart is None else [n**3 if chart.full[b] else len(chart.modes[b]) for b in range(3)])
    timings['other_evolution_s']=perf_counter()-start-sum(timings.values())
    stored=array_storage(model,chart,state,fields)
    began=perf_counter();direct=model.evaluate(*fields);tensor=model.tensor_readback(*fields)
    direct_energy=sum(v for k,v in direct['components'].items() if k!='mechanical_and_core')+tensor['energy']+tensor['core_energy']
    force_error=max(float(np.max(abs(a-b))) for a,b in zip(state['forces'],direct['forces']))
    energy_error=abs(state['energy']-direct_energy)
    timings['tensor_validation_s']=perf_counter()-began
    report={'n':n,'kind':kind,'dt':dt,'budget':budget,'cap':cap,'reset_at_cap':reset,'end_time':t,'steps':steps,'backtracks':backtracks,
        'initial_energy':initial_energy,'final_energy':state['energy'],'max_energy_increase':float(max(0,np.max(np.diff(energies)))),
        'inventory_max_absolute_error':float(max(abs(np.array(inventories)-mass))),'tensor_energy_absolute_error':energy_error,
        'direct_force_max_error':force_error,'equilibrium_rms':tensor['equilibrium_rms'],'divergence_relative':tensor['divergence_relative'],
        'plastic_trace':tensor['plastic_trace'],'final_coordinates':counts[-1],'final_full_blocks':[True]*3 if chart is None else chart.full,
        'maximum_used_residuals':np.max(residuals,axis=0).tolist() if residuals else None,'events':events,
        'persistent_array_bytes':stored,'timings':timings,'environment':{'python':platform.python_version(),'numpy':np.__version__,
        'scipy':scipy.__version__,'platform':platform.platform(),'threads':1},
        'parameters':{'length':12.,'initial_core':.8,'profile_scale':float(np.sqrt(2)),'core_coefficient':.3,'field_mobility':10.,'epsilon_xz':.06}}
    report['checks']={'energy_descent':report['max_energy_increase']<1e-10,'inventory':report['inventory_max_absolute_error']<1e-10,
        'positive_population':0<fields[1].min() and fields[1].max()<1,'full_tensor_energy':energy_error<1e-10,
        'direct_coupled_forces':force_error<1e-10,'mechanical_equilibrium':tensor['equilibrium_rms']<1e-10,
        'burgers_closure':tensor['divergence_relative']<1e-10,'plastic_volume':abs(tensor['plastic_trace'])<1e-10,
        'all_block_budgets':not residuals or np.max(residuals)<=budget,
        'activation_transfer_energy':all(abs(e['energy_jump'])<1e-10 for e in events),
        'reset_preserves_state':all(e.get('state_jump_max',0)==0 for e in events)}
    report['checks']={k:bool(v) for k,v in report['checks'].items()}
    if chart is not None:
        report['patch_history']=chart.patch_history
        report['stored_basis_modes']=[len(m) for m in chart.modes]
        report['basis_support_fraction']=[[float(np.count_nonzero(mode)/n**3) for mode in modes] for modes in chart.modes]
    began=perf_counter()
    np.savez_compressed(output/'fields.npz',phase=fields[0],population=fields[1],gamma=fields[2],alpha=tensor['alpha'],stress=tensor['stress'],
        time=times,energy=energies,inventory=inventories,coordinates=counts,residuals=np.array(residuals))
    if chart is not None:chart.save(output/'restart.npz')
    timings['array_io_s']=perf_counter()-began
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;report['peak_rss_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    began=perf_counter();dump(output/'report.json',report);dump(output/'output-timing.json',{'report_io_s':perf_counter()-began})


def collect(output):
    reports={p.parent.name:json.loads(p.read_text()) for p in sorted(output.glob('*/report.json'))}
    for name,r in reports.items():
        r['timings'].update(json.loads((output/name/'output-timing.json').read_text()))
        r.update(json.loads((output/name/'process.json').read_text()))
    pairs=[(f'localized-{n}',f'field-{n}') for n in (13,21,29)]
    pairs += [('localized-21-tight','field-21'),('localized-21-halfdt','field-21-halfdt'),('zero-cap-21','field-21')]
    matched=[compare(output,a,b) for a,b in pairs]
    for r in matched:
        r['gates']={key:r['errors'][key]['relative_l2']<.01 for key in ('phase','population','alpha')}
        r['gates']['energy']=r['energy_relative']<.001
    time=[compare(output,f'{kind}-21',f'{kind}-21-halfdt') for kind in ('field','localized')]
    source=[Path(__file__),ROOT/'src/bmvr/localized_transport.py',ROOT/'src/bmvr/coupled_transport.py',
        ROOT/'src/bmvr/continuum3d.py',ROOT/'src/bmvr/response_cache.py',ROOT/'unified_multiscale_defect_framework.md']
    result={'reports':reports,'matched':matched,'time':time,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        'scope':'Localized trial modes for all blocks. Full-grid quadrature and dense response storage remain. Timings are single runs.'}
    dump(output/'summary.json',result)
    print(json.dumps([{'run':r['run'],'errors_pct':{k:100*v['relative_l2'] for k,v in r['errors'].items()},'gates':r['gates']} for r in matched],indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/localized-transport-2026-09-22')
    p.add_argument('--worker',choices=('field','localized'));p.add_argument('--n',type=int,default=13);p.add_argument('--dt',type=float,default=.01)
    p.add_argument('--budget',type=float,default=.05);p.add_argument('--cap',type=int,default=32);p.add_argument('--collect',action='store_true')
    p.add_argument('--reset',action='store_true');args=p.parse_args()
    if args.worker:worker(args.n,args.worker,args.dt,args.budget,args.cap,args.output,args.reset);return
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.collect:
        jobs=[(f'{kind}-{n}',n,kind,.01,.05,32) for n in (13,21,29) for kind in ('field','localized')]
        jobs += [('localized-21-tight',21,'localized',.01,.025,32),('field-21-halfdt',21,'field',.005,.05,32),
            ('localized-21-halfdt',21,'localized',.005,.05,32),('zero-cap-21',21,'localized',.01,.05,0)]
        for name,n,kind,dt,budget,cap in jobs:
            folder=args.output/name
            if (folder/'report.json').exists():raise FileExistsError('Use a fresh output directory.')
            began=perf_counter();subprocess.run([sys.executable,__file__,'--worker',kind,'--n',str(n),'--dt',str(dt),
                '--budget',str(budget),'--cap',str(cap),'--output',str(folder)]+(['--reset'] if args.reset else []),cwd=ROOT,check=True)
            dump(folder/'process.json',{'launch_to_exit_s':perf_counter()-began});print(name,'finished',flush=True)
    collect(args.output)


if __name__=='__main__':main()
