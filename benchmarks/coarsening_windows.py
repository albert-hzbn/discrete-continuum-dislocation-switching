"""Longer CH windows, independent seeds, and separate numerical sensitivities."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform,subprocess,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bmvr import _resource as resource
from bmvr.grid import Grid
from bmvr.energy import DoubleWell
from bmvr.solvers import ch_step
from bmvr.diagnostics import structure_factor
from bmvr.coarsening import initial_coarsening_field,coarsening_windows,radial_structure
from bmvr.adaptivity import transfer


def worker(folder,length,n,dt,seed):
    folder.mkdir(parents=True,exist_ok=True);wall=perf_counter()
    grid=Grid.uniform(length,n,periodic=True);model=DoubleWell();symbol=grid.spectrum()
    field=initial_coarsening_field(length,n,seed);initial=field.copy();mean0=field.mean()
    initialization=perf_counter()-wall;solver=diagnostics=0.;history=[];arrays={'initial':initial}
    max_amplitude=float(np.max(np.abs(field)));max_mean_drift=0.
    steps=round(4096/dt);record_every=round(16/dt)
    def record(time):
        energy=model.value(grid,field);_,ell=structure_factor(grid,field)
        history.append([time,energy,float(field.mean()),ell,2*np.sqrt(2)/3*grid.area/energy,float(np.var(field))])
        if time in (256,512,1024,2048,4096):
            k,radial,counts,power=radial_structure(grid,field)
            arrays[f'field_{int(time)}']=field.copy();arrays[f'structure_{int(time)}']=power
            arrays[f'radial_{int(time)}']=radial;arrays['radial_k']=k;arrays['radial_counts']=counts
    record(0.)
    for step in range(1,steps+1):
        start=perf_counter();field=ch_step(grid,field,dt,symbol=symbol);solver+=perf_counter()-start
        start=perf_counter()
        max_amplitude=max(max_amplitude,float(np.max(np.abs(field))))
        max_mean_drift=max(max_mean_drift,abs(float(field.mean()-mean0)))
        if step%record_every==0:record(step*dt)
        diagnostics+=perf_counter()-start
    history=np.array(history);arrays['history']=history
    windows=coarsening_windows(history[:,0],history[:,3],history[:,4],length)
    metrics={'max_mean_drift':max_mean_drift,'max_sampled_energy_increase':float(max(0.,np.diff(history[:,1]).max())),
             'max_amplitude_every_step':max_amplitude,'final_spectral_length':history[-1,3],'final_energy_length':history[-1,4],
             'final_degrees_of_freedom':int(field.size),'steps':steps,'end':4096.}
    checks={'mean_conservation':max_mean_drift<1e-10,
            'recorded_energy_decay':metrics['max_sampled_energy_increase']<1e-9*max(1.,history[0,1]),
            'sufficient_energy_stability_bound':max_amplitude<=np.sqrt(5/3),
            'phase_separation':history[-1,5]>history[0,5],
            'net_length_growth':history[-1,3]>history[0,3]}
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report={'parameters':{'length':length,'n':n,'h':length/n,'dt':dt,'seed':seed,'W':1.,'kappa':1.,'mobility':1.,'stabilization':2.,
                          'initial_cutoff':1.2,'initial_rms':.04,'interface_tanh_scale':float(np.sqrt(2))},
            'metrics':metrics,'windows':windows,'checks':{k:bool(v) for k,v in checks.items()},'all_checks_pass':all(checks.values()),
            'peak_rss_before_io_bytes':int(peak if sys.platform=='darwin' else peak*1024),
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                           'platform':platform.platform(),'threads':1},
            'timings':{'initialization_s':initialization,'solver_s':solver,'diagnostics_s':diagnostics,'compilation_s':0.,
                       'unclassified_pre_io_s':max(0.,perf_counter()-wall-initialization-solver-diagnostics)},
            'limits':['Uniform PF coarsening study; no adaptive performance claim.','Sampled energy plus an every-step sufficient amplitude bound.',
                      'Finite-window and finite-ensemble exponent assessment; scale-separation failures retained.']}
    start=perf_counter();np.savez_compressed(folder/'fields.npz',**arrays);report['timings']['array_io_s']=perf_counter()-start
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(folder.name, sum(checks.values()),'/',len(checks),'numerical checks;',
          'last slopes',round(windows[-1]['spectral_exponent'],4),round(windows[-1]['energy_exponent'],4),
          'length/box',round(windows[-1]['max_length_over_box'],3),flush=True)


def collect(folder):
    reports={p.parent.name:json.loads(p.read_text()) for p in folder.glob('*/report.json')}
    summary={'reports':reports,'ensemble':[],'sensitivities':[],'fine_sensitivities':[],'interpretation_checks':{}}
    for length in (128,256):
        names=[f'L{length}-seed{seed}' for seed in (1701,2901,4701)]
        if not all(name in reports for name in names):continue
        for index in range(4):
            rows=[reports[name]['windows'][index] for name in names]
            result={'length':length,'start':rows[0]['start'],'end':rows[0]['end'],
                    'all_scale_separated':all(r['scale_separated'] for r in rows),
                    'max_length_over_box':max(r['max_length_over_box'] for r in rows)}
            for metric in ('spectral','energy'):
                values=[r[metric+'_exponent'] for r in rows]
                result[metric+'_mean']=float(np.mean(values));result[metric+'_seed_sd']=float(np.std(values,ddof=1))
            summary['ensemble'].append(result)
    baseline=folder/'L128-seed1701/fields.npz'
    if baseline.exists():
        with np.load(baseline) as base:
            for name,kind in (('mesh-refined','mesh'),('time-refined','time')):
                path=folder/name/'fields.npz'
                if not path.exists():continue
                with np.load(path) as fine:
                    late=base['history'][:,0]>=1024
                    row={'kind':kind,'maximum_relative_length_errors':{}}
                    for key,col in (('spectral',3),('energy',4)):
                        row['maximum_relative_length_errors'][key]=float(np.max(np.abs(base['history'][late,col]-fine['history'][late,col])/fine['history'][late,col]))
                    b=reports['L128-seed1701']['windows'];f=reports[name]['windows']
                    row['maximum_exponent_change']=max(abs(x[key+'_exponent']-y[key+'_exponent']) for x,y in zip(b[-2:],f[-2:]) for key in ('spectral','energy'))
                    state=fine['field_4096']
                    if kind=='mesh':state=transfer(Grid.uniform(128,256,True),Grid.uniform(128,128,True),state)
                    diff=base['field_4096']-state
                    row['final_field_errors']={'l1':float(np.mean(np.abs(diff))),'l2':float(np.sqrt(np.mean(diff**2))),'linf':float(np.max(np.abs(diff)))}
                    row['passes']=max(row['maximum_relative_length_errors'].values())<=.05 and row['maximum_exponent_change']<=.03
                    summary['sensitivities'].append(row)
    ensemble=summary['ensemble'];late=[r for r in ensemble if r['start']>=1024]
    checks=summary['interpretation_checks']
    required=[f'L{length}-seed{seed}' for length in (128,256) for seed in (1701,2901,4701)]+['mesh-refined','time-refined']
    checks['complete_matrix']=all(name in reports for name in required) and len(ensemble)==8 and len(summary['sensitivities'])==2
    checks['late_scale_separation']=len(late)==4 and all(r['all_scale_separated'] for r in late)
    checks['exponents_near_one_third']=len(late)==4 and all(abs(r[k+'_mean']-1/3)<=.05 for r in late for k in ('spectral','energy'))
    checks['numerical_sensitivity']=len(summary['sensitivities'])==2 and all(r['passes'] for r in summary['sensitivities'])
    if len(late)==4:
        checks['window_stability']=all(abs(late[i][k+'_mean']-late[i+1][k+'_mean'])<=.05 for i in (0,2) for k in ('spectral','energy'))
        checks['domain_stability']=all(abs(late[i][k+'_mean']-late[i+2][k+'_mean'])<=.05 for i in (0,1) for k in ('spectral','energy'))
    summary['finite_window_one_third_consistency']=all(checks.values())
    if 'fine-base' in reports:
        with np.load(folder/'fine-base/fields.npz') as base:
            for name,kind in (('fine-mesh','mesh'),('fine-time','time')):
                if name not in reports:continue
                with np.load(folder/name/'fields.npz') as fine:
                    late=base['history'][:,0]>=1024
                    row={'kind':kind,'maximum_relative_length_errors':{}}
                    for key,col in (('spectral',3),('energy',4)):
                        row['maximum_relative_length_errors'][key]=float(np.max(np.abs(base['history'][late,col]-fine['history'][late,col])/fine['history'][late,col]))
                    b=reports['fine-base']['windows'];f=reports[name]['windows']
                    row['maximum_exponent_change']=max(abs(x[key+'_exponent']-y[key+'_exponent']) for x,y in zip(b[-2:],f[-2:]) for key in ('spectral','energy'))
                    state=fine['field_4096']
                    if kind=='mesh':state=transfer(Grid.uniform(128,512,True),Grid.uniform(128,256,True),state)
                    diff=base['field_4096']-state
                    row['final_field_errors']={'l1':float(np.mean(np.abs(diff))),'l2':float(np.sqrt(np.mean(diff**2))),'linf':float(np.max(np.abs(diff)))}
                    row['passes']=max(row['maximum_relative_length_errors'].values())<=.05 and row['maximum_exponent_change']<=.03
                    summary['fine_sensitivities'].append(row)
    summary['numerical_checks_passed']=sum(sum(r['checks'].values()) for r in reports.values())
    summary['numerical_checks_total']=sum(len(r['checks']) for r in reports.values())
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:summary[key] for key in ('ensemble','sensitivities','fine_sensitivities','interpretation_checks','numerical_checks_passed','numerical_checks_total')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/coarsening-2026-09-13')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--collect',action='store_true')
    parser.add_argument('--extended',action='store_true',help='Also run the finer sensitivity follow-up, preserving initial failures.')
    parser.add_argument('--length',type=int,default=128);parser.add_argument('--n',type=int,default=128)
    parser.add_argument('--dt',type=float,default=1.);parser.add_argument('--seed',type=int,default=1701)
    args=parser.parse_args();folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(folder,args.length,args.n,args.dt,args.seed)
    else:
        if not args.collect:
            jobs=[(f'L{length}-seed{seed}',length,length,1.,seed) for length in (128,256) for seed in (1701,2901,4701)]
            jobs += [('mesh-refined',128,256,1.,1701),('time-refined',128,128,.5,1701)]
            if args.extended:jobs += [('fine-base',128,256,.5,1701),('fine-mesh',128,512,.5,1701),('fine-time',128,256,.25,1701)]
            for name,length,n,dt,seed in jobs:
                subprocess.run([sys.executable,__file__,'--worker','--length',str(length),'--n',str(n),'--dt',str(dt),
                                '--seed',str(seed),'--output',str(folder/name)],check=True,cwd=ROOT)
        collect(folder)
