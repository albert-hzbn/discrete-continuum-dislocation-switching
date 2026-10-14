"""Longer two-loop comparison of one continuum law and its adaptive charts."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,hashlib,json,platform,subprocess,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bmvr import _resource as resource
import numpy as np
import scipy
from bmvr.continuum3d import Cube
from bmvr.loops3d import circular_loop
from bmvr.transport_metric import PlanarTransport


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def setup():
    grid=Cube(12,13);x,y,z=grid.coordinates()
    model=PlanarTransport(grid,background=np.eye(3)[...,None,None,None]*.03*np.cos(2*np.pi*x/12)*np.sin(2*np.pi*y/12))
    loops=[circular_loop(radius=1.5,count=8,center=(center,0,0),burgers=(sign*.2,0,0),core_tension=0,
                         glide_normal=np.array([0.,0.,1.])) for center,sign in ((-2,1),(2,-1))]
    return model,loops


def run(kind,dt,output):
    output.mkdir(parents=True,exist_ok=True);started=perf_counter()
    model,loops=setup();details=[];amps=np.array([]);gamma=model.gamma_from_loops(loops)
    initial=model.evaluate(gamma);state=initial;initial_gamma=gamma.copy()
    initialization=perf_counter()-started;mode='field' if kind=='field' else 'chart'
    cap=0 if kind=='fallback' else 2
    times=[0.];energies=[state['energy']];traces=[float(np.trace(state['mean_beta']))];divergence=[state['divergence_relative']]
    sampled_residual=[];used_residual=[];events=[];coordinate_counts=[gamma.size if mode=='field' else 32]
    t=0.;steps=0;solver_s=audit_s=0.;maximum_detail_bytes=0
    while t<4-1e-12:
        started=perf_counter()
        if mode=='chart':
            if kind=='plain':current=model.chart_state(loops,details,amps)
            else:
                details,amps,current,event=model.enrich(loops,details,amps,budget=.05,max_details=cap)
                maximum_detail_bytes=max(maximum_detail_bytes,event['detail_storage_bytes'])
                if event['added_details'] or event['fallback_required']:
                    events.append({'time':t,**event})
                if event['fallback_required']:
                    promoted=current['gamma'].copy()
                    events[-1]['promotion_state_error']=float(np.max(np.abs(promoted-current['gamma'])))
                    gamma=promoted;mode='field'
            if mode=='chart':
                used_residual.append(current['projection']['relative_residual'])
                sampled_residual.append([t,current['projection']['relative_residual']])
        audit_s+=perf_counter()-started;started=perf_counter()
        if mode=='field':gamma,state,used,trials=model.field_step(gamma,min(dt,4-t))
        else:
            loops,amps,state,used,trials=model.chart_step(loops,details,amps,min(dt,4-t),current=current)
            gamma=state['gamma']
        solver_s+=perf_counter()-started;t+=used;steps+=1
        times.append(t);energies.append(state['energy']);traces.append(float(np.trace(state['mean_beta'])));divergence.append(state['divergence_relative'])
        coordinate_counts.append(gamma.size if mode=='field' else sum(2*len(loop.nodes) for loop in loops)+len(details))
    record={'kind':kind,'dt':dt,'end':t,'steps':steps,'final_mode':mode,
        'parameters':{'n':13,'length':12.,'core_coefficient':.3,'initial_core':.8,'field_mobility':10.,'budget':.05,'detail_cap':cap},
        'initial_energy':initial['energy'],'final_energy':state['energy'],
        'max_energy_increase':float(max(0,np.max(np.diff(energies)))),
        'max_plastic_trace':float(np.max(np.abs(traces))),'max_divergence_relative':float(max(divergence)),
        'maximum_used_chart_residual':float(max(used_residual)) if used_residual else None,
        'initial_used_chart_residual':float(used_residual[0]) if used_residual else None,
        'source_change_relative':float(np.linalg.norm(state['alpha']-initial['alpha'])/np.linalg.norm(initial['alpha'])),
        'maximum_detail_storage_bytes':maximum_detail_bytes,'full_scalar_state_bytes':gamma.nbytes,
        'final_active_coordinates':coordinate_counts[-1],'events':events,
        'timings':{'initialization_s':initialization,'audit_and_enrichment_s':audit_s,'solver_s':solver_s,'compilation_s':0.},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'threads':1}}
    record['checks']={'energy_descent':record['max_energy_increase']<1e-10,
        'burgers_closure':record['max_divergence_relative']<1e-10,'plastic_volume':record['max_plastic_trace']<1e-10,
        'nontrivial_evolution':record['source_change_relative']>1e-5,
        'activation_state_unchanged':all(abs(e['energy_jump'])<1e-12 and e.get('promotion_state_error',0)==0 for e in events)}
    record['kinetic_budget_met']=record['maximum_used_chart_residual'] is None or record['maximum_used_chart_residual']<=.05
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record['peak_rss_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    started=perf_counter()
    np.savez_compressed(output/'fields.npz',gamma=gamma,initial_gamma=initial_gamma,alpha=state['alpha'],stress=state['stress'],
                        time=times,energy=energies,coordinates=coordinate_counts,sampled_residual=np.array(sampled_residual).reshape(-1,2))
    record['timings']['array_io_s']=perf_counter()-started
    dump(output/'report.json',record)
    print(output.name,'mode',mode,'kinetic budget',record['kinetic_budget_met'],'events',len(events),flush=True)


def collect(output):
    reports={p.parent.name:json.loads(p.read_text()) for p in sorted(output.glob('*/report.json'))}
    comparisons=[]
    for first,second in [('plain','field'),('adaptive','field'),('fallback','field'),('adaptive-half','field-half'),('field','field-half'),('adaptive','adaptive-half')]:
        errors={}
        with np.load(output/first/'fields.npz') as a,np.load(output/second/'fields.npz') as b:
            for key in ('gamma','alpha','stress'):
                d=a[key]-b[key]
                errors[key]={'l1':float(np.mean(np.abs(d))),'l2':float(np.sqrt(np.mean(d*d))),
                             'linf':float(np.max(np.abs(d))),'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(b[key]),1e-30))}
            equal=bool(np.array_equal(a['gamma'],b['gamma']))
        comparisons.append({'run':first,'reference':second,'errors':errors,'gamma_identical':equal,
                            'energy_relative':abs(reports[first]['final_energy']/reports[second]['final_energy']-1)})
    summary={'reports':reports,'comparisons':comparisons,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/bmvr/transport_metric.py',Path(__file__),ROOT/'unified_multiscale_defect_framework.md']},
             'limits':['One planar Burgers family; no unknown topology reconstruction.','Dense correction storage and full audits are charged.','Single timing per configuration; no statistical speedup claim.']}
    dump(output/'summary.json',summary)
    print([(r['run'],r['reference'],r['errors']['alpha']['relative_l2']) for r in comparisons],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/common-transport-2026-09-13')
    p.add_argument('--worker',choices=('field','plain','adaptive','fallback'));p.add_argument('--dt',type=float,default=.02);p.add_argument('--collect',action='store_true');args=p.parse_args()
    if args.worker:run(args.worker,args.dt,args.output);return
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.collect:
        for name,kind,dt in [(kind,kind,.02) for kind in ('field','plain','adaptive','fallback')]+[('field-half','field',.01),('adaptive-half','adaptive',.01)]:
            if (args.output/name/'report.json').exists():raise FileExistsError('Choose a fresh output directory.')
            subprocess.run([sys.executable,__file__,'--worker',kind,'--dt',str(dt),'--output',str(args.output/name)],cwd=ROOT,check=True)
    collect(args.output)


if __name__=='__main__':main()
