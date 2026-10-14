"""Independent Nye-field evolution, known-chart transfer and fixed-parameter refinement."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse,hashlib,json,platform,subprocess
from bmvr import _resource as resource
from time import perf_counter
from dataclasses import replace
import numpy as np
import scipy
from bmvr.continuum3d import Cube
from bmvr.loops3d import circular_loop
from bmvr.representation3d import NyeTransfer
from three_dimensional import volume_project


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def setup(n):
    grid=Cube(12,n)
    x,y,z=grid.coordinates()
    background=np.eye(3)[...,None,None,None]*(.03*np.cos(2*np.pi*x/12)*np.sin(2*np.pi*y/12))
    model=NyeTransfer(grid,background=background,glide_normal=np.array([0.,0.,1.]))
    loop=circular_loop(radius=2,count=8,center=(.2,.1,0),core_tension=0,glide_normal=np.array([0.,0.,1.]))
    return model,loop


def run(n,dt,output,switch=False):
    output.mkdir(parents=True,exist_ok=True)
    started=perf_counter();model,loop=setup(n);loops=[loop];correction=np.zeros((3,3))
    beta=model.promote(loops);initial=model.evaluate(beta);state=initial
    init_s=perf_counter()-started
    times,energies,means,divergence=[0.],[state['energy']],[state['mean_beta']],[state['divergence_relative']]
    events=[];mode='line' if switch else 'field';t=0.;steps=0;solver_s=transfer_s=0.
    eps=1e-4;line=model.line_state(loops)
    plus=model.promote([loop.moved(loop.nodes+eps*line['line_rates'][0])])
    minus=model.promote([loop.moved(loop.nodes-eps*line['line_rates'][0])])
    chart_alpha_rate=-model.curl((plus-minus)/(2*eps))
    full_alpha_rate=-model.curl(initial['rate'])
    rate_discrepancy=float(np.linalg.norm(chart_alpha_rate-full_alpha_rate)/np.linalg.norm(full_alpha_rate))
    while t<.4-1e-12:
        next_event=.1 if switch and t<.1-1e-12 else .3 if switch and t<.3-1e-12 else .4
        requested=min(dt,next_event-t)
        started=perf_counter()
        if mode=='field':
            beta,state,used,trials=model.field_step(beta,requested)
        else:
            loops,state,used,trials=model.line_step(loops,correction,requested)
            beta=model.promote(loops,correction)
        solver_s+=perf_counter()-started;t+=used;steps+=1
        if switch and abs(t-.1)<1e-10:
            started=perf_counter();before=state['energy'];beta=model.promote(loops,correction);state=model.evaluate(beta);mode='field'
            events.append({'time':t,'kind':'promotion','energy_jump':state['energy']-before,'accepted':True})
            transfer_s+=perf_counter()-started
        if switch and abs(t-.3)<1e-10:
            started=perf_counter();fit=model.recover(beta,loops)
            events.append({'time':t,'kind':'reconstruction',**{k:v for k,v in fit.items() if k not in ('loops','mean_correction','beta')}})
            if fit['accepted']:
                loops,correction,beta=fit['loops'],fit['mean_correction'],fit['beta'];mode='line';state=model.evaluate(beta)
            transfer_s+=perf_counter()-started
        times.append(t);energies.append(state['energy']);means.append(state['mean_beta']);divergence.append(state['divergence_relative'])
    started=perf_counter()
    recovery=model.recover(beta,[loop])
    contradictory=replace(loop,burgers=np.array([0.,.2,0.]))
    incompatible_beta=model.promote([loop,contradictory])
    refused=model.recover(incompatible_beta,[loop],max_evaluations=20)
    model.save(output/'checkpoint.npz',beta,[loop],t,steps)
    restored,rb,rh,clock=NyeTransfer.load(output/'checkpoint.npz')
    original_next=model.field_step(beta,dt);restored_next=restored.field_step(rb,dt)
    diagnostic_s=perf_counter()-started
    report={'grid':{'n':n,'length':12,'h':12/n},'dt':dt,'end':t,'steps':steps,'mode':'switch' if switch else 'field',
        'parameters':{'core_coefficient':.3,'field_mobility':10.,'line_mobility':loop.mobility,'initial_core':loop.core,'loop':loop.to_dict()},
        'initial_energy':initial['energy'],'final_energy':state['energy'],
        'max_recorded_energy_increase':float(max(0,np.max(np.diff(energies)))),
        'alpha_change_relative':float(np.linalg.norm(state['alpha']-initial['alpha'])/np.linalg.norm(initial['alpha'])),
        'initial_chart_full_alpha_rate_discrepancy':rate_discrepancy,
        'max_plastic_trace':float(np.max(np.abs(np.trace(np.array(means),axis1=1,axis2=2)))),
        'max_divergence_relative':float(max(divergence)),
        'events':events,'final_recovery':{k:v for k,v in recovery.items() if k not in ('loops','mean_correction','beta')},
        'incompatible_recovery':{k:v for k,v in refused.items() if k not in ('loops','mean_correction','beta')},
        'timings':{'initialization_s':init_s,'solver_s':solver_s,'transfer_s':transfer_s,'diagnostics_and_checkpoint_io_s':diagnostic_s},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'threads':1}}
    report['checks']={'energy_descent':report['max_recorded_energy_increase']<1e-10,
        'periodic_burgers_closure':report['max_divergence_relative']<1e-10,
        'glide_plastic_volume':report['max_plastic_trace']<1e-10,
        'independent_evolution':report['alpha_change_relative']>1e-5,
        'incompatible_chart_refused':not refused['accepted'] and np.array_equal(refused['beta'],incompatible_beta),
        'field_restart_identical':np.array_equal(original_next[0],restored_next[0])}
    if switch:
        report['checks']['actual_round_trip']=len(events)==2 and all(e['accepted'] for e in events)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    started=perf_counter()
    np.savez_compressed(output/'fields.npz',beta=beta,alpha=state['alpha'],stress=state['stress'],initial_alpha=initial['alpha'],
                        time=times,energy=energies,mean_beta=means)
    report['timings']['array_io_s']=perf_counter()-started
    dump(output/'report.json',report)
    print(output.name,report['checks'],'reconstruction',recovery['accepted'],flush=True)


def spectral_restrict(field, old_n, new_n, length):
    """Project odd periodic collocation fields, retaining the cell-center phase."""
    modes=np.rint(np.fft.fftfreq(new_n)*new_n).astype(int)
    coefficients=np.fft.fftn(field,axes=(-3,-2,-1))/old_n**3
    for axis in (-3,-2,-1):coefficients=np.take(coefficients,modes % old_n,axis=axis)
    phase=np.exp(2j*np.pi*(modes[:,None,None]+modes[None,:,None]+modes[None,None,:])*(1/new_n-1/old_n)/2)
    return np.fft.ifftn(coefficients*phase,axes=(-3,-2,-1)).real*new_n**3


def compare(output,first,second):
    a=json.loads((output/first/'report.json').read_text());b=json.loads((output/second/'report.json').read_text())
    errors={};cell_average_diagnostic={}
    with np.load(output/first/'fields.npz') as data,np.load(output/second/'fields.npz') as ref:
        for key in ('alpha','stress'):
            averaged=volume_project(ref[key],b['grid']['n'],a['grid']['n'],12)
            cell_average_diagnostic[key]=float(np.linalg.norm(data[key]-averaged)/np.linalg.norm(averaged))
            mapped=spectral_restrict(ref[key],b['grid']['n'],a['grid']['n'],12)
            error=data[key]-mapped
            errors[key]={'l1':float(np.mean(np.abs(error))),'l2':float(np.sqrt(np.mean(error**2))),
                         'linf':float(np.max(np.abs(error))),'relative_l2':float(np.linalg.norm(error)/np.linalg.norm(mapped))}
    return {'run':first,'reference':second,'errors':errors,'cell_average_diagnostic_relative_l2':cell_average_diagnostic,
            'comparison':'Fourier restriction with corrected cell-center origin; cell-average mismatch retained separately.',
            'energy_relative':abs(a['final_energy']/b['final_energy']-1)}


def collect(output):
    reports={p.parent.name:json.loads(p.read_text()) for p in sorted(output.glob('*/report.json'))}
    space=[compare(output,f'space-{n}','space-25') for n in (9,13,17)]
    time=[compare(output,f'time-{dt}','time-0.0025') for dt in (.04,.02,.01,.005)]
    summary={'reports':reports,'space':space,'time':time,'switch_vs_field':compare(output,'switch','space-13'),
        'monotone_errors':{label:{key:all(a['errors'][key]['l2']>b['errors'][key]['l2'] for a,b in zip(rows,rows[1:])) for key in ('alpha','stress')} for label,rows in [('space',space),('time',time)]},
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/bmvr/representation3d.py',Path(__file__),ROOT/'unified_multiscale_defect_framework.md']},
        'limits':['Known-connectivity inverse only; arbitrary topology is not reconstructed.','Distinct drag closures give nonzero chart/full rate discrepancy.','No performance advantage is inferred from these single-run verification timings.']}
    dump(output/'summary.json',summary)
    print(summary['monotone_errors'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/representation-transfer-2026-09-13')
    p.add_argument('--worker',action='store_true');p.add_argument('--n',type=int,default=13);p.add_argument('--dt',type=float,default=.02)
    p.add_argument('--switch',action='store_true');p.add_argument('--collect',action='store_true');args=p.parse_args()
    if args.worker:run(args.n,args.dt,args.output,args.switch);return
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.collect:
        matrix=[(f'space-{n}',n,.02,False) for n in (9,13,17,25)]
        matrix += [(f'time-{dt}',13,dt,False) for dt in (.04,.02,.01,.005,.0025)]+[('switch',13,.02,True)]
        for name,n,dt,switch in matrix:
            if (args.output/name/'report.json').exists():raise FileExistsError('Choose a fresh output directory.')
            command=[sys.executable,__file__,'--worker','--n',str(n),'--dt',str(dt),'--output',str(args.output/name)]
            if switch:command.append('--switch')
            subprocess.run(command,check=True,cwd=ROOT)
    collect(args.output)


if __name__=='__main__':main()
