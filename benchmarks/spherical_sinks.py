"""Fixed-width spherical capture, reciprocal storage, and convergence studies."""

import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bmvr import _resource as resource
from bmvr.spherical_sink import SphericalSink,collocation_reference,robin_capture,exact_discrete_evolution,reaction_profile


def diagnose_strong_capture(folder):
    """Retain the failed coarse-grid gate and isolate reaction quadrature error."""
    from scipy.integrate import quad
    rows=[]
    for n in (32,64,128,256,512,1024):
        model=SphericalSink(cells=n,reactivity=10.)
        concentration,rate,_=model.steady()
        reference,exact,_=collocation_reference(model.r,reactivity=10.,tolerance=1e-10)
        row={'n':n,'signed_capture_relative_error':(rate-exact)/exact,
             'layer_integral_relative_error':(model.exchange.sum()-4*np.pi*4*10)/(4*np.pi*4*10),
             'profile_relative_l2':float(np.sqrt(model.vol@((concentration-reference)**2)/(model.vol@(reference**2))))}
        weights=np.zeros(n)
        for i,(left,right) in enumerate(zip(model.edges[:-1],model.edges[1:])):
            left=max(left,1.5);right=min(right,2.5)
            if left<right:
                weights[i]=quad(lambda r:4*np.pi*r*r*reaction_profile(r,reactivity=10.),left,right,epsabs=1e-11)[0]
        model.exchange=weights
        _,integrated_rate,_=model.steady()
        row['cell_integrated_reaction_capture_relative_error']=(integrated_rate-exact)/exact
        rows.append(row)
    (folder/'strong_capture_refinement.json').write_text(json.dumps({
        'purpose':'Diagnose retained nonmonotone coarse-grid capture error; original gate and default midpoint scheme unchanged.',
        'ablation':'Replace only midpoint reaction weights by integrated shell weights; diffusion stencil and unknown locations unchanged.',
        'rows':rows},indent=2)+'\n')


def main(folder):
    folder.mkdir(parents=True,exist_ok=True)
    wall=perf_counter();timings={'reference_s':0.,'assembly_s':0.,'solver_s':0.,'diagnostics_s':0.}
    report={'parameters':{'radius':2.,'length':8.,'width':1.,'diffusion':1.,'chi':1.,
                         'concentration_infinity':1.,'fixed_surface_equilibrium':0.,'capacity_fraction_of_volume':.2},
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                           'platform':platform.platform(),'threads':1},'steady':[],'width_limit':[],'closed':[],'time_refinement':[]}
    arrays={};checks={}
    for reactivity in (.1,1.,10.):
        tag=str(reactivity)
        start=perf_counter()
        _,loose,_=collocation_reference(np.array([0.]),reactivity=reactivity,tolerance=1e-8)
        _,tight,reference_meta=collocation_reference(np.array([0.]),reactivity=reactivity,tolerance=1e-10)
        timings['reference_s']+=perf_counter()-start
        reference_delta=abs(loose-tight)/abs(tight)
        checks['reference_tolerance_'+tag]=reference_delta<1e-7
        for n in (32,64,128,256):
            start=perf_counter();model=SphericalSink(cells=n,reactivity=reactivity);timings['assembly_s']+=perf_counter()-start
            start=perf_counter();c,capture,incoming=model.steady();timings['solver_s']+=perf_counter()-start
            start=perf_counter();exact,_,_=collocation_reference(model.r,reactivity=reactivity,tolerance=1e-10);timings['reference_s']+=perf_counter()-start
            start=perf_counter();difference=c-exact
            row={'n':n,'h':8/n,'reactivity':reactivity,'profile_relative_l2':float(np.sqrt(model.vol@(difference**2)/(model.vol@(exact**2)))),
                 'profile_l1':float(model.vol@np.abs(difference)/model.vol.sum()),'profile_linf':float(np.max(np.abs(difference))),
                 'capture_coefficient':capture,'reference_capture_coefficient':tight,'capture_relative_error':abs(capture-tight)/abs(tight),
                 'capture_length':capture,'sink_strength_per_volume':capture/model.vol.sum(),
                 'steady_flux_balance_relative':abs(capture-incoming)/abs(capture),
                 'reference_tolerance_relative_change':reference_delta,'reference':reference_meta}
            report['steady'].append(row)
            arrays[f'r_{n}']=model.r;arrays[f'c_{tag}_{n}']=c;arrays[f'exact_{tag}_{n}']=exact
            checks[f'steady_balance_{tag}_{n}']=row['steady_flux_balance_relative']<1e-9
            timings['diagnostics_s']+=perf_counter()-start
        rows=[r for r in report['steady'] if r['reactivity']==reactivity]
        checks['profile_convergence_'+tag]=all(a['profile_relative_l2']>b['profile_relative_l2'] for a,b in zip(rows,rows[1:]))
        checks['capture_convergence_'+tag]=all(a['capture_relative_error']>b['capture_relative_error'] for a,b in zip(rows,rows[1:]))
        checks['finest_profile_'+tag]=rows[-1]['profile_relative_l2']<=.002
        checks['finest_capture_'+tag]=rows[-1]['capture_relative_error']<=.002
        for width in (1.,.5,.25):
            start=perf_counter();_,rate,meta=collocation_reference(np.array([0.]),width=width,reactivity=reactivity,tolerance=1e-10)
            timings['reference_s']+=perf_counter()-start
            limit=robin_capture(reactivity=reactivity)
            report['width_limit'].append({'width':width,'reactivity':reactivity,'capture_coefficient':rate,'robin_limit':limit,
                                          'relative_difference':abs(rate-limit)/limit,'reference':meta})

    start=perf_counter();model=SphericalSink(cells=128);capacity=.2*model.vol.sum();mass,stiffness=model.closed_system(capacity)
    step=model.stepper(capacity,.25);timings['assembly_s']+=perf_counter()-start
    for name,c0,s0 in (('capture',1.,0.),('release',.2,1.)):
        state=np.r_[np.full(model.cells,c0),s0];initial=state.copy();total=float(mass@state);equilibrium=total/mass.sum()
        history=[];max_balance=0.;max_energy_increment=0.;min_population=float(state.min());previous=.5*mass@(state**2)
        for i in range(641):
            start=perf_counter();energy=float(.5*mass@(state**2))
            max_balance=max(max_balance,abs(mass@state-total)/total)
            max_energy_increment=max(max_energy_increment,energy-previous)
            min_population=min(min_population,float(state.min()));previous=energy
            history.append([i*.25,float(model.vol@state[:-1]),float(capacity*state[-1]),energy,float(state[-1])])
            timings['diagnostics_s']+=perf_counter()-start
            if i<640:
                start=perf_counter();state=step(state);timings['solver_s']+=perf_counter()-start
        error=float(np.max(np.abs(state-equilibrium))/equilibrium)
        row={'name':name,'n':128,'dt':.25,'end':160.,'steps':640,'capacity':float(capacity),
             'initial_bulk_concentration':c0,'initial_surface_concentration':s0,'equilibrium_concentration':equilibrium,
             'equilibrium_relative_linf_error':error,'relative_inventory_drift':float(max_balance),
             'max_energy_increase':float(max_energy_increment),'minimum_population':min_population,
             'stored_population_change':float(capacity*(state[-1]-initial[-1]))}
        report['closed'].append(row);arrays[name+'_history']=np.array(history);arrays[name+'_final']=state
        checks[name+'_inventory']=max_balance<1e-10
        checks[name+'_dissipation']=max_energy_increment<1e-10*max(1.,history[0][3])
        checks[name+'_positivity']=min_population>=-1e-12
        checks[name+'_equilibrium']=error<.001
        checks[name+'_exchange_direction']=row['stored_population_change']*(1 if name=='capture' else -1)>0

    initial=np.r_[np.ones(model.cells),0.]
    start=perf_counter();exact=exact_discrete_evolution(model,capacity,initial,8.);timings['reference_s']+=perf_counter()-start
    arrays['time_reference']=exact
    for dt in (.5,.25,.125,.0625):
        start=perf_counter();step=model.stepper(capacity,dt);timings['assembly_s']+=perf_counter()-start
        start=perf_counter();state=initial.copy()
        for i in range(round(8/dt)):state=step(state)
        elapsed=perf_counter()-start;timings['solver_s']+=elapsed
        difference=state-exact
        report['time_refinement'].append({'dt':dt,'steps':round(8/dt),'solver_s':elapsed,
            'combined_relative_l2':float(np.sqrt(mass@(difference**2)/(mass@(exact**2)))),
            'bulk_linf':float(np.max(np.abs(difference[:-1]))),'surface_absolute_error':float(abs(difference[-1]))})
        arrays[f'time_{dt}']=state
    for key in ('combined_relative_l2','bulk_linf','surface_absolute_error'):
        checks['time_convergence_'+key]=all(a[key]>b[key] for a,b in zip(report['time_refinement'],report['time_refinement'][1:]))
    report['checks']={key:bool(value) for key,value in checks.items()};report['all_checks_pass']=all(checks.values())
    report['limits']=['Radial symmetry and a fixed permeable spherical trapping layer; not an excluded-volume pore.',
                      'Quadratic storage closure without site saturation or material calibration.',
                      'Width-limit studies are separate from fixed-width spatial convergence; no performance superiority claim.']
    start=perf_counter();diagnose_strong_capture(folder);timings['strong_capture_diagnostic_s']=perf_counter()-start
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_before_io_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    timings['unclassified_pre_io_s']=max(0.,perf_counter()-wall-sum(timings.values()))
    start=perf_counter();np.savez_compressed(folder/'fields.npz',**arrays);timings['array_io_s']=perf_counter()-start
    timings['compilation_s']=0.;report['timings']=timings
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':int(sum(checks.values())),'total':len(checks),'failed':[k for k,v in checks.items() if not v],
                      'closed':report['closed'],'steady_finest':[r for r in report['steady'] if r['n']==256]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/spherical-sinks-2026-09-13')
    main(parser.parse_args().output.resolve())
