"""Quantitative grain, equilibrium precipitate, and pore verification studies."""

import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import argparse,json,platform,subprocess,sys
from time import perf_counter
import numpy as np
import scipy
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bmvr import _resource as resource
from bmvr.grid import Grid,circle
from bmvr.energy import DoubleWell,grain_energy,pore_energy,pore_forces
from bmvr.elasticity import coherent_elasticity
from bmvr.solvers import grain_step
from bmvr.microstructure import dilatational_elasticity,contour_radii,junction_angles,stationary_junction,grain_adjacency


def save(folder,report,arrays):
    folder.mkdir(parents=True,exist_ok=True)
    report['checks']={key:bool(value) for key,value in report['checks'].items()}
    report['all_checks_pass']=all(report['checks'].values())
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_before_io_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    report['environment']={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                            'platform':platform.platform(),'threads':1}
    start=perf_counter();np.savez_compressed(folder/'fields.npz',**arrays);report['timings']['array_io_s']=perf_counter()-start
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(folder.name,sum(report['checks'].values()),'/',len(report['checks']),'checks',flush=True)
    # Scientific failures are retained while independent studies continue.


def junction_run(folder,n,max_iterations=6000,initial_file=None):
    initial=None
    if initial_file is not None:
        with np.load(initial_file) as data:initial=data['phases'].copy()
    grid=Grid.uniform(32,n);start=perf_counter();phases,result=stationary_junction(grid,max_iterations=max_iterations,initial_phases=initial)
    solver=perf_counter()-start;start=perf_counter()
    rings=[junction_angles(grid,phases,r) for r in (4.,6.)]
    checks={'stationary':result['converged'],'simplex':np.max(np.abs(phases.sum(axis=0)-1))<1e-12 and phases.min()>=0,
            'energy':max(np.diff(result['energy']))<=1e-9*max(1,abs(result['energy'][0])),
            'angles':rings[-1]['mean_abs_error']<=5}
    save(folder,{'n':n,'h':32/n,'width':float(np.pi),'optimizer':result,'rings':rings,'checks':checks,
                  'continued_from':None if initial_file is None else str(initial_file),
                  'timings':{'equilibrium_solver_s':solver,'diagnostics_s':perf_counter()-start}},
         {'phases':phases,'xe':grid.xe,'ye':grid.ye})


def grain_statistics(folder,seed,n=96,end=40.):
    wall=perf_counter();length=64.;grid=Grid.uniform(length,n,periodic=True);rng=np.random.default_rng(seed)
    centers=(np.arange(6)+.5)*length/6-length/2
    sx,sy=np.meshgrid(centers,centers);seeds=np.c_[sx.ravel(),sy.ravel()]+rng.uniform(-.42,.42,(36,2))*length/6
    dx=(grid.X[None]-seeds[:,0,None,None]+length/2)%length-length/2
    dy=(grid.Y[None]-seeds[:,1,None,None]+length/2)%length-length/2
    score=-(dx*dx+dy*dy)/4;score-=score.max(axis=0)
    phases=np.exp(score);phases/=phases.sum(axis=0);initial=np.argmax(phases,axis=0)
    initialization=perf_counter()-wall;solver=diagnostics=0.;t=0.;steps=0
    times=[];energies=[];areas=[];counts=[];neighbors=[]
    def record():
        area=np.sum(phases*grid.vol,axis=(1,2));labels=np.argmax(phases,axis=0)
        active=(np.bincount(labels.ravel(),minlength=36)>0)&(area>1)
        times.append(t);energies.append(grain_energy(grid,phases));areas.append(area)
        counts.append(int(active.sum()));neighbors.append(grain_adjacency(grid,phases)[0])
    record()
    while t<end-1e-12:
        start=perf_counter();phases,used=grain_step(grid,phases,min(.05,end-t));solver+=perf_counter()-start
        t+=used;steps+=1
        if steps%20==0 or t>=end-1e-12:
            start=perf_counter();record();diagnostics+=perf_counter()-start
    times=np.array(times);areas=np.array(areas);neighbors=np.array(neighbors);counts=np.array(counts)
    mean_area=grid.area/counts;fit=times>=10
    coefficients=np.polyfit(times[fit],mean_area[fit],1);prediction=np.polyval(coefficients,times[fit])
    variance=float(np.sum((mean_area[fit]-mean_area[fit].mean())**2))
    r2=1-float(np.sum((prediction-mean_area[fit])**2)/variance) if variance>1e-20 else None
    observed=[];expected=[];eligible_sides=[]
    for i in range(1,len(times)-1):
        eligible=(times[i]>=10)&(areas[i]>4*np.pi**3)&(neighbors[i-1]==neighbors[i])&(neighbors[i]==neighbors[i+1])
        observed.extend(((areas[i+1]-areas[i-1])/(times[i+1]-times[i-1]))[eligible].tolist())
        expected.extend((np.pi/3*(neighbors[i]-6))[eligible].tolist());eligible_sides.extend(neighbors[i,eligible].tolist())
    observed,expected=np.array(observed),np.array(expected)
    valid=len(observed)>0 and np.linalg.norm(expected)>0
    normalized_error=float(np.linalg.norm(observed-expected)/np.linalg.norm(expected)) if valid else None
    slope=float(expected@observed/(expected@expected)) if valid else None
    metrics={'steps':steps,'seed':seed,'initial_grains':int(counts[0]),'final_grains':int(counts[-1]),
              'mean_area_slope':float(coefficients[0]),'mean_area_fit_r_squared':r2,'fit_interval':[10.,end],
              'qualified_von_neumann_samples':len(observed),'von_neumann_normalized_rms_error':normalized_error,
              'von_neumann_fitted_scale':slope,'phase_sum_error':float(np.max(np.abs(phases.sum(axis=0)-1))),
              'max_energy_increase':float(max(0.,max(np.diff(energies))))}
    checks={'simplex':metrics['phase_sum_error']<1e-12 and phases.min()>=0,
            'energy':metrics['max_energy_increase']<1e-8,'grain_elimination':counts[-1]<counts[0],
            'finite_time_growth':coefficients[0]>1e-8 and np.ptp(mean_area)>1e-6,'von_neumann_25_percent':valid and normalized_error<=.25}
    save(folder,{'n':n,'h':length/n,'width':float(np.pi),'metrics':metrics,'checks':checks,
                  'timings':{'initialization_s':initialization,'solver_s':solver,'diagnostics_s':diagnostics},
                  'limits':['Finite three-seed ensemble and finite-width grains; no asymptotic exponent claim.',
                            'Neighbor threshold 1 length unit; qualifying area >4*pi*width^2 and stable sides across three observations.']},
         {'time':times,'energy':np.array(energies),'areas':areas,'grain_count':counts,'neighbors':neighbors,
          'initial_labels':initial,'final_labels':np.argmax(phases,axis=0),'phases':phases,
          'observed_area_rate':observed,'expected_area_rate':expected,'eligible_sides':np.array(eligible_sides),
          'seeds':seeds,'xe':grid.xe,'ye':grid.ye})


def precipitate(folder,n,mode='condensed',aspect=1.):
    wall=perf_counter();grid=Grid.uniform(32,n,periodic=True);model=DoubleWell(k=(1.,1.4))
    common=Grid.uniform(32,257,periodic=True);mean=float(circle(common,6).mean())
    radial=np.sqrt((grid.X*np.sqrt(aspect))**2+(grid.Y/np.sqrt(aspect))**2)
    initial=np.tanh((6-radial)/np.sqrt(2));initial+=mean-initial.mean()
    elastic_fn=coherent_elasticity if mode=='fft' else dilatational_elasticity
    evaluations=0
    def objective(raw):
        nonlocal evaluations
        state=raw.reshape(grid.shape);state=state-state.mean()+mean
        elastic=elastic_fn(grid,state);force=model.force(grid,state)+elastic['force']
        evaluations+=1
        return model.value(grid,state)+elastic['energy'],((force-force.mean())*grid.vol).ravel()
    initialization=perf_counter()-wall;start=perf_counter()
    optimum=minimize(objective,initial.ravel(),jac=True,method='L-BFGS-B',
                      options={'ftol':1e-15,'gtol':1e-9,'maxiter':1800,'maxls':40})
    solve=perf_counter()-start;start=perf_counter()
    phase=optimum.x.reshape(grid.shape);phase=phase-phase.mean()+mean
    full=coherent_elasticity(grid,phase);reduced=dilatational_elasticity(grid,phase)
    force=model.force(grid,phase)+full['force'];force-=force.mean()
    radii=contour_radii(grid,phase);axis=contour_radii(grid,phase,angles=np.array([0,np.pi/2,np.pi/4]))
    metrics={'stationary_force_rms':float(np.sqrt(np.mean(force**2))),'inventory_error':abs(grid.integrate(phase)-mean*grid.area),
              'aspect_y_over_x':float(axis[1]/axis[0]),'expected_aspect':float(np.sqrt(1.4)),
              'axis_and_diagonal_radii':axis.tolist(),'energy':model.value(grid,phase)+full['energy'],
              'elastic_energy':full['energy'],'gradient_energy':grid.gradient_energy(phase,(1.,1.4)),
              'elastic_equilibrium_rms':full['equilibrium_residual'],'optimizer_success':bool(optimum.success),
              'optimizer_message':str(optimum.message),'iterations':int(optimum.nit),'evaluations':evaluations,
              'condensed_energy_error':abs(full['energy']-reduced['energy']),
              'condensed_force_error':float(np.max(np.abs(full['force']-reduced['force'])))}
    checks={'stationary':metrics['stationary_force_rms']<1e-5,'inventory':metrics['inventory_error']<1e-9,
            'elastic_equilibrium':metrics['elastic_equilibrium_rms']<1e-10,
            'exact_reduction':metrics['condensed_energy_error']<1e-10 and metrics['condensed_force_error']<1e-10,
            'aspect':abs(metrics['aspect_y_over_x']/np.sqrt(1.4)-1)<.02}
    save(folder,{'n':n,'h':32/n,'mode':mode,'initial_aspect':aspect,'profile_scale':float(np.sqrt(2)),
                  'parameters':{'W':1.,'gradient':[1.,1.4],'misfit':.08,'lam':2.,'shear':1.,'mean_strain':0.},
                  'mean_phase':mean,'metrics':metrics,'checks':checks,
                  'timings':{'initialization_s':initialization,'equilibrium_solver_s':solve,'full_fft_verification_and_output_fields_s':perf_counter()-start},
                  'limits':['Homogeneous isotropic plane strain and zero mean strain only. Optimizer iterations are not physical time.']},
         {'phase':phase,'initial':initial,'contour_radii':radii,'sxx':full['sxx'],'syy':full['syy'],'sxy':full['sxy'],
          'exx':full['exx'],'eyy':full['eyy'],'exy':full['exy'],'xe':grid.xe,'ye':grid.ye})


def pore_capillary(folder):
    wall=perf_counter();rows=[];gamma=np.sqrt(2)/6;exchange=10.
    for n in (128,192,256):
        grid=Grid.uniform(64,n)
        for radius in (6.,10.,14.):
            p=(circle(grid,radius)+1)/2;zero=np.zeros(grid.shape)
            energy=pore_energy(grid,zero,p);_,mu=pore_forces(grid,zero,p)
            mobility=exchange*p**2*(1-p)**2
            neutral=grid.integrate(mobility*mu)/grid.integrate(mobility)
            delta=.001;plus=(circle(grid,radius+delta)+1)/2;minus=(circle(grid,radius-delta)+1)/2
            capillary=(pore_energy(grid,zero,plus)-pore_energy(grid,zero,minus))/(grid.integrate(plus)-grid.integrate(minus))
            rates=[]
            for chemical in (0.,2*np.sqrt(2)/(5*radius)):
                measured=grid.integrate(mobility*(chemical-mu))
                expected=2*np.pi*exchange*(radius*chemical/(6*np.sqrt(2))-1/30)
                rates.append({'chemical_potential':chemical,'measured':measured,'expected':expected,'relative_error':abs(measured/expected-1)})
            rows.append({'n':n,'h':64/n,'radius':radius,'surface_energy_per_length':energy/(2*np.pi*radius),
                         'capillary_potential':capillary,'capillary_expected':gamma/radius,
                         'capillary_relative_error':abs(capillary/(gamma/radius)-1),
                         'initial_neutral_potential':neutral,'neutral_expected':np.sqrt(2)/(5*radius),
                         'neutral_relative_error':abs(neutral/(np.sqrt(2)/(5*radius))-1),'initial_area_rates':rates})
    finest=[r for r in rows if r['n']==256]
    checks={'capillary_2_percent':all(r['capillary_relative_error']<.02 for r in finest),
            'initial_neutral_2_percent':all(r['neutral_relative_error']<.02 for r in finest),
            'initial_rate_3_percent':all(x['relative_error']<.03 for r in finest for x in r['initial_area_rates'])}
    save(folder,{'rows':rows,'surface_energy_exact':float(gamma),'mobility':'10*p^2*(1-p)^2','checks':checks,
                  'timings':{'study_s':perf_counter()-wall},'limits':['Initial neutral rate is not equilibrium Gibbs-Thomson; its leading value is 6/5 times the energetic potential.']},
         {'radii':np.array([6.,10.,14.])})


def pore_time(folder,dt):
    wall=perf_counter();grid=Grid.uniform(32,96);rows=[];arrays={'xe':grid.xe,'ye':grid.ye}
    solver=0.
    for tag,concentration in [('low',.005),('high',.15)]:
        p=(circle(grid,6)+1)/2;n=np.full(grid.shape,concentration);inventory=grid.integrate(n+p)
        time=0.;energy=[pore_energy(grid,n,p)];times=[0.];areas=[grid.integrate(p)];mass=[inventory];steps=0
        while time<4-1e-12:
            start=perf_counter();un,up=pore_forces(grid,n,p);rate=10*p**2*(1-p)**2*(up-un)
            dn,dp=grid.lap(un)+rate,-rate;used=min(dt,4-time);old=energy[-1]
            for _ in range(30):
                nn,pp=n+used*dn,p+used*dp;value=pore_energy(grid,nn,pp)
                if nn.min()>=0 and pp.min()>=0 and pp.max()<=1 and value<=old+1e-11:break
                used*=.5
            else:raise RuntimeError('Pore step failed bounds or energy.')
            n,p=nn,pp;time+=used;steps+=1;solver+=perf_counter()-start
            times.append(time);energy.append(value);areas.append(grid.integrate(p));mass.append(grid.integrate(n+p))
        row={'tag':tag,'steps':steps,'area_change':areas[-1]-areas[0],
              'inventory_drift':float(np.max(np.abs(np.array(mass)-inventory))),
              'max_energy_increase':float(max(0.,max(np.diff(energy))))}
        rows.append(row)
        arrays.update({tag+'_p':p,tag+'_n':n,tag+'_time':np.array(times),tag+'_energy':np.array(energy),tag+'_area':np.array(areas)})
    checks={'inventory':all(r['inventory_drift']<1e-8 for r in rows),'energy':all(r['max_energy_increase']<1e-8 for r in rows),
            'low_shrinks':rows[0]['area_change']<0,'high_grows':rows[1]['area_change']>0}
    save(folder,{'dt':dt,'n':96,'h':32/96,'end':4.,'rows':rows,'checks':checks,
                  'timings':{'solver_s':solver,'setup_and_diagnostics_s':max(0.,perf_counter()-wall-solver)}},arrays)


def collect(folder):
    report={'date':'2026-09-13'}
    reference=folder/'precipitate-129'
    if reference.exists():
        with np.load(reference/'fields.npz') as fine:
            report['precipitate_contour_errors']=[]
            for n in (33,65,97):
                path=folder/f'precipitate-{n}'
                if not path.exists():continue
                with np.load(path/'fields.npz') as data:
                    error=data['contour_radii']-fine['contour_radii']
                    report['precipitate_contour_errors'].append({'n':n,'rms':float(np.sqrt(np.mean(error**2))),
                                                               'max':float(np.max(np.abs(error)))})
    ref=folder/'pore-time-0.0025'
    if ref.exists():
        with np.load(ref/'fields.npz') as fine:
            rows=[]
            for dt in (.02,.01,.005):
                path=folder/f'pore-time-{dt}'
                if not path.exists():continue
                with np.load(path/'fields.npz') as data:
                    errors={name:float(np.sqrt(np.mean((data[name]-fine[name])**2))) for name in ('low_p','low_n','high_p','high_n')}
                    rows.append({'dt':dt,'rms_errors':errors})
            report['pore_time_errors']=rows
    timings={}
    for mode in ('fft','condensed'):
        paths=list(folder.glob(f'timing-{mode}-*/report.json'))
        if paths:timings[mode]=[json.loads(p.read_text())['timings']['equilibrium_solver_s'] for p in paths]
    if len(timings)==2:
        totals={mode:[sum(json.loads(path.read_text())['timings'][key] for key in ('initialization_s','equilibrium_solver_s','full_fft_verification_and_output_fields_s'))
                      for path in folder.glob(f'timing-{mode}-*/report.json')] for mode in timings}
        with np.load(folder/'timing-fft-1/fields.npz') as full,np.load(folder/'timing-condensed-1/fields.npz') as reduced:
            difference=(reduced['phase']-full['phase'])/2
            agreement={'same_initial_state':bool(np.array_equal(full['initial'],reduced['initial'])),
                       'phase_fraction_l1':float(np.mean(np.abs(difference))),
                       'phase_fraction_l2':float(np.sqrt(np.mean(difference**2))),
                       'phase_fraction_linf':float(np.max(np.abs(difference))),
                       'phase_fraction_relative_l2':float(np.linalg.norm(difference)/np.linalg.norm((full['phase']+1)/2))}
        report['condensation_timing']={'runs':timings,'median_ratio':float(np.median(timings['fft'])/np.median(timings['condensed'])),
                                       'total_compute_runs':totals,'total_compute_ratio':float(np.median(totals['fft'])/np.median(totals['condensed'])),
                                       'agreement':agreement}
    final_junctions=[folder/name/'report.json' for name in ('junction-64','junction-96-refined','junction-128-refined')]
    report['final_junctions']=[{'folder':path.parent.name,**json.loads(path.read_text())} for path in final_junctions if path.exists()]
    report['long_grain_statistics']=[{'folder':path.parent.name,**json.loads(path.read_text())} for path in folder.glob('grains-long-*/report.json')]
    reports=[json.loads(p.read_text()) for p in folder.glob('*/report.json')]
    report['checks_passed']=sum(sum(r['checks'].values()) for r in reports)
    report['checks_total']=sum(len(r['checks']) for r in reports)
    report['failed_checks']=[{'folder':p.parent.name,'failed':[k for k,v in json.loads(p.read_text())['checks'].items() if not v]}
                            for p in folder.glob('*/report.json') if not json.loads(p.read_text())['all_checks_pass']]
    selected=['junction-64','junction-96-refined','junction-128-refined']
    selected += [f'grains-long-{seed}' for seed in (11,29,47)]
    selected += [f'precipitate-{n}' for n in (33,65,97,129)]+['precipitate-ellipse']
    selected += [f'timing-{mode}-{repeat}' for repeat in range(1,4) for mode in ('fft','condensed')]
    selected += ['pore-capillary']+[f'pore-time-{dt}' for dt in (.02,.01,.005,.0025)]
    missing=[name for name in selected if not (folder/name/'report.json').exists()]
    selected_reports=[json.loads((folder/name/'report.json').read_text()) for name in selected if name not in missing]
    report['selected_final_studies']={'folders':selected,'missing':missing,
        'checks_passed':sum(sum(r['checks'].values()) for r in selected_reports),
        'checks_total':sum(len(r['checks']) for r in selected_reports),
        'all_checks_pass':not missing and all(r['all_checks_pass'] for r in selected_reports)}
    if not missing:
        contour=report['precipitate_contour_errors']
        pore_errors=report['pore_time_errors']
        report['cross_study_checks']={
            'junction_angle_error_decreases':bool(np.all(np.diff([r['rings'][-1]['mean_abs_error'] for r in report['final_junctions']])<0)),
            'precipitate_contour_error_decreases':bool(np.all(np.diff([r['rms'] for r in contour])<0)),
            'pore_time_errors_decrease':all(np.all(np.diff([r['rms_errors'][key] for r in pore_errors])<0) for key in ('low_p','low_n','high_p','high_n')),
            'timing_initial_states_identical':report['condensation_timing']['agreement']['same_initial_state']}
    (folder/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print({key:report[key] for key in ('checks_passed','checks_total','failed_checks')},flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/microstructure-2026-09-13')
    parser.add_argument('--worker',choices=('junction','grains','precipitate','pore-capillary','pore-time'))
    parser.add_argument('--n',type=int,default=96);parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--mode',choices=('fft','condensed'),default='condensed');parser.add_argument('--aspect',type=float,default=1.)
    parser.add_argument('--dt',type=float,default=.02);parser.add_argument('--collect',action='store_true')
    parser.add_argument('--end',type=float,default=40.);parser.add_argument('--max-iterations',type=int,default=6000)
    parser.add_argument('--initial-file',type=Path)
    parser.add_argument('--extended',action='store_true',help='Also continue finer junction solves and evolve the same grain seeds to t=160.')
    parser.add_argument('--part',choices=('all','grains','precipitate','pore'),default='all');args=parser.parse_args()
    if args.worker:
        if args.worker=='junction':junction_run(args.output,args.n,args.max_iterations,args.initial_file)
        elif args.worker=='grains':grain_statistics(args.output,args.seed,args.n,args.end)
        elif args.worker=='precipitate':precipitate(args.output,args.n,args.mode,args.aspect)
        elif args.worker=='pore-capillary':pore_capillary(args.output)
        else:pore_time(args.output,args.dt)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.collect:
        jobs=[]
        if args.part in ('all','grains'):
            jobs += [(f'junction-{n}',['--worker','junction','--n',str(n)]) for n in (64,96,128)]
            jobs += [(f'grains-{seed}',['--worker','grains','--seed',str(seed)]) for seed in (11,29,47)]
            if args.extended:
                jobs += [(f'junction-{n}-refined',['--worker','junction','--n',str(n),'--max-iterations',str(limit),
                           '--initial-file',str((args.output/f'junction-{n}'/'fields.npz').resolve())])
                         for n,limit in ((96,30000),(128,40000))]
                jobs += [(f'grains-long-{seed}',['--worker','grains','--seed',str(seed),'--end','160']) for seed in (11,29,47)]
        if args.part in ('all','precipitate'):
            jobs += [(f'precipitate-{n}',['--worker','precipitate','--n',str(n)]) for n in (33,65,97,129)]
            jobs += [('precipitate-ellipse',['--worker','precipitate','--n','97','--aspect','1.5'])]
            for repeat in range(1,4):
                for mode in (('fft','condensed') if repeat%2 else ('condensed','fft')):
                    jobs.append((f'timing-{mode}-{repeat}',['--worker','precipitate','--n','97','--mode',mode]))
        if args.part in ('all','pore'):
            jobs += [('pore-capillary',['--worker','pore-capillary'])]
            jobs += [(f'pore-time-{dt}',['--worker','pore-time','--dt',str(dt)]) for dt in (.02,.01,.005,.0025)]
        for name,options in jobs:
            subprocess.run([sys.executable,__file__,*options,'--output',str(args.output/name)],check=True,cwd=ROOT)
    collect(args.output)


if __name__=='__main__':main()
