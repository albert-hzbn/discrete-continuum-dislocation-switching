"""Section 38 heat limits and coupled inertial thermoplastic refinement."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
from bmvr import _resource as resource
from time import perf_counter
import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
from bmvr.finite_elasticity import FiniteParameters
from bmvr.thermoplastic import ThermalPlastic2D, ThermalPlasticParameters


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def make_model(n,variant):
    mechanical=FiniteParameters(alpha_T=0) if variant in ('heat','no-expansion') else FiniteParameters()
    transport=ThermalPlasticParameters(yield_stress=(100,100)) if variant in ('heat','no-slip') else ThermalPlasticParameters()
    if variant=='no-conduction':transport=ThermalPlasticParameters(conductivity=0)
    eta=0. if variant=='heat' else lambda x,y:.3*np.exp(-((x-.5)**2+(y-.5)**2)/(2*.15**2))
    fixed=np.arange(2*(n+1)**2) if variant=='heat' else None
    model=ThermalPlastic2D(n,mechanical=mechanical,transport=transport,eta=eta,fixed=fixed)
    m=model.mechanics.mesh;u=np.zeros(2*m.node_count)
    if variant!='heat':u[0::2]=.15*m.coordinates[:,1]
    if variant=='nonaffine':u[0::2]*=np.sin(np.pi*m.coordinates[:,0])
    model.values=u[model.fixed].copy()
    state=model.initial(T=lambda x,y:1+.1*np.cos(np.pi*x)*np.cos(np.pi*y),u=u,equilibrate=variant!='heat')
    return model,state


def run(n,dt,variant,output):
    start=perf_counter();model,state=make_model(n,variant);setup=perf_counter()-start
    final_time=.5 if variant=='heat' else .2;steps=round(final_time/dt)
    assert abs(steps*dt-final_time)<1e-13
    name=f'{variant}-{n}-{dt:g}';start=perf_counter()
    model.save(output/f'{name}-initial.npz',state)
    rows=[dict(time=0.,**model.rate(state)['diagnostics'])];initial=state
    io=perf_counter()-start;solver=0.;entropy_integral=plastic_heat=0.
    for step in range(1,steps+1):
        start=perf_counter()
        try:state,diagnostics=model.step(state,dt)
        except (ValueError,RuntimeError) as error:
            model.save(output/f'{name}-last-accepted.npz',state,(step-1)*dt,step-1)
            write_json(output/f'{name}-failure.json',{'step':step,'error':str(error),'dt':dt,'n':n,'variant':variant})
            raise
        solver+=perf_counter()-start
        entropy_integral+=dt*diagnostics['midpoint_entropy_production'];plastic_heat+=dt*diagnostics['midpoint_plastic_power']
        rows.append(dict(time=step*dt,**diagnostics))
    start=perf_counter();model.save(output/f'{name}-final.npz',state,steps*dt,steps)
    e=np.array([r['total_energy'] for r in rows]);s=np.array([r['entropy'] for r in rows])
    scale=max(rows[0]['elastic_energy'],1e-12) if variant!='heat' else model.mechanics.p.cv*.1
    drift=float(np.max(abs(e-e[0]))/scale)
    checks=dict(positive_T=min(r['min_T'] for r in rows)>0,positive_J=min(r['min_J'] for r in rows)>0,
        positive_capacity=min(r['min_capacity'] for r in rows)>0,
        plastic_volume=max(r['plastic_volume_error'] for r in rows)<1e-10,
        positive_entropy_production=min(r['entropy_production'] for r in rows)>=-1e-14,
        nondecreasing_entropy=bool(np.all(np.diff(s)>=-1e-10)),energy_drift=drift<1e-4,
        conservative_conduction=max(abs(r['heat_power_sum']) for r in rows)<1e-12,
        midpoint_converged=max(r['midpoint_residual'] for r in rows[1:])<=1e-12,
        fixed_boundary=bool(np.array_equal(state.u[model.fixed],model.values) and not np.any(state.v[model.fixed])))
    result=dict(name=name,n=n,dt=dt,steps=steps,variant=variant,rows=rows,
        relative_energy_drift=drift,energy_drift_scale=scale,
        entropy_integral=entropy_integral,entropy_integration_error=float(abs(s[-1]-s[0]-entropy_integral)),
        plastic_dissipation_integral=plastic_heat,checks=checks,
        parameters=dict(mechanical=asdict(model.mechanics.p),transport=asdict(model.p)),
        initialization_seconds=setup,solver_seconds=solver,diagnostics_output_seconds=io+perf_counter()-start,
        evolving_array_bytes=sum(v.nbytes for v in vars(state).values()),
        evolving_stored_scalars=sum(v.size for v in vars(state).values()))
    if variant=='heat':
        rate=2*.4*np.pi**2/10;rate_h=8*.4/(10*model.mechanics.mesh.h**2)*np.sin(np.pi/(2*n))**2
        continuum=1+(initial.T-1)*np.exp(-rate*final_time)
        discrete=1+(initial.T-1)*np.exp(-rate_h*final_time)
        result['continuum_mode_error']=float(np.linalg.norm(state.T-continuum)/np.linalg.norm(continuum-1))
        result['discrete_time_error']=float(np.linalg.norm(state.T-discrete)/np.linalg.norm(discrete-1))
    write_json(output/f'{name}.json',result)
    print(f'{name}: drift={drift:.3g}, checks={all(checks.values())}, solve={solver:.2f}s',flush=True)
    return result


def comparison(coarse,fine,output):
    a,sa,_=ThermalPlastic2D.load(output/f'{coarse["name"]}-final.npz')
    b,sb,_=ThermalPlastic2D.load(output/f'{fine["name"]}-final.npz')
    na,nb=coarse['n'],fine['n']
    centers_a=(np.arange(na)+.5)/na;centers_b=(np.arange(nb)+.5)/nb
    X=np.stack(np.meshgrid(centers_a,centers_a,indexing='ij'),axis=-1)
    target_T=RegularGridInterpolator((centers_b,centers_b),sb.T.reshape(nb,nb))(X).ravel()
    def qgrid(model,n):
        points=model.mechanics.mesh.quadrature_points.reshape(n,n,2,2,2).transpose(0,2,1,3,4).reshape(2*n,2*n,2)
        return points
    xa=qgrid(a,na);xb=qgrid(b,nb)
    g=sb.gamma.reshape(nb,nb,2,2,2).transpose(0,2,1,3,4).reshape(2*nb,2*nb,2)
    target_gamma=RegularGridInterpolator((xb[:,0,0],xb[0,:,1]),g)(xa)
    gamma_a=sa.gamma.reshape(na,na,2,2,2).transpose(0,2,1,3,4).reshape(2*na,2*na,2)
    return dict(temperature=float(np.linalg.norm(sa.T-target_T)/np.linalg.norm(target_T-target_T.mean())),
                slip=float(np.linalg.norm(gamma_a-target_gamma)/np.linalg.norm(target_gamma)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True);start=perf_counter()
    heat=[run(n,.002,'heat',output) for n in (8,16,32)]
    baseline=[run(n,.001,'coupled',output) for n in (8,16,24)]
    time_runs=[run(16,dt,'coupled',output) for dt in (.002,.0005)]
    ablations=[run(16,.001,v,output) for v in ('no-expansion','no-slip','no-conduction')]
    mesh=comparison(baseline[1],baseline[2],output)
    time_coarse=comparison(time_runs[0],time_runs[1],output);time_medium=comparison(baseline[1],time_runs[1],output)
    gates=dict(heat_decreases=heat[0]['continuum_mode_error']>heat[1]['continuum_mode_error']>heat[2]['continuum_mode_error'],
        heat_finest=heat[2]['continuum_mode_error']<.002,mesh_temperature=mesh['temperature']<.05,mesh_slip=mesh['slip']<.05,
        time_temperature=time_medium['temperature']<.01,time_slip=time_medium['slip']<.01,
        temperature_time_error_decreases=time_coarse['temperature']>time_medium['temperature'],
        slip_time_error_decreases=time_coarse['slip']>time_medium['slip'],
        energy_drift_decreases=time_runs[0]['relative_energy_drift']>baseline[1]['relative_energy_drift']>time_runs[1]['relative_energy_drift'])
    runs=heat+baseline+time_runs+ablations
    result=dict(scope='Inertial finite thermoplasticity with evolving temperature and two plane-strain slip systems; prescribed chemistry/phase, fixed displacement, insulated boundaries.',
        runs=runs,mesh_comparison=mesh,time_comparison_coarse=time_coarse,time_comparison_medium=time_medium,
        gates=gates,all_checks=all(all(r['checks'].values()) for r in runs),all_gates=all(gates.values()),
        study_seconds=perf_counter()-start,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            platform=platform.platform(),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024))
    write_json(output/'report.json',result);print(json.dumps({'gates':gates,'seconds':result['study_seconds']}),flush=True)


if __name__=='__main__':main()
