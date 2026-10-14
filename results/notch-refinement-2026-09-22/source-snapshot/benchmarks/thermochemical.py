"""Section 39 analytical limits and fully evolving thermochemical studies."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import resource
from time import perf_counter
import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
from bmvr.finite_elasticity import FiniteParameters
from bmvr.thermoplastic import ThermalPlasticParameters
from bmvr.thermochemical import ThermalChemical2D, ChemicalParameters


def write_json(path,value):
    def scalar(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(f'Unsupported JSON value {type(x).__name__}')
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=scalar)+'\n')


def make_model(n,variant,degree=1):
    limit=variant in ('diffusion','phase')
    mechanical=FiniteParameters(alpha_c=0,alpha_eta=0,alpha_T=0) if limit else FiniteParameters()
    transport=ThermalPlasticParameters(yield_stress=(100,100)) if limit else ThermalPlasticParameters()
    chemistry=ChemicalParameters()
    if variant=='diffusion':chemistry=ChemicalParameters(mobility_kind='constant',partition=0,phase_mobility=0)
    if variant=='phase':chemistry=ChemicalParameters(mobility=0,partition=0)
    if variant=='frozen-fields':chemistry=ChemicalParameters(mobility=0,phase_mobility=0)
    if variant=='no-partition':chemistry=ChemicalParameters(partition=0)
    fixed=np.arange(2*(degree*n+1)**2) if limit else None
    model=ThermalChemical2D(n,mechanical=mechanical,transport=transport,chemistry=chemistry,fixed=fixed,degree=degree)
    m=model.mechanics.mesh;u=np.zeros(2*m.node_count)
    if not limit:u[0::2]=.15*m.coordinates[:,1]*np.sin(np.pi*m.coordinates[:,0])
    model.values=u[model.fixed].copy()
    c=lambda x,y:.3+(.05 if variant=='diffusion' else .08)*np.cos(np.pi*x)*np.cos(np.pi*y)
    eta=lambda x,y:.5+.3*np.tanh((.22-np.sqrt((x-.5)**2+(y-.5)**2))/.08)
    if variant=='diffusion':eta=0.
    if variant=='phase':eta=.3;c=.3
    T=1. if limit else lambda x,y:1+.1*np.cos(np.pi*x)*np.cos(np.pi*y)
    state=model.initial(T=T,u=u,c=c,eta=eta,equilibrate=not limit)
    return model,state


def run(n,dt,variant,output,degree=1):
    start=perf_counter();model,state=make_model(n,variant,degree);setup=perf_counter()-start
    end=1. if variant=='phase' else .2;steps=round(end/dt);assert abs(steps*dt-end)<1e-13
    name=('q2-' if degree==2 else '')+f'{variant}-{n}-{dt:g}';initial=state;start=perf_counter()
    rows=[dict(time=0.,**model.rate(state)['diagnostics'])];model.save(output/f'{name}-initial.npz',state)
    io=perf_counter()-start;solver=0.;entropy_integral=0.
    for step in range(1,steps+1):
        start=perf_counter()
        try:state,d=model.step(state,dt)
        except (ValueError,RuntimeError) as error:
            model.save(output/f'{name}-last-accepted.npz',state,(step-1)*dt,step-1)
            write_json(output/f'{name}-failure.json',dict(n=n,dt=dt,variant=variant,step=step,error=str(error)))
            raise
        solver+=perf_counter()-start;entropy_integral+=dt*d['midpoint_entropy_production']
        rows.append(dict(time=step*dt,**d))
    start=perf_counter();model.save(output/f'{name}-final.npz',state,end,steps)
    first=rows[0];energy=np.array([r['total_energy'] for r in rows]);entropy=np.array([r['entropy'] for r in rows])
    scale=max(first['elastic_energy']+first['chemical_energy']-first['partition_energy']+first['gradient_energy']+abs(first['partition_energy']),1e-12)
    drift=float(np.max(abs(energy-energy[0]))/scale)
    mass_error=max(abs(r['mass']-first['mass']) for r in rows)
    checks=dict(positive_T=min(r['min_T'] for r in rows)>0,positive_J=min(r['min_J'] for r in rows)>0,
        positive_capacity=min(r['min_capacity'] for r in rows)>0,
        plastic_volume=max(r['plastic_volume_error'] for r in rows)<1e-10,
        entropy_production=min(r['entropy_production'] for r in rows)>=-1e-14,
        entropy_increase=bool(np.all(np.diff(entropy)>=-1e-10)),energy_drift=drift<1e-4,
        conductive_balance=max(abs(r['heat_power_sum']) for r in rows)<1e-12,
        midpoint_converged=max(r['midpoint_residual'] for r in rows[1:])<=1e-12,
        fixed_boundary=bool(np.array_equal(state.u[model.fixed],model.values) and not np.any(state.v[model.fixed])),
        composition_inventory=mass_error<1e-10,
        composition_bounds=min(r['min_c'] for r in rows)>=0 and max(r['max_c'] for r in rows)<=1,
        phase_bounds=min(r['min_eta'] for r in rows)>=0 and max(r['max_eta'] for r in rows)<=1,
        chemical_energy_transfer=max(abs(r['chemical_flux_sum']) for r in rows)<1e-12,
        microstress_transfer=max(abs(r['microstress_power_sum']) for r in rows)<1e-12)
    result=dict(name=name,n=n,dt=dt,degree=degree,variant=variant,steps=steps,rows=rows,relative_energy_drift=drift,
        energy_scale=scale,mass_error=mass_error,entropy_integral=entropy_integral,
        entropy_integration_error=float(abs(entropy[-1]-entropy[0]-entropy_integral)),checks=checks,
        initialization_seconds=setup,solver_seconds=solver,diagnostics_output_seconds=io+perf_counter()-start,
        evolving_array_bytes=sum(v.nbytes for v in vars(state).values()),evolving_stored_scalars=sum(v.size for v in vars(state).values()),
        parameters=dict(mechanical=asdict(model.mechanics.p),transport=asdict(model.p),chemistry=asdict(model.chemistry)))
    if variant=='diffusion':
        k2=2*np.pi**2;kh2=8*np.sin(np.pi/(2*n))**2/model.mechanics.mesh.h**2
        continuum=.3+(initial.c-.3)*np.exp(-.02*k2*(1+.002*k2)*end)
        discrete=.3+(initial.c-.3)*np.exp(-.02*kh2*(1+.002*kh2)*end)
        result.update(continuum_mode_error=float(np.linalg.norm(state.c-continuum)/np.linalg.norm(continuum-.3)),
            discrete_time_error=float(np.linalg.norm(state.c-discrete)/np.linalg.norm(discrete-.3)))
    if variant=='phase':
        y0=-.4;y=y0/np.sqrt(y0*y0+(1-y0*y0)*np.exp(-end));eta=(1+y)/2
        T=1+(.3**2*.7**2-eta**2*(1-eta)**2)/10
        result.update(phase_error=float(np.max(abs(state.eta-eta))),temperature_error=float(np.max(abs(state.T-T))))
    write_json(output/f'{name}.json',result)
    print(f'{name}: drift={drift:.3g}, mass={mass_error:.3g}, checks={all(checks.values())}, solve={solver:.2f}s',flush=True)
    return result


def comparison(coarse,fine,output,fine_output=None,loader=ThermalChemical2D):
    a,sa,_=loader.load(output/f'{coarse["name"]}-final.npz');b,sb,_=loader.load((fine_output or output)/f'{fine["name"]}-final.npz')
    na,nb=coarse['n'],fine['n'];ca=(np.arange(na)+.5)/na;cb=(np.arange(nb)+.5)/nb
    X=np.stack(np.meshgrid(ca,ca,indexing='ij'),axis=-1);errors={}
    for key in ('c','eta','T'):
        target=getattr(sb,key) if na==nb else RegularGridInterpolator((cb,cb),getattr(sb,key).reshape(nb,nb))(X).ravel()
        errors[key]=float(np.linalg.norm(getattr(sa,key)-target)/np.linalg.norm(target-target.mean()))
    qa=int(np.sqrt(a.mechanics.mesh.quadrature_count));qb=int(np.sqrt(b.mechanics.mesh.quadrature_count))
    def qgrid(model,n,q):return model.mechanics.mesh.quadrature_points.reshape(n,n,q,q,2).transpose(0,2,1,3,4).reshape(q*n,q*n,2)
    xa,xb=qgrid(a,na,qa),qgrid(b,nb,qb);gb=sb.gamma.reshape(nb,nb,qb,qb,2).transpose(0,2,1,3,4).reshape(qb*nb,qb*nb,2)
    if na==nb and qa==qb:
        if not np.allclose(xa,xb,atol=1e-14,rtol=0):raise ValueError('Same-size quadrature grids differ.')
        target=sb.gamma
    else:
        target=RegularGridInterpolator((xb[:,0,0],xb[0,:,1]),gb)(xa)
        target=target.reshape(na,qa,na,qa,2).transpose(0,2,1,3,4).reshape(na*na,qa*qa,2)
    weights=a.mechanics.mesh.weights[None,:,None]
    errors['slip']=float(np.sqrt(np.sum(weights*(sa.gamma-target)**2)/np.sum(weights*target**2)))
    return errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True);start=perf_counter()
    diffusion=[run(n,.0005,'diffusion',output) for n in (8,16,24)]
    phase=[run(4,dt,'phase',output) for dt in (.01,.005,.0025)]
    coupled=[run(n,.0005,'coupled',output) for n in (8,16,24)]
    time_runs=[run(16,dt,'coupled',output) for dt in (.001,.00025)]
    ablations=[run(16,.0005,v,output) for v in ('frozen-fields','no-partition')]
    mesh=comparison(coupled[1],coupled[2],output);time_coarse=comparison(time_runs[0],time_runs[1],output);time_medium=comparison(coupled[1],time_runs[1],output)
    gates=dict(diffusion_decreases=diffusion[0]['continuum_mode_error']>diffusion[1]['continuum_mode_error']>diffusion[2]['continuum_mode_error'],
        diffusion_finest=diffusion[-1]['continuum_mode_error']<.002,
        phase_decreases=phase[0]['phase_error']>phase[1]['phase_error']>phase[2]['phase_error'],
        phase_finest=phase[-1]['phase_error']<1e-6,
        phase_temperature_decreases=phase[0]['temperature_error']>phase[1]['temperature_error']>phase[2]['temperature_error'],
        phase_temperature_finest=phase[-1]['temperature_error']<1e-6,
        energy_drift_decreases=time_runs[0]['relative_energy_drift']>coupled[1]['relative_energy_drift']>time_runs[1]['relative_energy_drift'])
    for key in mesh:
        gates['mesh_'+key]=mesh[key]<.05;gates['time_'+key]=time_medium[key]<.01
        gates['time_decreases_'+key]=time_coarse[key]>time_medium[key]
    runs=diffusion+phase+coupled+time_runs+ablations
    report=dict(scope='Finite thermochemical/plastic dynamics with evolving c, eta and T; fixed/closed boundaries and two plane-strain slip systems.',
        runs=runs,mesh_comparison=mesh,time_comparison_coarse=time_coarse,time_comparison_medium=time_medium,gates=gates,
        all_checks=all(all(r['checks'].values()) for r in runs),all_gates=all(gates.values()),study_seconds=perf_counter()-start,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024))
    write_json(output/'report.json',report);print(json.dumps({'gates':gates,'seconds':report['study_seconds']}),flush=True)


if __name__=='__main__':main()
