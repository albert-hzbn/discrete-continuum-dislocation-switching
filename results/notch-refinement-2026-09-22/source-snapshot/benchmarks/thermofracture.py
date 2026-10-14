"""Section 42 coupled fracture/thermochemistry, invariant and refinement matrix."""
import argparse
from dataclasses import asdict
import json,os,platform,resource
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
from bmvr.finite_elasticity import FiniteParameters
from bmvr.thermoplastic import ThermalPlasticParameters
from bmvr.thermochemical import ChemicalParameters
from bmvr.thermofracture import ThermalFracture2D,ThermalFractureParameters
from thermochemical import comparison,write_json


def make_model(n,variant):
    homogeneous=variant=='homogeneous'
    model=ThermalFracture2D(n,degree=1 if homogeneous else 2,
        mechanical=FiniteParameters(alpha_c=0,alpha_eta=0,alpha_T=0) if homogeneous else None,
        transport=ThermalPlasticParameters(yield_stress=(100,100)) if homogeneous else None,
        chemistry=ChemicalParameters(mobility=0,phase_mobility=0,partition=0) if homogeneous else None,
        fracture=ThermalFractureParameters(mobility=0) if variant=='frozen-damage' else
            ThermalFractureParameters(phase_resistance=0) if variant=='no-phase-resistance' else None,
        fixed=np.arange(2*(n+1)**2) if homogeneous else None)
    m=model.mechanics.mesh;x,y=m.coordinates.T;u=np.zeros(2*m.node_count)
    u[0::2]=.12*x if homogeneous else .15*y*np.sin(np.pi*x)
    u[1::2]=(.12 if homogeneous else .10)*y;model.values=u[model.fixed].copy()
    state=model.initial(u=u,equilibrate=not homogeneous,
        c=.2 if homogeneous else lambda x,y:.3+.08*np.cos(np.pi*x)*np.cos(np.pi*y),
        eta=.3 if homogeneous else lambda x,y:.5+.3*np.tanh((.22-np.sqrt((x-.5)**2+(y-.5)**2))/.08),
        T=1. if homogeneous else lambda x,y:1+.1*np.cos(np.pi*x)*np.cos(np.pi*y),
        damage=.1 if homogeneous else lambda x,y:.65*np.exp(-((y-.5)/.08)**2)*(1-np.tanh((x-.4)/.08))/2)
    return model,state


def run(n,dt,variant,output,builder=make_model,end_time=None):
    start=perf_counter();model,state=builder(n,variant);setup=perf_counter()-start
    end=(1. if variant=='homogeneous' else .2) if end_time is None else end_time
    steps=round(end/dt);assert abs(steps*dt-end)<1e-13
    name=f'{variant}-{n}-{dt:g}';start=perf_counter();initial_rate=model.rate(state)
    rows=[dict(time=0.,**initial_rate['diagnostics'])];model.save(output/f'{name}-initial.npz',state)
    io=perf_counter()-start;solve=0.;min_increment=0.;entropy_integral=0.
    for step in range(1,steps+1):
        start=perf_counter();previous=state.damage
        try:state,d=model.step(state,dt)
        except (ValueError,RuntimeError) as error:
            model.save(output/f'{name}-last-accepted.npz',state,(step-1)*dt,step-1)
            write_json(output/f'{name}-failure.json',dict(error=str(error),step=step,n=n,dt=dt,variant=variant))
            raise
        solve+=perf_counter()-start;min_increment=min(min_increment,float(np.min(state.damage-previous)))
        entropy_integral+=dt*d['midpoint_entropy_production'];rows.append(dict(time=step*dt,**d))
    start=perf_counter();model.save(output/f'{name}-final.npz',state,end,steps)
    first=rows[0];scale=max(first['elastic_energy']+first['chemical_energy']+first['gradient_energy']+
        2*abs(first['partition_energy'])+first['fracture_energy']+first['damage_binding_energy'],1e-12)
    drift=max(abs(r['total_energy']-first['total_energy']) for r in rows)/scale
    mass=max(abs(r['mass']-first['mass']) for r in rows)
    checks=dict(positive_T=min(r['min_T'] for r in rows)>0,positive_J=min(r['min_J'] for r in rows)>0,
        positive_capacity=min(r['min_capacity'] for r in rows)>0,
        plastic_volume=max(r['plastic_volume_error'] for r in rows)<1e-10,
        entropy_production=min(r['entropy_production'] for r in rows)>=-1e-14,
        entropy_increase=bool(np.all(np.diff([r['entropy'] for r in rows])>=-1e-10)),energy_drift=drift<1e-4,
        conductive_balance=max(abs(r['heat_power_sum']) for r in rows)<1e-12,
        midpoint_converged=max(r['midpoint_residual'] for r in rows[1:])<=1e-12,
        fixed_boundary=bool(np.array_equal(state.u[model.fixed],model.values) and not np.any(state.v[model.fixed])),
        composition_inventory=mass<1e-10,
        composition_bounds=min(r['min_c'] for r in rows)>=0 and max(r['max_c'] for r in rows)<=1,
        phase_bounds=min(r['min_eta'] for r in rows)>=0 and max(r['max_eta'] for r in rows)<=1,
        chemical_energy_transfer=max(abs(r['chemical_flux_sum']) for r in rows)<1e-12,
        microstress_transfer=max(abs(r['microstress_power_sum']) for r in rows)<1e-12,
        damage_bounds=min(r['min_damage'] for r in rows)>=0 and max(r['max_damage'] for r in rows)<=1,
        damage_irreversibility=min_increment>=-1e-13,damage_dissipation=min(r['damage_power'] for r in rows)>=-1e-14)
    result=dict(name=name,n=n,degree=model.mechanics.degree,dt=dt,steps=steps,variant=variant,rows=rows,
        energy_scale=scale,relative_energy_drift=drift,mass_error=mass,min_damage_increment=min_increment,
        entropy_integral=entropy_integral,entropy_integration_error=abs(rows[-1]['entropy']-first['entropy']-entropy_integral),
        checks=checks,initialization_seconds=setup,solver_seconds=solve,diagnostics_output_seconds=io+perf_counter()-start,
        evolving_array_bytes=sum(v.nbytes for v in vars(state).values()),evolving_stored_scalars=sum(v.size for v in vars(state).values()),
        parameters=dict(mechanical=asdict(model.mechanics.p),transport=asdict(model.p),chemistry=asdict(model.chemistry),fracture=asdict(model.fracture)))
    if variant=='homogeneous':
        f=model.fracture;psi=initial_rate['local']['undamaged_elastic'][0,0]
        A=2*(1-f.residual)*psi;B=f.toughness/f.ell+2*f.phase_resistance*.3**2*(3-2*.3)
        target=A/(A+B)+(.1-A/(A+B))*np.exp(-f.mobility*(A+B)*end)
        def density(d):return (f.residual+(1-f.residual)*(1-d)**2)*psi+B*d*d/2
        T=1+(density(.1)-density(target))/model.mechanics.p.cv
        result.update(damage_error=float(np.max(abs(state.damage-target))),temperature_error=float(np.max(abs(state.T-T))))
    write_json(output/f'{name}.json',result)
    print(name,'checks',all(checks.values()),'drift',drift,'damage',rows[-1]['damage_integral'],'solver',solve,flush=True)
    return result


def compare(a,b,output):
    errors=comparison(a,b,output,loader=ThermalFracture2D)
    _,sa,_=ThermalFracture2D.load(output/f'{a["name"]}-final.npz');_,sb,_=ThermalFracture2D.load(output/f'{b["name"]}-final.npz')
    na,nb=a['n'],b['n'];ca=(np.arange(na)+.5)/na;cb=(np.arange(nb)+.5)/nb
    target=sb.damage if na==nb else RegularGridInterpolator((cb,cb),sb.damage.reshape(nb,nb))(np.stack(np.meshgrid(ca,ca,indexing='ij'),axis=-1)).ravel()
    errors['damage']=float(np.linalg.norm(sa.damage-target)/np.linalg.norm(target))
    return errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True);start=perf_counter()
    homogeneous=[run(2,dt,'homogeneous',output) for dt in (.01,.005,.0025)]
    coupled=[run(n,.00025,'coupled',output) for n in (8,16,24)]
    time=[run(16,dt,'coupled',output) for dt in (.001,.0005)]
    ablations=[run(16,.00025,v,output) for v in ('frozen-damage','no-phase-resistance')]
    spatial=compare(coupled[1],coupled[2],output);tc=compare(time[0],coupled[1],output);tf=compare(time[1],coupled[1],output)
    gates={}
    for key in ('damage','temperature'):
        values=[x[key+'_error'] for x in homogeneous]
        gates['homogeneous_'+key+'_decreases']=values[0]>values[1]>values[2]
        gates['homogeneous_'+key+'_finest']=values[2]<1e-6
    for key in spatial:
        gates['mesh_'+key]=spatial[key]<.05;gates['time_'+key]=tf[key]<.01;gates['time_decreases_'+key]=tc[key]>tf[key]
    runs=homogeneous+coupled+time+ablations
    report=dict(scope='Common-energy isotropic finite thermochemical fracture on full grids; no tension split, contact or adaptive efficiency claim.',
        runs=runs,spatial=spatial,time_coarse=tc,time_fine=tf,gates=gates,all_checks=all(all(x['checks'].values()) for x in runs),
        all_gates=all(gates.values()),study_seconds=perf_counter()-start,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024))
    write_json(output/'report.json',report);print('Accuracy gates:',gates,flush=True)


if __name__=='__main__':main()
