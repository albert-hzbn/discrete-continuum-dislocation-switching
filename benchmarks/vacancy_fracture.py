"""Section 50 benchmarks: sink relaxation, generation back-stress and the graded notch with reactive vacancies."""
import argparse,json,os,platform
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from bmvr.finite_elasticity import FiniteParameters
from bmvr.thermoplastic import ThermalPlasticParameters
from bmvr.thermochemical import ChemicalParameters
from bmvr.thermofracture import ThermalFractureParameters
from bmvr.nodal_damage import NodalThermalFracture2D
from bmvr.vacancy_fracture import NodalVacancyFracture2D,VacancyParameters
from thermofracture import run
from thermochemical import write_json
import finite_damage_split
from graded_notch import edges,compare,events,slip_measures,label

ACTIVE=VacancyParameters(theta=.02,equilibrium=.05,binding=.003,sink=.5,damage_sink=5.,mobility=.25,generation=4.)
CHECKPOINTS=(.1,.15)
END=.2
FIELDS=('c','eta','T','vac')


def homogeneous(vacancy,shear=0.,plastic=False):
    model=NodalVacancyFracture2D(2,degree=2,mechanical=FiniteParameters(alpha_c=0,alpha_eta=0,alpha_T=0),
        transport=ThermalPlasticParameters() if plastic else ThermalPlasticParameters(yield_stress=(100,100)),
        chemistry=ChemicalParameters(mobility=0,phase_mobility=0,partition=0),fracture=ThermalFractureParameters(mobility=0),
        fixed=np.arange(2*25),vacancy=vacancy)
    m=model.mechanics.mesh;u=np.zeros(2*m.node_count);u[0::2]=shear*m.coordinates[:,1];model.values=u.copy()
    return model,model.initial(u=u,c=.3,eta=0.,damage=0.)


def integrate(model,state,dt,end):
    first=model.rate(state)['diagnostics'];steps=round(end/dt);rows=[first]
    for _ in range(steps):state,d=model.step(state,dt);rows.append(d)
    return state,rows


def sink_relaxation():
    p=VacancyParameters(theta=.02,equilibrium=.05,sink=.5);x0=.08;end=2.
    exact=solve_ivp(lambda t,x:-p.sink*(p.formation+p.theta*np.log(x/(1-x))),(0,end),[x0],rtol=1e-13,atol=1e-15).y[0,-1]
    out={}
    for dt in (.04,.02,.01):
        model,s=homogeneous(p);s.vac[:]=x0;s,rows=integrate(model,s,dt,end)
        drift=max(abs(r['total_energy']-rows[0]['total_energy']) for r in rows)
        out[f'{dt:g}']=dict(error=float(np.max(abs(s.vac-exact))),energy_drift=drift,heat_released=rows[-1]['vacancy_energy']-rows[0]['vacancy_energy'])
    e=[out[k]['error'] for k in ('0.04','0.02','0.01')]
    return dict(exact=float(exact),x0=x0,runs=out,observed_orders=[float(np.log2(e[0]/e[1])),float(np.log2(e[1]/e[2]))],
        linearized_rate=p.sink*p.theta/(p.equilibrium*(1-p.equilibrium)))


def generation_backstress():
    out={}
    for name,chi in (('control',0.),('generation',4.)):
        p=VacancyParameters(theta=.02,equilibrium=.05,sink=.5,generation=chi)
        model,s=homogeneous(p,shear=.1,plastic=True);s,rows=integrate(model,s,.005,1.)
        drift=max(abs(r['total_energy']-rows[0]['total_energy']) for r in rows)
        out[name]=dict(final_slip=float(np.abs(s.gamma).max()),final_vac=float(s.vac.max()),
            final_mu=float(rows[-1]['max_vacancy_mu']),stored_power_last=rows[-1]['slip_stored_power'],
            plastic_power_last=rows[-1]['plastic_power'],energy_drift=drift,min_entropy_production=min(r['entropy_production'] for r in rows))
    out['slip_reduction']=1-out['generation']['final_slip']/out['control']['final_slip']
    return out


def notch_model(n,variant):
    def model_class(n_,**kwargs):return NodalVacancyFracture2D(n_,edges=edges(n_),preconditioned=True,vacancy=ACTIVE,**kwargs)
    return finite_damage_split.make_model(n,variant,model_class=model_class)


def notch_run(n,out):
    run(n,.00025,'vacancy-notch',out,builder=notch_model,end_time=END,checkpoints=CHECKPOINTS)


def slip_L1(loader,path):
    m,s,_=loader.load(path);return float(np.sum(m.mechanics.mesh.cell_weights[...,None]*abs(s.gamma)))


def state_path(out,n,t):return out/(f'vacancy-notch-{n}-0.00025-final.npz' if t==END else f'vacancy-notch-{n}-0.00025-{label(t)}.npz')


def notch_assemble(out,control):
    runs={n:json.loads((out/f'vacancy-notch-{n}-0.00025.json').read_text()) for n in (32,48,64)}
    pairs={}
    for a,b in ((32,48),(48,64)):
        pairs[f'{a}_{b}']={f't{t:g}':compare(state_path(out,a,t),state_path(out,b,t),NodalVacancyFracture2D,FIELDS) for t in CHECKPOINTS+(END,)}
    gates={k:v<.05 for k,v in pairs['48_64'][f't{END:g}'].items()}
    effects={}
    for n in (32,48,64):
        mv,sv,_=NodalVacancyFracture2D.load(state_path(out,n,END));_,sc,_=NodalThermalFracture2D.load(control/f'graded-notch-{n}-0.00025-t{END:g}.npz')
        w=mv.mechanics.mesh.cell_weights;dv,dc=mv.damage_quadrature(sv.damage),mv.damage_quadrature(sc.damage)
        band=dc.mean(axis=1)>.5;x=sv.vac/ACTIVE.equilibrium
        crun=json.loads((control/f'graded-notch-{n}-0.00025.json').read_text())
        rows=runs[n]['rows'];t=np.array([r['time'] for r in rows])
        generated=float(np.trapezoid([r['vacancy_generation_rate'] for r in rows],t));absorbed=float(np.trapezoid([r['vacancy_sink_rate'] for r in rows],t))
        effects[str(n)]=dict(damage_relative_change=float(np.sqrt(np.sum(w*(dv-dc)**2)/np.sum(w*dc**2))),
            slip_L1_change=slip_L1(NodalVacancyFracture2D,state_path(out,n,END))/slip_L1(NodalThermalFracture2D,control/f'graded-notch-{n}-0.00025-t{END:g}.npz')-1,
            max_supersaturation=float(x.max()-1),band_mean_enrichment=float(x[band].mean()-1) if band.any() else None,
            outside_band_mean=float(x[~band].mean()-1),tip_event_vacancy=events(runs[n])['tip_max_damage_09'],
            tip_event_control=events(crun)['tip_max_damage_09'],generated=generated,absorbed=absorbed,
            inventory_change=rows[-1]['vacancy_inventory']-rows[0]['vacancy_inventory'],ledger_residual=rows[-1]['vacancy_inventory']-rows[0]['vacancy_inventory']-generated-absorbed,
            plastic_power_ratio=rows[-1]['plastic_power']/max(crun['rows'][len(rows)-1]['plastic_power'],1e-300),
            min_vacancy_mu=min(r['min_vacancy_mu'] for r in rows),checks=runs[n]['checks'])
    return dict(runs={str(n):{k:v for k,v in r.items() if k!='rows'} for n,r in runs.items()},pairs=pairs,gates_48_64=gates,effects=effects,
        control=str(control),parameters=ACTIVE.__dict__)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--control',type=Path,default=Path('results/graded-notch-v3-2026-09-26'))
    parser.add_argument('--notch',type=int);parser.add_argument('--homogeneous',action='store_true');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);start=perf_counter()
    if args.notch:notch_run(args.notch,out);return
    if args.homogeneous:
        write_json(out/'homogeneous.json',dict(sink_relaxation=sink_relaxation(),generation_backstress=generation_backstress()));return
    report=dict(scope='Section 50 reactive vacancies: synthetic nondimensional coefficients; no material calibration.',
        homogeneous=json.loads((out/'homogeneous.json').read_text()),notch=notch_assemble(out,args.control),study_seconds=perf_counter()-start,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}))
    write_json(out/'report.json',report);print(json.dumps(dict(gates=report['notch']['gates_48_64'],effects=report['notch']['effects']),default=str)[:3000],flush=True)


if __name__=='__main__':main()
