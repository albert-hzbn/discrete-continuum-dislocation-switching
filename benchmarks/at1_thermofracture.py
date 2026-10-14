"""Section 44 finite threshold, fixed-parameter loading and refinement."""
import argparse,json,os,platform
from bmvr import _resource as resource
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from scipy.optimize import brentq
from bmvr.finite_elasticity import FiniteParameters,constitutive
from bmvr.thermoplastic import ThermalPlasticParameters
from bmvr.thermochemical import ChemicalParameters
from bmvr.thermofracture import ThermalFracture2D,ThermalFractureParameters
from thermochemical import write_json
from thermofracture import run,compare
from finite_damage_split import make_model as notch_model,geometry


def make_model(n,variant):
    f=ThermalFractureParameters(crack_density='AT1',elastic_split='volumetric-deviatoric')
    if variant.startswith('at1-load-'):
        return notch_model(n,variant,fracture=f,grip=float(variant.removeprefix('at1-load-')))
    ratio=float(variant.removeprefix('threshold-'));p=FiniteParameters(alpha_c=0,alpha_eta=0,alpha_T=0)
    target=ratio*3*f.toughness/(16*f.ell*(1-f.residual))
    stretch=brentq(lambda a:constitutive(np.diag([a,a,1.]),parameters=p)['elastic']-target,1,2)
    model=ThermalFracture2D(n,mechanical=p,fracture=f,transport=ThermalPlasticParameters(yield_stress=(100,100)),
        chemistry=ChemicalParameters(mobility=0,phase_mobility=0,partition=0),fixed=np.arange(2*(n+1)**2))
    u=((stretch-1)*model.mechanics.mesh.coordinates).ravel();model.values=u.copy()
    return model,model.initial(u=u,c=.2,eta=.3,T=1,damage=0)


def threshold_errors(result,out):
    model,s,_=ThermalFracture2D.load(out/f'{result["name"]}-final.npz')
    _,initial,_=ThermalFracture2D.load(out/f'{result["name"]}-initial.npz')
    psi=float(model.rate(initial)['local']['damage_elastic'][0,0]);f=model.fracture
    A=2*(1-f.residual)*psi;C=3*f.toughness/(8*f.ell);B=2*f.phase_resistance*.3**2*(3-2*.3)
    target=max(A-C,0)/(A+B)*(1-np.exp(-f.mobility*(A+B)))
    def energy(d):return (f.residual+(1-f.residual)*(1-d)**2)*psi+C*d+B*d*d/2
    T=1+(energy(0)-energy(target))/model.mechanics.p.cv
    return dict(damage_error=float(np.max(abs(s.damage-target))),temperature_error=float(np.max(abs(s.T-T))),
        analytical_final_damage=target,analytical_final_temperature=T,actual_energy_ratio=psi/(C/(2*(1-f.residual))))


def safe_compare(a,b,out):
    with np.errstate(invalid='ignore',divide='ignore'):errors=compare(a,b,out)
    zero=[]
    for key,value in errors.items():
        if np.isfinite(value):continue
        _,sa,_=ThermalFracture2D.load(out/f'{a["name"]}-final.npz');_,sb,_=ThermalFracture2D.load(out/f'{b["name"]}-final.npz')
        field='gamma' if key=='slip' else key
        av,bv=getattr(sa,field),getattr(sb,field)
        if np.count_nonzero(av)==0 and np.count_nonzero(bv)==0:errors[key]=0.;zero.append(key)
        else:errors[key]=None
    return dict(errors=errors,exact_zero_fields=zero)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);start=perf_counter();homogeneous=[]
    for ratio in (.8,4.):
        for dt in (.01,.005,.0025):
            result=run(2,dt,f'threshold-{ratio:g}',out,builder=make_model,end_time=1.)
            result.update(threshold_errors(result,out));write_json(out/f'{result["name"]}.json',result);homogeneous.append(result)
    loading=[run(16,.00025,f'at1-load-{load:g}',out,builder=make_model,end_time=.4) for load in (0.,.04,.06,.12)]
    finer=[run(n,.00025,'at1-load-0.06',out,builder=make_model,end_time=.4) for n in (24,32)]
    times=[run(16,dt,'at1-load-0.06',out,builder=make_model,end_time=.4) for dt in (.001,.0005)]
    mesh=safe_compare(finer[0],finer[1],out);tc=safe_compare(times[0],loading[2],out);tf=safe_compare(times[1],loading[2],out)
    gates={}
    for run_ in homogeneous:
        ratio=float(run_['variant'].removeprefix('threshold-'))
        if ratio<1 or run_['dt']==.0025:
            for field in ('damage','temperature'):
                gates[run_['name']+'_'+field]=run_[field+'_error']<(1e-12 if ratio<1 else 1e-6)
    for key,value in mesh['errors'].items():
        gates['mesh_'+key]=value is not None and value<.05
        gates['time_'+key]=tf['errors'][key] is not None and tf['errors'][key]<.01
    geo={x['name']:geometry(x,out) for x in loading+finer}
    g=geo[finer[-1]['name']];g24=geo[finer[0]['name']]
    geometry_gates=dict(top_fraction=g['top_fraction'] is not None and g['top_fraction']<.2,
        notch_fraction=g['notch_fraction'] is not None and g['notch_fraction']>.5,
        front_advance=g['advance'] is not None and g['advance']>.02,
        front_mesh=g24['final_front'] is not None and g['final_front'] is not None and abs(g24['final_front']-g['final_front'])<.04)
    activation={}
    for run_ in loading:
        model,s,_=ThermalFracture2D.load(out/f'{run_["name"]}-initial.npz');rate=model.rate(s);low=s.damage<1e-4
        activation[run_['name']]=dict(low_damage_area_fraction=float(low.mean()),active_fraction=float(np.mean(rate['damage'][low]>1e-12)),
            mean_rate=float(np.mean(rate['damage'][low])),max_rate=float(np.max(rate['damage'][low])))
    runs=homogeneous+loading+finer+times
    report=dict(scope='AT1 finite-initiation law with fixed physical coefficients; separate from retained AT2 outcomes and costs.',
        runs=runs,gates=gates,spatial=mesh,time_coarse=tc,time_fine=tf,geometry=geo,geometry_gates=geometry_gates,activation=activation,
        all_checks=all(all(x['checks'].values()) for x in runs),all_gates=all(gates.values()),study_seconds=perf_counter()-start,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024))
    write_json(out/'report.json',report);print('Accuracy:',gates,'Geometry:',geometry_gates,flush=True)


if __name__=='__main__':main()
