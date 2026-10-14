"""Section 45 continuous-damage refinement at unchanged active-notch physics."""
import argparse,json,os,platform
from bmvr import _resource as resource
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
from bmvr.nodal_damage import NodalThermalFracture2D
from thermofracture import run
from thermochemical import comparison,write_json
from finite_damage_split import make_model as notch_model


def make_model(n,variant):return notch_model(n,variant,model_class=NodalThermalFracture2D)


def compare(a,b,out):
    errors=comparison(a,b,out,loader=NodalThermalFracture2D)
    ma,sa,_=NodalThermalFracture2D.load(out/f'{a["name"]}-final.npz')
    mb,sb,_=NodalThermalFracture2D.load(out/f'{b["name"]}-final.npz')
    n=b['n'];axis=np.linspace(0,1,n+1);points=ma.mechanics.mesh.quadrature_points
    target=RegularGridInterpolator((axis,axis),sb.damage.reshape(n+1,n+1))(points)
    actual=ma.damage_quadrature(sa.damage);w=ma.mechanics.mesh.weights
    errors['damage']=float(np.sqrt(np.sum(w*(actual-target)**2)/np.sum(w*target**2)))
    return errors


def front(d,n,threshold=.4):
    axis=np.linspace(0,1,n+1);values=np.array([np.interp(.5,axis,row) for row in d.reshape(n+1,n+1)])
    if values[0]<threshold:return None
    below=np.flatnonzero(values<threshold)
    if not len(below):return 1.
    j=int(below[0]);return float(axis[j-1]+(threshold-values[j-1])/(values[j]-values[j-1])*(axis[j]-axis[j-1]))


def geometry(run,out):
    m,a,_=NodalThermalFracture2D.load(out/f'{run["name"]}-initial.npz')
    _,b,_=NodalThermalFracture2D.load(out/f'{run["name"]}-final.npz');mesh=m.mechanics.mesh
    delta=np.sum(mesh.weights*(m.damage_quadrature(b.damage)-m.damage_quadrature(a.damage)),axis=1)
    x,y=mesh.coordinates[mesh.cells].mean(axis=1).T;first,last=front(a.damage,run['n']),front(b.damage,run['n'])
    return dict(initial_front=first,final_front=last,advance=None if first is None or last is None else last-first,
        top_fraction=None if delta.sum()<=0 else float(delta[y>.85].sum()/delta.sum()),
        notch_fraction=None if delta.sum()<=0 else float(delta[(abs(y-.5)<.16)&(x<.75)].sum()/delta.sum()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);start=perf_counter()
    spatial=[run(n,.00025,'nodal-tensile',out,builder=make_model,end_time=.4) for n in (16,24,32)]
    temporal=[run(24,dt,'nodal-tensile',out,builder=make_model,end_time=.4) for dt in (.001,.0005)]
    mesh=compare(spatial[1],spatial[2],out);tc=compare(temporal[0],spatial[1],out);tf=compare(temporal[1],spatial[1],out)
    gates={}
    for key in mesh:
        gates['mesh_'+key]=mesh[key]<.05;gates['time_'+key]=tf[key]<.01;gates['time_decreases_'+key]=tc[key]>tf[key]
    geo={str(x['n']):geometry(x,out) for x in spatial};g=geo['32'];g24=geo['24']
    gg=dict(top_fraction=g['top_fraction'] is not None and g['top_fraction']<.2,
        notch_fraction=g['notch_fraction'] is not None and g['notch_fraction']>.5,
        front_advance=g['advance'] is not None and g['advance']>.02,
        front_mesh=g24['final_front'] is not None and g['final_front'] is not None and abs(g24['final_front']-g['final_front'])<.04)
    runs=spatial+temporal
    report=dict(scope='Continuous Q1 nodal damage in the original AT2 active-notch continuum model; full-grid reference, no adaptive efficiency claim.',
        runs=runs,spatial=mesh,time_coarse=tc,time_fine=tf,gates=gates,geometry=geo,geometry_gates=gg,
        all_checks=all(all(x['checks'].values()) for x in runs),all_accuracy_gates=all(gates.values()),all_geometry_gates=all(gg.values()),
        study_seconds=perf_counter()-start,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024))
    write_json(out/'report.json',report);print('Accuracy:',gates,'Geometry:',gg,flush=True)


if __name__=='__main__':main()
