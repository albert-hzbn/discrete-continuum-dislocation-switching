"""Section 43: finite split and held-tension notch evolution with free sides."""
import argparse,json,os,platform
from bmvr import _resource as resource
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from bmvr.thermofracture import ThermalFracture2D,ThermalFractureParameters
from thermochemical import write_json
from thermofracture import run,compare


def make_model(n,variant,fracture=None,grip=.12,model_class=ThermalFracture2D):
    model=model_class(n,degree=2,fracture=fracture or ThermalFractureParameters(elastic_split='volumetric-deviatoric'))
    m=model.mechanics.mesh;x,y=m.coordinates.T
    bottom=np.flatnonzero(y==0);top=np.flatnonzero(y==1)
    pin=bottom[np.argmin(abs(x[bottom]-.5))]
    model.fixed=np.r_[2*bottom+1,2*top+1,2*pin]
    u=np.zeros(2*m.node_count);u[0::2]=-.25*grip*(x-.5);u[1::2]=grip*y
    model.values=u[model.fixed].copy()
    state=model.initial(u=u,equilibrate=True,
        c=lambda x,y:.3+.08*np.cos(np.pi*x)*np.cos(np.pi*y),
        eta=lambda x,y:.5+.3*np.tanh((.22-np.sqrt((x-.5)**2+(y-.5)**2))/.08),
        T=lambda x,y:1+.1*np.cos(np.pi*x)*np.cos(np.pi*y),
        damage=lambda x,y:.65*np.exp(-((y-.5)/.08)**2)*(1-np.tanh((x-.4)/.08))/2)
    return model,state


def front(damage,n,threshold=.4):
    """Right crossing of the left-connected centerline superlevel interval."""
    axis=(np.arange(n)+.5)/n
    values=np.array([np.interp(.5,axis,row) for row in damage.reshape(n,n)])
    if values[0]<threshold:return None
    below=np.flatnonzero(values<threshold)
    if not len(below):return 1.
    j=int(below[0]);left,right=values[j-1],values[j]
    return float(axis[j-1]+(threshold-left)/(right-left)*(axis[j]-axis[j-1]))


def geometry(run,output):
    model,a,_=ThermalFracture2D.load(output/f'{run["name"]}-initial.npz')
    _,b,_=ThermalFracture2D.load(output/f'{run["name"]}-final.npz')
    n=run['n'];m=model.mechanics.mesh;x,y=m.coordinates[m.cells].mean(axis=1).T;delta=b.damage-a.damage
    total=float(delta.sum());first,last=front(a.damage,n),front(b.damage,n)
    r=model.rate(b)
    return dict(initial_front=first,final_front=last,advance=None if first is None or last is None else last-first,
        top_fraction=None if total<=0 else float(delta[y>.85].sum()/total),
        notch_fraction=None if total<=0 else float(delta[(abs(y-.5)<.16)&(x<.75)].sum()/total),
        final_elastic_compression_fraction=float(np.mean(r['local']['Je']<1)),
        boundary_description=f'Bottom/top y fixed at 0/{np.max(model.values[model.fixed%2==1]):g}; one bottom x pin; remaining tractions zero.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True);start=perf_counter()
    spatial=[run(n,.00025,'tensile-split',output,builder=make_model,end_time=.4) for n in (16,24,32)]
    temporal=[run(24,dt,'tensile-split',output,builder=make_model,end_time=.4) for dt in (.001,.0005)]
    mesh=compare(spatial[1],spatial[2],output);tc=compare(temporal[0],spatial[1],output);tf=compare(temporal[1],spatial[1],output)
    geo={str(x['n']):geometry(x,output) for x in spatial};gates={}
    for key in mesh:
        gates['mesh_'+key]=mesh[key]<.05;gates['time_'+key]=tf[key]<.01;gates['time_decreases_'+key]=tc[key]>tf[key]
    g=geo['32'];geometry_gates=dict(top_fraction=g['top_fraction'] is not None and g['top_fraction']<.2,
        notch_fraction=g['notch_fraction'] is not None and g['notch_fraction']>.5,
        front_advance=g['advance'] is not None and g['advance']>.02,
        front_mesh=geo['24']['final_front'] is not None and g['final_front'] is not None and abs(geo['24']['final_front']-g['final_front'])<.04)
    report=dict(scope='Finite volumetric/deviatoric damage with free lateral tractions and held tensile grips; no contact or adaptive efficiency claim.',
        runs=spatial+temporal,spatial=mesh,time_coarse=tc,time_fine=tf,gates=gates,geometry=geo,geometry_gates=geometry_gates,
        all_checks=all(all(x['checks'].values()) for x in spatial+temporal),all_accuracy_gates=all(gates.values()),
        all_geometry_gates=all(geometry_gates.values()),study_seconds=perf_counter()-start,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024))
    write_json(output/'report.json',report);print('Accuracy:',gates,'Geometry:',geometry_gates,flush=True)


if __name__=='__main__':main()
