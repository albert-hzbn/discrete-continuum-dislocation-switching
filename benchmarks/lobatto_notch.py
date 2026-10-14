"""Section 47.4 graded notch with corner-sampling Gauss-Lobatto quadrature, run to t=0.4."""
import argparse,json,os,platform
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from bmvr.nodal_damage import NodalThermalFracture2D
from thermofracture import run
from thermochemical import write_json
import finite_damage_split
from graded_notch import BANDS,edges,compare,events,slip_measures,geometry,label,tip_estimate,front

CHECKPOINTS=(.1,.15,.2,.25,.3,.35)
MESHES=(48,64,96,128)
VARIANT='lobatto-notch'


MODE={'mode':True}


def make_model(n,variant):
    def model_class(n_,**kwargs):return NodalThermalFracture2D(n_,edges=edges(n_),preconditioned=MODE['mode'],quadrature='lobatto',**kwargs)
    return finite_damage_split.make_model(n,variant,model_class=model_class)


def reached(record):
    return [t for t in CHECKPOINTS if record['failure'] is None or t<=record['failure']['last_accepted_time']+1e-12]


def pair(coarse,fine,where):
    times=[t for t in CHECKPOINTS if t in reached(coarse) and t in reached(fine)]
    a,b=where[coarse['n']],where[fine['n']]
    rows={label(t):compare(a/f'{coarse["name"]}-{label(t)}.npz',b/f'{fine["name"]}-{label(t)}.npz') for t in times}
    if coarse['failure'] is None and fine['failure'] is None:
        rows['final']=compare(a/f'{coarse["name"]}-final.npz',b/f'{fine["name"]}-final.npz')
    return rows


def front_history(record,folder):
    # Midline damage fronts: process zone (d=0.4) and broken material (d=0.9), at every saved state.
    names=['initial']+[label(t) for t in reached(record)]+(['final'] if record['failure'] is None else [])
    times=[0.]+list(reached(record))+([.4] if record['failure'] is None else [])
    rows={}
    for t,name in zip(times,names):
        m,s,_=NodalThermalFracture2D.load(folder/f'{record["name"]}-{name}.npz')
        rows[f'{t:g}']=dict(process=front(m,s.damage,.4),broken=front(m,s.damage,.9))
    return rows


def assemble(out,gauss,start,first=None):
    # Section 47.5: meshes rerun with the full preconditioner live in out; the rest come from the first GLL folder.
    where={n:out if (out/f'{VARIANT}-{n}-0.00025.json').exists() else first for n in MESHES}
    r={n:json.loads((where[n]/f'{VARIANT}-{n}-0.00025.json').read_text()) for n in MESHES}
    c={f'{a}_{b}':pair(r[a],r[b],where) for a,b in zip(MESHES,MESHES[1:])}
    gates=({k:v<.05 for k,v in c['64_96']['final'].items()} if 'final' in c['64_96'] else 'unevaluated: a trajectory did not reach t=0.4')
    qualification={}
    for key,errors in c['96_128'].items():
        if key in c['64_96']:qualification[key]=dict(below_5_percent=all(v<.05 for v in errors.values()),
            decreasing_slip_damage=all(errors[f]<c['64_96'][key][f] for f in ('slip','damage')))
    # Quadrature sensitivity against the Section 46 Gauss n96 trajectory (diagnostic).
    g96=json.loads((gauss/'graded-notch-96-0.00025.json').read_text())
    sensitivity={}
    for t in (.1,.15,.2,.25):
        a,b=where[96]/f'{VARIANT}-96-0.00025-{label(t)}.npz',gauss/f'graded-notch-96-0.00025-{label(t)}.npz'
        if a.exists() and b.exists():sensitivity[label(t)]=compare(a,b)
    per_mesh={}
    for n,x in r.items():
        hx,hy=(np.diff(e) for e in edges(n))
        item=dict(cells=n*n,fine_ell_over_h=[.08/hx.min(),.08/hy.min()],events=events(x),complete=x['failure'] is None,failure=x['failure'],
            reached=reached(x),solver_seconds=x['solver_seconds'],checks=x['checks'],relative_energy_drift=x['relative_energy_drift'])
        item.update(folder=str(where[n]),preconditioned=MODE['mode'] if where[n]==out else True,fronts=front_history(x,where[n]))
        if x['failure'] is None:item.update(slip_final=slip_measures(where[n]/f'{x["name"]}-final.npz'),geometry_final=geometry(x,where[n]))
        per_mesh[str(n)]=item
    report=dict(scope='Section 47 corner-sampling (Gauss-Lobatto) quadrature on the Section 46.5 graded notch family; same physics. Not dynamic adaptivity or a speedup claim.',
        bands=BANDS,checkpoints=CHECKPOINTS,comparisons=c,gates_64_96_final=gates,qualification_96_128=qualification,
        quadrature_sensitivity_vs_gauss_96=sensitivity,gauss_n64_96_reference_discrepancy='see results/graded-notch-v3-2026-09-26/report.json qualification.comparison_64_96',
        per_mesh=per_mesh,all_checks=all(all(x['checks'].values()) for x in r.values()),tip_timing_estimate=tip_estimate([r[n] for n in MESHES[-3:]]),
        study_seconds=perf_counter()-start,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            concurrency='GLL trajectories ran concurrently with other single-thread studies; solver seconds are not a cost comparison.'))
    write_json(out/'report.json',report);print('gates',gates,'qualification',qualification,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gauss',type=Path,default=Path('results/graded-notch-v3-2026-09-26'))
    parser.add_argument('--first',type=Path,default=Path('results/lobatto-notch-2026-09-26'))
    parser.add_argument('--run',type=int);parser.add_argument('--full',action='store_true');parser.add_argument('--adaptive',action='store_true')
    args=parser.parse_args()
    if args.full:MODE['mode']='full'
    if args.adaptive:MODE['mode']='adaptive'
    out=args.output;out.mkdir(parents=True,exist_ok=True);start=perf_counter()
    if args.run:run(args.run,.00025,VARIANT,out,builder=make_model,end_time=.4,checkpoints=CHECKPOINTS,on_failure='record')
    else:assemble(out,args.gauss,start,args.first)


if __name__=='__main__':main()
