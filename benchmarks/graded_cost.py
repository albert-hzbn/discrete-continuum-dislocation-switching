"""Section 48: graded versus uniform notch cost at equal accuracy against a common uniform n128 reference."""
import argparse,json,os,platform,subprocess,sys
from pathlib import Path
import numpy as np
import scipy
from bmvr.nodal_damage import NodalThermalFracture2D
from thermofracture import run
from thermochemical import write_json
import finite_damage_split
from graded_notch import make_model as graded_model,compare,label,events

CHECKPOINTS=(.1,.15)
END=.2
CANDIDATES=[('graded',32),('graded',48),('graded',64),('uniform',48),('uniform',64),('uniform',96)]
REFERENCE=('uniform',128)
TARGETS=(.1,.05,.025)
FIELDS=('c','eta','T','slip','damage')


def uniform_model(n,variant):
    def model_class(n_,**kwargs):return NodalThermalFracture2D(n_,preconditioned=True,**kwargs)
    return finite_damage_split.make_model(n,variant,model_class=model_class)


BUILDERS=dict(graded=graded_model,uniform=uniform_model)


def name(family,n,repeat=0):return f'cost-{family}{"-r"+str(repeat) if repeat else ""}'


def trajectory(family,n,repeat,out):
    run(n,.00025,name(family,n,repeat),out,builder=BUILDERS[family],end_time=END,checkpoints=CHECKPOINTS)


def assemble(out):
    ref=json.loads((out/f'{name(*REFERENCE)}-{REFERENCE[1]}-0.00025.json').read_text())
    path=lambda r,t:out/(f'{r["name"]}-final.npz' if t==END else f'{r["name"]}-{label(t)}.npz')
    rows={}
    for family,n in CANDIDATES:
        r=json.loads((out/f'{name(family,n)}-{n}-0.00025.json').read_text())
        timing=json.loads((out/f'time-{family}-{n}-0.json').read_text())
        errors={f't{t:g}':compare(path(r,t),path(ref,t)) for t in CHECKPOINTS+(END,)}
        repeats=[json.loads(p.read_text()) for p in sorted(out.glob(f'time-{family}-{n}-[1-9].json'))]
        rows[f'{family}-{n}']=dict(family=family,n=n,cells=n*n,errors=errors,solver_seconds=r['solver_seconds'],
            initialization_seconds=r['initialization_seconds'],output_seconds=r['diagnostics_output_seconds'],
            wall_seconds=timing['wall_seconds'],peak_rss_bytes=timing['peak_rss_bytes'],checks=r['checks'],
            max_midpoint_iterations=events(r)['max_midpoint_iterations'],
            repeat_solver_seconds=[x['solver_seconds'] for x in repeats],repeat_wall_seconds=[x['wall_seconds'] for x in repeats])
    reference_error=compare(path(json.loads((out/f'{name("uniform",96)}-96-0.00025.json').read_text()),END),path(ref,END))
    cheapest={}
    for target in TARGETS:
        cheapest[str(target)]={}
        for family in ('graded','uniform'):
            passing=[k for k,v in rows.items() if v['family']==family and all(v['errors'][f't{END:g}'][f]<target for f in FIELDS)]
            cheapest[str(target)][family]=min(passing,key=lambda k:rows[k]['wall_seconds']) if passing else None
    ratios={}
    for target,pick in cheapest.items():
        g,u=pick['graded'],pick['uniform']
        if g and u:ratios[target]=dict(kind='ratio',wall=rows[u]['wall_seconds']/rows[g]['wall_seconds'],solver=rows[u]['solver_seconds']/rows[g]['solver_seconds'])
        elif g:ratios[target]=dict(kind='lower bound against uniform n96 (no uniform candidate passes)',
            wall=rows['uniform-96']['wall_seconds']/rows[g]['wall_seconds'],solver=rows['uniform-96']['solver_seconds']/rows[g]['solver_seconds'])
        else:ratios[target]=dict(kind='no graded candidate passes')
    claims={}
    for target,pick in cheapest.items():
        g=pick['graded']
        if g is None or 'wall' not in ratios[target]:claims[target]=dict(claimed=False,reason='no graded candidate passes');continue
        walls=[rows[g]['wall_seconds']]+rows[g]['repeat_wall_seconds'];spread=(max(walls)-min(walls))/min(walls)
        claims[target]=dict(claimed=ratios[target]['wall']>1+spread,spread=spread,ratio=ratios[target]['wall'],repeats=len(walls))
    interpretation=' '.join(f"At {100*float(t):g}%: ratio {v['ratio']:.2f} versus repeat spread {100*v['spread']:.0f}% over {v['repeats']} processes; {'claimed' if v['claimed'] else 'not claimed'}."
        if 'ratio' in v else f"At {100*float(t):g}%: {v['reason']}." for t,v in claims.items())
    report=dict(scope='Section 48 equal-accuracy cost of static graded versus uniform notch meshes at t<=0.2, Gauss rule; conventional technique.',
        claims=claims,interpretation=interpretation,
        reference=dict(name=ref['name'],solver_seconds=ref['solver_seconds'],checks=ref['checks'],uniform96_vs_reference=reference_error),
        candidates=rows,cheapest=cheapest,ratios=ratios,targets=TARGETS,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            concurrency='Six candidates ran concurrently (one per performance core); the reference ran alongside, and the graded n48/n64 repeats ran later under different load.'))
    write_json(out/'report.json',report);print(json.dumps(dict(cheapest=cheapest,ratios=ratios)),flush=True)


def timed(family,n,repeat,out):
    """Fresh child process for one trajectory; records launch-to-exit wall time and the child's peak RSS."""
    from bmvr import _resource as resource
    from time import perf_counter
    start=perf_counter()
    subprocess.run([sys.executable,__file__,'--output',str(out),'--run',family,str(n),str(repeat)],check=True)
    wall=perf_counter()-start;rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    record=json.loads((out/f'{name(family,n,repeat)}-{n}-0.00025.json').read_text())
    write_json(out/f'time-{family}-{n}-{repeat}.json',dict(wall_seconds=wall,peak_rss_bytes=rss,solver_seconds=record['solver_seconds']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',nargs=3,metavar=('FAMILY','N','REPEAT'))
    parser.add_argument('--timed',nargs=3,metavar=('FAMILY','N','REPEAT'));args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.run:trajectory(args.run[0],int(args.run[1]),int(args.run[2]),out)
    elif args.timed:timed(args.timed[0],int(args.timed[1]),int(args.timed[2]),out)
    else:assemble(out)


if __name__=='__main__':main()
