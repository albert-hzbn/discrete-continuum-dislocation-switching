"""Section 40 manufactured and unchanged-physics Q2 thermochemical study."""
import argparse
import json
import os
from pathlib import Path
import platform
from bmvr import _resource as resource
from time import perf_counter
import numpy as np
import scipy
from bmvr.finite_elasticity import FiniteElasticity2D,constitutive
from finite_elasticity import manufactured
from thermochemical import run,comparison,write_json


def manufactured_run(n,output):
    start=perf_counter();model=FiniteElasticity2D(n,degree=2,body=lambda x,y:manufactured(x,y)[2]);m=model.mesh
    x,y=m.coordinates.T;exact=manufactured(x,y)[0];initial=np.column_stack((.2*x,-.1*y)).ravel();fixed=model.boundary()
    setup=perf_counter()-start;start=perf_counter();u,state,status=model.equilibrate(initial,fixed,exact.ravel()[fixed]);solve=perf_counter()-start
    start=perf_counter();x,y=m.quadrature_points.transpose(2,0,1);exact,F,_=manufactured(x,y)
    uq=np.einsum('ga,eai->egi',m.N,u.reshape(-1,2)[m.cells]);perturbation=exact-np.stack((.2*x,-.1*y),axis=-1)
    def norm(a):return np.sqrt(np.sum(m.weights.reshape((1,len(m.weights))+(1,)*(a.ndim-2))*a*a))
    P=constitutive(F,tangent=False)['P']
    errors=dict(displacement=float(norm(uq-exact)/norm(perturbation)),gradient=float(norm(state['F']-F)/norm(F-np.eye(3))),Piola=float(norm(state['local']['P']-P)/norm(P)))
    model.save(output/f'q2-manufactured-{n}.npz',u)
    r=dict(n=n,degree=2,errors=errors,status=status,min_J_bound=state['min_J_bound'],unknowns=len(u),
        initialization_seconds=setup,solver_seconds=solve,diagnostics_output_seconds=perf_counter()-start,
        checks=dict(converged=status['relative_residual']<=1e-9,orientation=state['min_J_bound']>0,
            energy_search=bool(np.all(np.diff(status['potential'])<=1e-12))))
    write_json(output/f'q2-manufactured-{n}.json',r);print(f'Q2 manufactured {n}: {errors}',flush=True)
    return r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--q1-base',type=Path,required=True);parser.add_argument('--q1-fine',type=Path,required=True);args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True);start=perf_counter()
    exact=[manufactured_run(n,output) for n in (4,8,16,32)]
    runs=[run(n,.00025,'coupled',output,degree=2) for n in (12,16,24)]
    mesh=comparison(runs[1],runs[2],output)
    base=json.loads((args.q1_base/'report.json').read_text());fine=json.loads((args.q1_fine/'report.json').read_text())
    q1_24=next(r for r in base['runs'] if r['n']==24 and r['variant']=='coupled')
    q1_48=next(r for r in fine['runs'] if r['n']==48)
    cross24=comparison(q1_24,runs[2],args.q1_base,output)
    cross48=comparison(runs[2],q1_48,output,args.q1_fine)
    gates=dict(manufactured_displacement=exact[-1]['errors']['displacement']<.01,
        manufactured_gradient=exact[-1]['errors']['gradient']<.02,
        decreasing_displacement=all(a['errors']['displacement']>b['errors']['displacement'] for a,b in zip(exact,exact[1:])),
        decreasing_gradient=all(a['errors']['gradient']>b['errors']['gradient'] for a,b in zip(exact,exact[1:])))
    gates.update({'coupled_'+key:value<.05 for key,value in mesh.items()})
    report=dict(scope='Q2 displacement and consistent nine-point quadrature; unchanged finite thermochemical physics and finite-volume cell fields.',
        manufactured=exact,runs=runs,mesh_comparison=mesh,cross_q1_24_to_q2_24=cross24,cross_q2_24_to_q1_48=cross48,
        gates=gates,all_checks=all(all(r['checks'].values()) for r in exact+runs),all_gates=all(gates.values()),study_seconds=perf_counter()-start,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024))
    write_json(output/'report.json',report);print({'gates':gates,'errors':mesh,'seconds':report['study_seconds']},flush=True)


if __name__=='__main__':main()
