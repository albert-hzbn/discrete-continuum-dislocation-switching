"""Fresh-process worker for exact evaluator parity and solver costs."""
from time import perf_counter
START=perf_counter()
import argparse
import json
import os
from pathlib import Path
import platform
from bmvr import _resource as resource
import numpy as np
import scipy
import bmvr.thermochemical as implementation
from thermochemical import run,write_json
IMPORT_SECONDS=perf_counter()-START


def metadata():
    return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),machine=platform.machine(),cpu_count=os.cpu_count(),
        load_average=os.getloadavg(),implementation_file=str(Path(implementation.__file__).resolve()),
        threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('micro','trajectory'),required=True)
    parser.add_argument('--state',type=Path);parser.add_argument('--n',type=int);parser.add_argument('--degree',type=int)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);environment=metadata()
    if args.mode=='micro':
        model,state,_=implementation.ThermalChemical2D.load(args.state)
        for _ in range(5):model.rate(state)
        batches=[]
        for _ in range(5):
            start=perf_counter()
            for _ in range(30):rate=model.rate(state)
            batches.append(perf_counter()-start)
        arrays={f'rate_{key}':rate[key] for key in ('u','v','T','c','eta','gamma','Lp','PT','chi')}
        arrays.update({f'local_{key}':rate['local'][key] for key in ('energy','P','M','sigma','gc','geta','entropy','heat_capacity')})
        for _ in range(3):state,_=model.step(state,.00025)
        arrays.update({f'state_{key}':value for key,value in vars(state).items()})
        np.savez_compressed(args.output/'parity.npz',**arrays)
        report=dict(mode='micro',warmup_calls=5,batch_calls=30,batch_seconds=batches,
                    batch_median_seconds=float(np.median(batches)),environment=environment)
    else:
        start=perf_counter();result=run(args.n,.00025,'coupled',args.output,degree=args.degree)
        elapsed=perf_counter()-start
        report=dict(mode='trajectory',result=result,environment=environment,
            import_seconds=IMPORT_SECONDS,run_seconds=elapsed,
            worker_other_seconds=elapsed-result['initialization_seconds']-result['solver_seconds']-result['diagnostics_output_seconds'])
    report['worker_elapsed_seconds']=perf_counter()-START
    report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    write_json(args.output/'worker.json',report)


if __name__=='__main__':main()
