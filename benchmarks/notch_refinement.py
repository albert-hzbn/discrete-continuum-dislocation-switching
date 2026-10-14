"""Unchanged AT2 n48 extension; run with the preserved source PYTHONPATH."""
import argparse,json,sys,os,platform
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--coarse',type=Path,default=Path('results/finite-damage-split-2026-09-22'))
    parser.add_argument('--snapshot',type=Path,default=Path('results/notch-refinement-2026-09-22/source-snapshot'));args=parser.parse_args()
    # Script-directory imports otherwise precede PYTHONPATH and bypass the snapshot.
    sys.path[:0]=[str((args.snapshot/'src').resolve()),str((args.snapshot/'benchmarks').resolve())]
    import thermofracture
    import bmvr.thermofracture as implementation
    from thermofracture import run
    from finite_damage_split import make_model,geometry
    from thermochemical import comparison,write_json
    from bmvr.thermofracture import ThermalFracture2D
    out=args.output;old=args.coarse;out.mkdir(parents=True,exist_ok=True)
    fine=run(48,.00025,'tensile-split',out,builder=make_model,end_time=.4)
    coarse=json.loads((old/'tensile-split-32-0.00025.json').read_text())
    errors=comparison(coarse,fine,old,out,loader=ThermalFracture2D)
    _,a,_=ThermalFracture2D.load(old/f'{coarse["name"]}-final.npz');_,b,_=ThermalFracture2D.load(out/f'{fine["name"]}-final.npz')
    x=(np.arange(32)+.5)/32;y=(np.arange(48)+.5)/48
    ref=RegularGridInterpolator((y,y),b.damage.reshape(48,48))(np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1)).ravel()
    errors['damage']=float(np.linalg.norm(a.damage-ref)/np.linalg.norm(ref))
    report=dict(run=fine,comparison_32_48=errors,gates={k:v<.05 for k,v in errors.items()},geometry=geometry(fine,out),
        retained_failure=str(old/'report.json'),source_snapshot=str(args.snapshot/'manifest.json'),
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
            model_module=implementation.__file__,benchmark_module=thermofracture.__file__,
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}))
    write_json(out/'report.json',report);print(json.dumps({'errors':errors,'gates':report['gates']}),flush=True)


if __name__=='__main__':main()
