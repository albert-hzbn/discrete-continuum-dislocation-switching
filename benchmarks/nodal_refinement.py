"""Section 45.5 nodal n48 extension at unchanged active-notch physics."""
import argparse,json
from pathlib import Path
import numpy as np
from thermofracture import run
from thermochemical import write_json
from nodal_damage import make_model,compare,geometry
from bmvr.nodal_damage import NodalThermalFracture2D


def slip_measures(run_record,out):
    m,s,_=NodalThermalFracture2D.load(out/f'{run_record["name"]}-final.npz');w=m.mechanics.mesh.weights[None,:,None]
    return dict(L1=float(np.sum(w*abs(s.gamma))),L2=float(np.sqrt(np.sum(w*s.gamma**2))),max=float(abs(s.gamma).max()),
        active_fraction=float(np.mean(np.any(s.gamma!=0,axis=-1))),ell_over_h=float(m.fracture.ell/m.mechanics.mesh.h))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base',type=Path,default=Path('results/nodal-damage-constrained-2026-09-26'));args=parser.parse_args()
    out,base=args.output,args.base;out.mkdir(parents=True,exist_ok=True)
    fine=run(48,.00025,'nodal-tensile',out,builder=make_model,end_time=.4)
    coarse=json.loads((base/'nodal-tensile-32-0.00025.json').read_text())
    for suffix in ('initial','final'):
        target=out/f'{coarse["name"]}-{suffix}.npz'
        if not target.exists():target.symlink_to((base/f'{coarse["name"]}-{suffix}.npz').resolve())
    errors=compare(coarse,fine,out)
    measures={str(n):slip_measures(json.loads((base/f'nodal-tensile-{n}-0.00025.json').read_text()),base) for n in (16,24,32)}
    measures['48']=slip_measures(fine,out)
    report=dict(run=fine,comparison_32_48=errors,gates={k:v<.05 for k,v in errors.items()},geometry=geometry(fine,out),
        slip_measures=measures,base=str(base),scope='Uniform extension of the Section 45 nodal refinement; retained failures are not superseded.')
    write_json(out/'report.json',report);print(json.dumps(dict(errors=errors,gates=report['gates'],slip=measures)),flush=True)


if __name__=='__main__':main()
