"""Section 71.5 diagnostic (post hoc): where does the successive-mesh slip discrepancy live, inside or outside the crack band?"""
import argparse,json
from pathlib import Path
import numpy as np
from graded_notch import load,cellwise_slip
from bmvr.nodal_damage import NodalThermalFracture2D
from bmvr.finite_elasticity import FiniteElasticity2D


def split(a_path,b_path,half=.02):
    ma,sa=load(a_path,NodalThermalFracture2D);mb,sb=load(b_path,NodalThermalFracture2D);A,B=ma.mechanics.mesh,mb.mechanics.mesh
    target=cellwise_slip(A,B,sb.gamma);w=A.cell_weights;cy=A.coordinates[A.cells].mean(axis=1)[:,1]
    err=w[...,None]*(sa.gamma-target)**2;ref=w[...,None]*target**2;inside=abs(cy-.5)<half
    total=float(np.sqrt(err.sum()/ref.sum()))
    return dict(total=total,outside=float(np.sqrt(err[~inside].sum()/ref[~inside].sum())),inside=float(np.sqrt(err[inside].sum()/ref[inside].sum())),
        share_of_squared_error_inside=float(err[inside].sum()/err.sum()),share_of_reference_inside=float(ref[inside].sum()/ref.sum()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--prefix',required=True);args=parser.parse_args()
    FiniteElasticity2D.subdivide=6;out={}
    for a,b in ((64,96),(96,128)):
        pa,pb=(args.input/f'{args.prefix}-{n}-0.00025-final.npz' for n in (a,b))
        if pa.exists() and pb.exists():out[f'{a}_{b}']=split(pa,pb)
    (args.input/'slip-band-split.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(out,indent=1))


if __name__=='__main__':main()
