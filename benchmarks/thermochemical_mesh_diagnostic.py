"""Locate the retained 16/24 quadrature stress and slip discrepancy."""
import argparse
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from bmvr.thermochemical import ThermalChemical2D
from thermochemical import write_json


def grid(model,n):
    return model.mechanics.mesh.quadrature_points.reshape(n,n,2,2,2).transpose(0,2,1,3,4).reshape(2*n,2*n,2)


def values(a,n):
    tail=a.shape[2:]
    return a.reshape(n,n,2,2,*tail).transpose(0,2,1,3,*range(4,4+len(tail))).reshape(2*n,2*n,*tail)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args();root=args.input;report={}
    for suffix in ('initial','final'):
        a,sa,_=ThermalChemical2D.load(root/f'coupled-16-0.0005-{suffix}.npz')
        b,sb,_=ThermalChemical2D.load(root/f'coupled-24-0.0005-{suffix}.npz')
        xa,xb=grid(a,16),grid(b,24)
        fields={'Piola':(a.rate(sa)['local']['P'],b.rate(sb)['local']['P'])}
        if suffix=='final':fields['slip']=(sa.gamma,sb.gamma)
        for name,(va,vb) in fields.items():
            target=RegularGridInterpolator((xb[:,0,0],xb[0,:,1]),values(vb,24))(xa)
            diff=values(va,16)-target;norms=(diff**2).reshape(32,32,-1).sum(axis=-1)
            report[suffix+'_'+name]=dict(relative_error=float(np.linalg.norm(diff)/np.linalg.norm(target)),
                top_20_percent_fraction_of_squared_error=float(norms[xa[:,:,1]>.8].sum()/norms.sum()))
    write_json(root/'slip-mesh-diagnostic.json',report)


if __name__=='__main__':main()
