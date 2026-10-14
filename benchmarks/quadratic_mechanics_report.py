"""Plot Q2 mechanical accuracy and saved coupled-field checks."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.thermochemical import ThermalChemical2D


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--q1-manufactured',type=Path,required=True);parser.add_argument('--q1-base',type=Path,required=True)
    parser.add_argument('--q1-fine',type=Path,required=True);args=parser.parse_args();start=perf_counter()
    r=json.loads((args.input/'report.json').read_text());old=json.loads((args.q1_manufactured/'report.json').read_text())
    base=json.loads((args.q1_base/'report.json').read_text());fine=json.loads((args.q1_fine/'report.json').read_text())
    figures=args.input/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    for data,label in ((old['manufactured'],'Q1'),(r['manufactured'],'Q2')):
        for key,style in (('displacement','o-'),('gradient','s--')):
            axes[0].loglog([x['unknowns'] for x in data],[x['errors'][key] for x in data],style,label=f'{label} {key}')
    axes[0].set(xlabel='Displacement coordinates',ylabel='Relative error',title='Matched-node manufactured solution');axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    labels=['Q1 16–24','Q1 32–48','Q2 16–24'];errors=[base['mesh_comparison']['slip'],fine['comparison_32_48']['slip'],r['mesh_comparison']['slip']]
    axes[1].bar(labels,100*np.array(errors),color=['#b85c38','#b85c38','#237a75']);axes[1].axhline(5,color='black',linestyle='--',label='Unchanged 5% target')
    axes[1].set(ylabel='Slip discrepancy (%)',title='Coupled spatial refinement');axes[1].legend(fontsize=8)
    for run in r['runs']:
        E=np.array([x['total_energy'] for x in run['rows']]);axes[2].plot([x['time'] for x in run['rows']],(E-E[0])/run['energy_scale'],label=f'Q2 n={run["n"]}')
    axes[2].set(xlabel='Time',ylabel='Scaled total-energy drift',title='Coupled first-law checks');axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
    for suffix in ('png','svg'):fig.savefig(figures/f'quadratic-verification.{suffix}',dpi=170)
    plt.close(fig)
    run=r['runs'][-1];model,s,_=ThermalChemical2D.load(args.input/f'{run["name"]}-final.npz');n=run['n'];rate=model.rate(s);m=model.mechanics.mesh
    slip=np.sum(m.weights*np.linalg.norm(s.gamma,axis=-1),axis=1)/m.h**2
    shear=np.sum(m.weights*rate['local']['sigma'][...,0,1],axis=1)/m.h**2
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,field,title,cmap in zip(axes,(slip,shear,s.eta),('Mean slip magnitude','Mean Cauchy shear','Evolving phase fraction'),('viridis','coolwarm','viridis')):
        im=ax.imshow(field.reshape(n,n).T,origin='lower',extent=(0,1,0,1),cmap=cmap);fig.colorbar(im,ax=ax,shrink=.8)
        ax.set(xlabel='Reference x',ylabel='Reference y',title=title)
    fig.suptitle('Q2 mechanics with unchanged thermochemical physics; n=24, t=0.2')
    for suffix in ('png','svg'):fig.savefig(figures/f'quadratic-fields.{suffix}',dpi=170)
    plt.close(fig)
    (args.input/'plotting.json').write_text(json.dumps({'plotting_seconds':perf_counter()-start},indent=2)+'\n')


if __name__=='__main__':main()
