"""Render Section 39 saved limits, local-field changes and balance evidence."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    start=perf_counter();output=args.input;r=json.loads((output/'report.json').read_text());figures=output/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    diffusion=[x for x in r['runs'] if x['variant']=='diffusion']
    axes[0].loglog([x['n'] for x in diffusion],[x['continuum_mode_error'] for x in diffusion],'o-',label='Continuum mode')
    axes[0].set(xlabel='Cells per side',ylabel='Composition perturbation error',title='Constant-mobility diffusion limit');axes[0].legend(fontsize=8)
    phase=[x for x in r['runs'] if x['variant']=='phase']
    for key,label in (('phase_error','Phase fraction'),('temperature_error','Temperature')):
        axes[1].loglog([x['dt'] for x in phase],[x[key] for x in phase],'o-',label=label)
    axes[1].set(xlabel='Time step',ylabel='Absolute final error',title='Homogeneous phase/heat limit');axes[1].legend(fontsize=8)
    for run in r['runs']:
        if run['n']==16 and run['variant']=='coupled':
            t=[v['time'] for v in run['rows']];E=np.array([v['total_energy'] for v in run['rows']])
            axes[2].plot(t,(E-E[0])/run['energy_scale'],label=f'dt = {run["dt"]}')
    axes[2].set(xlabel='Time',ylabel='Energy drift / initial nonthermal scale',title='Coupled first-law refinement');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    for suffix in ('png','svg'):fig.savefig(figures/f'thermochemical-verification.{suffix}',dpi=170)
    plt.close(fig)
    run=next(x for x in r['runs'] if x['variant']=='coupled' and x['n']==24)
    model,state,_=ThermalChemical2D.load(output/f'{run["name"]}-final.npz')
    _,initial,_=ThermalChemical2D.load(output/f'{run["name"]}-initial.npz');n=run['n']
    fig,grid=plt.subplots(2,3,figsize=(13,8),layout='constrained');axes=grid.ravel()
    data=[(state.c,'Final composition','viridis'),(state.eta,'Final phase fraction','viridis'),(state.T,'Final temperature','coolwarm'),
          (state.c-initial.c,'Composition change','coolwarm'),(state.eta-initial.eta,'Phase change','coolwarm')]
    for ax,(field,title,cmap) in zip(axes,data):
        options={}
        if 'change' in title:
            bound=float(np.max(abs(field)));options=dict(vmin=-bound,vmax=bound)
        im=ax.imshow(field.reshape(n,n).T,origin='lower',extent=(0,1,0,1),cmap=cmap,**options)
        ax.set(xlabel='Reference x',ylabel='Reference y',title=title);fig.colorbar(im,ax=ax,shrink=.8)
    for item in r['runs']:
        if item['n']==16 and item['dt']==.0005 and item['variant'] in ('coupled','frozen-fields','no-partition'):
            entropy=np.array([v['entropy'] for v in item['rows']])
            axes[5].plot([v['time'] for v in item['rows']],entropy-entropy[0],label=item['variant'])
    axes[5].set(xlabel='Time',ylabel='Total entropy increase',title='Physical coupling ablations');axes[5].legend(fontsize=8);axes[5].grid(alpha=.2)
    fig.suptitle('Evolving composition, phase, heat and finite plasticity; fixed non-affine grip, t = 0.2')
    for suffix in ('png','svg'):fig.savefig(figures/f'thermochemical-fields.{suffix}',dpi=170)
    plt.close(fig)
    (output/'plotting.json').write_text(json.dumps({'plotting_seconds':perf_counter()-start,'source':'report.json and matching checkpoints'},indent=2)+'\n')


if __name__=='__main__':main()
