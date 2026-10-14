"""Plot saved heat/thermoplastic evidence; no trajectory reruns."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.thermoplastic import ThermalPlastic2D


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--nonaffine',type=Path,required=True);args=parser.parse_args()
    start=perf_counter();base=json.loads((args.input/'report.json').read_text())
    supplement=json.loads((args.nonaffine/'report.json').read_text());figures=args.input/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    heat=[r for r in base['runs'] if r['variant']=='heat']
    axes[0].loglog([r['n'] for r in heat],[r['continuum_mode_error'] for r in heat],'o-',label='Continuum heat mode')
    axes[0].loglog([r['n'] for r in heat],[r['discrete_time_error'] for r in heat],'s--',label='Discrete mode (time error)')
    axes[0].set(xlabel='Cells per side',ylabel='Relative temperature perturbation error',title='Heat-equation limit');axes[0].legend(fontsize=8)
    for run in [r for r in base['runs'] if r['variant']=='coupled' and r['n']==16]:
        t=np.array([r['time'] for r in run['rows']]);E=np.array([r['total_energy'] for r in run['rows']])
        axes[1].plot(t,(E-E[0])/run['energy_drift_scale'],label=f'dt = {run["dt"]}')
    axes[1].set(xlabel='Time',ylabel='Energy drift / initial elastic energy',title='Closed-system first law');axes[1].legend(fontsize=8)
    for run in base['runs']:
        if run['n']==16 and run['dt']==.001 and run['variant'] in ('coupled','no-slip','no-conduction'):
            s=np.array([r['entropy'] for r in run['rows']]);t=[r['time'] for r in run['rows']]
            axes[2].plot(t,s-s[0],label=run['variant'])
    axes[2].set(xlabel='Time',ylabel='Total entropy increase',title='Conduction and plastic contributions');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    for suffix in ('png','svg'):fig.savefig(figures/f'thermoplastic-verification.{suffix}',dpi=170)
    plt.close(fig)
    run=supplement['runs'][-1];model,state,_=ThermalPlastic2D.load(args.nonaffine/f'{run["name"]}-final.npz');n=run['n']
    fig,grid=plt.subplots(2,2,figsize=(10,8),layout='constrained');axes=grid.ravel()
    im=axes[0].imshow(state.T.reshape(n,n).T,origin='lower',extent=(0,1,0,1),cmap='coolwarm')
    axes[0].set(xlabel='Reference x',ylabel='Reference y',title='Final temperature');fig.colorbar(im,ax=axes[0],shrink=.8)
    slip=np.linalg.norm(state.gamma,axis=-1).mean(axis=1).reshape(n,n).T
    im=axes[1].imshow(slip,origin='lower',extent=(0,1,0,1),cmap='viridis')
    axes[1].set(xlabel='Reference x',ylabel='Reference y',title='Mean element slip magnitude');fig.colorbar(im,ax=axes[1],shrink=.8)
    t=[r['time'] for r in run['rows']]
    axes[2].plot(t,[r['elastic_energy'] for r in run['rows']],label='Elastic')
    axes[2].set(xlabel='Time',ylabel='Energy per reference thickness',title='Elastic-energy release');axes[2].grid(alpha=.2)
    for key,label in (('hardening_energy','Stored hardening'),('kinetic_energy','Kinetic')):
        axes[3].plot(t,[r[key] for r in run['rows']],label=label)
    axes[3].set(xlabel='Time',ylabel='Energy per reference thickness',title='Inertial and hardening storage');axes[3].legend(fontsize=8);axes[3].grid(alpha=.2)
    fig.suptitle('Non-affine fixed grip, n = 16, t = 0.2; evolving heat and multiplicative plasticity')
    for suffix in ('png','svg'):fig.savefig(figures/f'thermoplastic-spatial.{suffix}',dpi=170)
    plt.close(fig)
    (args.input/'plotting.json').write_text(json.dumps({'plotting_seconds':perf_counter()-start,'sources':[str(args.input),str(args.nonaffine)]},indent=2)+'\n')


if __name__=='__main__':main()
