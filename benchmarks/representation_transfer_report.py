"""Render independent field evolution, chart switches and refinement evidence."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import argparse,json
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'results/representation-transfer-2026-09-13');args=p.parse_args()
    started=perf_counter();s=json.loads((args.input/'summary.json').read_text())
    out=args.input/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for key,label in [('space-13','Full field'),('switch','Line / field / line')]:
        with np.load(args.input/key/'fields.npz') as a:
            axes[0].plot(a['time'],a['energy']-a['energy'][0],label=label)
    axes[0].axvspan(.1,.3,color='#ccc',alpha=.3)
    axes[0].set(xlabel='Time',ylabel='Energy change',title='Shared energy through transfer');axes[0].legend(fontsize=8)
    for ax,label in zip(axes[1:],('space','time')):
        rows=s[label]
        x=[s['reports'][r['run']]['grid']['h'] if label=='space' else s['reports'][r['run']]['dt'] for r in rows]
        for key in ('alpha','stress'):
            ax.loglog(x,[r['errors'][key]['relative_l2'] for r in rows],'o-',label=key)
        ax.set(xlabel='h' if label=='space' else 'dt',ylabel='Relative L2',title=f'{label.capitalize()} refinement');ax.legend(fontsize=8)
        ax.set_xticks(sorted(x),[f'{value:.3g}' for value in sorted(x)])
        ax.xaxis.set_minor_formatter(NullFormatter())
    for ext in ('png','svg'):fig.savefig(out/f'transfer-convergence.{ext}',dpi=160)
    plt.close(fig)
    with np.load(args.input/'space-25'/'fields.npz') as a:
        initial=np.sqrt(np.sum(a['initial_alpha']**2,axis=(0,1)))[:,:,12]
        final=np.sqrt(np.sum(a['alpha']**2,axis=(0,1)))[:,:,12]
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    maximum=max(initial.max(),final.max())
    for ax,data,title in zip(axes,(initial,final,final-initial),('Initial density norm','After independent field motion','Change in density norm')):
        lim=max(abs(data.min()),abs(data.max())) if title.startswith('Change') else maximum
        im=ax.imshow(data.T,origin='lower',extent=(-6,6,-6,6),cmap='RdBu_r' if title.startswith('Change') else 'viridis',
                     vmin=-lim if title.startswith('Change') else 0,vmax=lim)
        ax.set(title=title,xlabel='x',ylabel='y');fig.colorbar(im,ax=ax,shrink=.8)
    for ext in ('png','svg'):fig.savefig(out/f'density-evolution.{ext}',dpi=160)
    plt.close(fig)
    (out/'plotting.json').write_text(json.dumps({'plotting_and_figure_io_s':perf_counter()-started},indent=2)+'\n')
    print('Rendered two PNG/SVG figure pairs.')


if __name__=='__main__':main()
