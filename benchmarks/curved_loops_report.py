"""Render curved-loop conservation/convergence and coupled-field evidence."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def render(folder):
    start=perf_counter()
    output=folder/'figures'
    output.mkdir(exist_ok=True)
    report=json.loads((folder/'convergence/report.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

    def save(fig,name):
        for extension in ('png','svg'):
            fig.savefig(output/f'{name}.{extension}',dpi=170,bbox_inches='tight')
        plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    rows=report['field_grid'][:-1]
    for key,label in [('relative_energy_error','Energy'),('relative_force_error','Nodal force')]:
        axes[0].semilogy([r['n'] for r in rows],[r[key] for r in rows],'o-',label=label)
    axes[0].set(xlabel='Grid cells per axis',ylabel='Relative error',title='Fixed polygon and core; reference n = 49')
    axes[0].legend(fontsize=8)
    rows=report['polygon'][:-1]
    axes[1].loglog([r['nodes'] for r in rows],[r['relative_energy_error'] for r in rows],'o-',label='Energy; reference 96 nodes')
    axes[1].loglog([r['nodes'] for r in rows],[r['relative_area_error'] for r in rows],'s--',label='Area; exact circle')
    axes[1].set(xlabel='Polygon nodes',ylabel='Relative error',title='Geometry refinement; fixed n = 25')
    axes[1].xaxis.set_major_locator(FixedLocator([8,16,32,64]))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[1].legend(fontsize=8)
    rows=report['quadrature'][:-1]
    axes[2].semilogy([r['order'] for r in rows],[r['relative_force_error'] for r in rows],'o-')
    axes[2].set(xlabel='Gauss points per segment',ylabel='Relative force error',title='Long-segment quadrature; reference 16')
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Gaussian-core loop verification: field, geometry and quadrature errors are separate')
    save(fig,'loop-convergence')

    with np.load(folder/'coupled-moving/fields.npz') as a,np.load(folder/'coupled-pinned/fields.npz') as b:
        fig,axes=plt.subplots(2,3,figsize=(12,7.5),layout='constrained')
        initial,final=a['nodes'][0],a['nodes'][-1]
        ix=np.r_[np.arange(len(initial)),0]
        axes[0,0].plot(initial[ix,0],initial[ix,1],'k--',label='Initial')
        axes[0,0].plot(final[ix,0],final[ix,1],'o-',ms=3,label='Final (actual scale)')
        axes[0,0].set(xlabel='x',ylabel='y',title='Loop in its fixed glide plane',aspect='equal')
        axes[0,0].legend(fontsize=8)
        displacement=np.linalg.norm(a['nodes']-initial,axis=2)
        axes[0,1].plot(a['time'],displacement.max(axis=1),label='Largest nodal displacement')
        axes[0,1].plot(a['time'],displacement.mean(axis=1),'--',label='Mean nodal displacement')
        axes[0,1].set(xlabel='Time',ylabel='Displacement',title='Resolved coordinate evolution')
        axes[0,1].legend(fontsize=8)
        axes[0,2].plot(a['time'],a['energy']-b['energy'])
        axes[0,2].set(xlabel='Time',ylabel='Moving − pinned energy',title='Matched coordinate ablation')
        n=a['phase'].shape[0]; mid=n//2
        fields=[a['phase'][:,:,mid]-b['phase'][:,:,mid],
                a['population'][:,:,mid]-b['population'][:,:,mid],a['stress'][0,2,:,:,mid]-b['stress'][0,2,:,:,mid]]
        for ax,value,title in zip(axes[1],fields,['Phase feedback','Population feedback','Shear-stress feedback, σxz']):
            lim=max(abs(value.min()),abs(value.max()))
            im=ax.imshow(value.T,origin='lower',extent=(-8,8,-8,8),cmap='coolwarm',vmin=-lim,vmax=lim)
            ax.set(xlabel='x',ylabel='y',title=title+'; z = 0')
            fig.colorbar(im,ax=ax,shrink=.8)
        fig.suptitle('Moving-loop feedback in a 3D phase/population/crystal-slip calculation')
        save(fig,'coupled-loop-feedback')
    (output/'plotting.json').write_text(json.dumps({'plotting_and_figure_io_s':perf_counter()-start},indent=2))
    print('Saved two PNG/SVG pairs in',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT/'results/curved-loops-fcc-2026-09-12')
    render(parser.parse_args().input.resolve())
