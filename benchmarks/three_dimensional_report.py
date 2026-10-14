"""Generate scientific 3D field slices and convergence/cost plots from saved arrays."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def render(folder):
    start = perf_counter()
    output = folder/'figures'
    output.mkdir(exist_ok=True)
    report = json.loads((folder/'space-33/report.json').read_text())
    convergence = json.loads((folder/'convergence.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        for suffix in ('png', 'svg'):
            fig.savefig(output/f'{name}.{suffix}', dpi=170, bbox_inches='tight')
        plt.close(fig)

    with np.load(folder/'space-33/fields.npz') as data:
        mid = report['grid']['n']//2
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout='constrained')
        values = [data['phase'][:,:,mid], data['population'][:,mid,:]-.2,
                  np.sqrt(np.mean(data['slip'][:,:,:,mid]**2,axis=0)), data['stress'][0,1,:,:,mid]]
        labels = [('Phase v; z = 0','x','y','coolwarm'), ('Population excess; y = 0','x','z','viridis'),
                  ('RMS over 12 slip systems; z = 0','x','y','magma'), ('Shear stress σxy; z = 0','x','y','coolwarm')]
        for ax, field, (title,xlabel,ylabel,cmap) in zip([axes[0,0],axes[0,1],axes[0,2],axes[1,0]],values,labels):
            im = ax.imshow(field.T,origin='lower',extent=(-8,8,-8,8),cmap=cmap,interpolation='nearest')
            ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
            fig.colorbar(im,ax=ax,shrink=.8)
        for i,label in enumerate(data['component_names']):
            axes[1,1].plot(data['time'],data['components'][:,i],label=str(label))
        axes[1,1].set(xlabel='Time',ylabel='Energy',title='Energy components')
        axes[1,1].legend(fontsize=8,ncol=2)
        with np.load(folder/'space-25/fields.npz') as coupled:
            axes[1,2].plot(coupled['time'],coupled['energy'],label='Coupled crystal slip')
        with np.load(folder/'frozen-slip/fields.npz') as frozen:
            axes[1,2].plot(frozen['time'],frozen['energy'],'--',label='Frozen slip')
        axes[1,2].set(xlabel='Time',ylabel='Energy',title='Fixed-load relaxation; both n = 25')
        axes[1,2].legend(fontsize=8)
        fig.suptitle('3D diffuse inclusion, population transport and local FCC slip (n = 33)')
        save(fig,'coupled-fields')

    fig, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    colors = ['#3f66b0','#218568','#aa7930','#8b579f']
    for key,color in zip(('phase','population','slip','stress'),colors):
        for ax,group,xfield in [(axes[0],'space','h'),(axes[1],'time','dt'),(axes[2],'space','solver_s')]:
            rows=convergence[group]
            ax.loglog([r[xfield] for r in rows],[r['errors'][key]['relative_l2'] for r in rows],
                       'o-',color=color,label=key)
    axes[0].set(xlabel='Cell spacing h',ylabel='Relative L2 error',title='Spatial study; reference n = 49')
    axes[1].set(xlabel='Timestep',ylabel='Relative L2 error',title='Temporal study; reference dt = 0.0025')
    axes[2].set(xlabel='Solver time (s)',ylabel='Relative L2 error',title='Uniform 3D accuracy versus cost')
    for ax,ticks in zip(axes,[(.48,.64,.94),(.005,.01,.02,.04),(.3,.6,1.4)]):
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_: f'{value:g}'))
        ax.xaxis.set_minor_locator(NullLocator())
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2,which='both')
    fig.suptitle('Fixed physical width; scalar errors normalized by phase fraction and population excess')
    save(fig,'convergence-cost')
    (output/'plotting.json').write_text(json.dumps({'plotting_and_figure_io_s':perf_counter()-start},indent=2))
    print('Saved two PNG/SVG pairs:', output)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT/'results/three-dimensional-2026-09-12')
    render(parser.parse_args().input.resolve())
