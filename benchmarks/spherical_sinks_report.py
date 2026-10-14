"""Render the saved spherical-sink convergence and population-exchange evidence."""

import argparse,json
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def render(folder):
    start=perf_counter();output=folder/'figures';output.mkdir(exist_ok=True)
    report=json.loads((folder/'report.json').read_text())
    diagnostic=json.loads((folder/'strong_capture_refinement.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#3868a8','#b17e2f','#27816c']
    def save(fig,name):
        for extension in ('png','svg'):fig.savefig(output/f'{name}.{extension}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    with np.load(folder/'fields.npz') as data:
        for k,color in zip((.1,1.,10.),colors):
            axes[0,0].plot(data['r_256'],data[f'exact_{k}_256'],color=color,label=f'kₛ = {k}')
            axes[0,0].scatter(data['r_256'][::8],data[f'c_{k}_256'][::8],color=color,s=10)
            rows=[r for r in report['steady'] if r['reactivity']==k]
            axes[0,1].loglog([r['h'] for r in rows],[r['profile_relative_l2'] for r in rows],'o-',color=color,label=f'kₛ = {k}')
            rows=[r for r in report['width_limit'] if r['reactivity']==k]
            axes[1,1].plot([r['width'] for r in rows],[100*r['relative_difference'] for r in rows],'o-',color=color,label=f'kₛ = {k}')
    axes[0,0].axvspan(1.5,2.5,color='gray',alpha=.15,label='Fixed reaction layer')
    axes[0,0].set(xlabel='Radius',ylabel='Concentration',title='Collocation curves; finite-volume markers, N = 256')
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(xlabel='Grid spacing; fixed width = 1',ylabel='Profile relative L₂ error',title='Spatial refinement against an independent ODE')
    axes[0,1].legend(fontsize=8)
    rows=diagnostic['rows']
    for key,label in (('signed_capture_relative_error','Default midpoint reaction'),
                      ('cell_integrated_reaction_capture_relative_error','Integrated reaction weights')):
        axes[1,0].semilogx([r['n'] for r in rows],[100*r[key] for r in rows],'o-',label=label)
    axes[1,0].axhline(0,color='black',lw=.6)
    axes[1,0].set(xlabel='Radial cell count',ylabel='Signed capture-rate error (%)',title='Strong capture: coarse-grid cancellation')
    axes[1,0].legend(fontsize=8)
    axes[1,1].set(xlabel='Physical reaction width',ylabel='Difference from Robin limit (%)',title='Separate width study; no mesh-width substitution')
    axes[1,1].legend(fontsize=8)
    fig.suptitle('Finite-width spherical trapping: one retained coarse-grid monotonicity failure')
    save(fig,'spherical-capture')

    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    with np.load(folder/'fields.npz') as data:
        for name,color in zip(('capture','release'),colors):
            history=data[name+'_history'];row=next(r for r in report['closed'] if r['name']==name)
            axes[0,0].plot(history[:,0],history[:,2]-history[0,2],color=color,label=name)
            equilibrium=row['equilibrium_concentration']
            axes[0,1].semilogy(history[:,0],np.abs(history[:,4]-equilibrium)/equilibrium,color=color,label=name)
            # Subtract the analytically known equilibrium free energy.
            total_volume=4*np.pi*8**3/3+row['capacity']
            energy_equilibrium=.5*total_volume*equilibrium**2
            axes[1,0].semilogy(history[:,0],history[:,3]-energy_equilibrium,color=color,label=name)
    rows=report['time_refinement']
    for key,label,color in zip(('combined_relative_l2','bulk_linf','surface_absolute_error'),
                               ('Combined relative L₂','Bulk L∞','Surface absolute'),colors):
        axes[1,1].loglog([r['dt'] for r in rows],[r[key] for r in rows],'o-',label=label,color=color)
    axes[0,0].set(xlabel='Time',ylabel='Stored population change',title='Reciprocal capture and release')
    axes[0,1].set(xlabel='Time',ylabel='Surface relative equilibrium error',title='Predicted common chemical potential')
    axes[1,0].set(xlabel='Time',ylabel='Free energy above equilibrium',title='Closed-system dissipation')
    axes[1,1].set(xlabel='Timestep',ylabel='Error at t = 8',title='Independent matrix-exponential time reference')
    for ax in axes.flat:ax.legend(fontsize=8)
    fig.suptitle('Mobile defect population + finite-capacity surface storage; geometry fixed')
    save(fig,'spherical-storage')
    (output/'plotting.json').write_text(json.dumps({'plotting_and_figure_io_s':perf_counter()-start},indent=2)+'\n')
    print('Saved two PNG/SVG figure pairs in',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=ROOT/'results/spherical-sinks-2026-09-13')
    render(parser.parse_args().input.resolve())
