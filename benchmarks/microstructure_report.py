"""Plot saved grain, precipitate and pore evidence without rerunning solvers."""

import argparse,json
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def render(folder):
    started=perf_counter();output=folder/'figures';output.mkdir(exist_ok=True)
    summary=json.loads((folder/'summary.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for extension in ('png','svg'):fig.savefig(output/f'{name}.{extension}',dpi=170,bbox_inches='tight')
        plt.close(fig)

    fig,axes=plt.subplots(1,4,figsize=(14,3.5),layout='constrained')
    palette=np.array([[.23,.40,.70],[.79,.44,.27],[.28,.58,.47]])
    for ax,row in zip(axes,summary['final_junctions']):
        with np.load(folder/row['folder']/'fields.npz') as data:
            rgb=np.einsum('ayx,ac->yxc',data['phases'],palette)
            ax.imshow(rgb,origin='lower',extent=(-16,16,-16,16))
        cx,cy=row['rings'][-1]['center'];theta=np.linspace(0,2*np.pi,361)
        ax.plot(cx+6*np.cos(theta),cy+6*np.sin(theta),'w--',lw=.8)
        ax.set(xlim=(-7,7),ylim=(-7,7),xlabel='x',ylabel='y',title=f"n = {row['n']}")
    rows=summary['final_junctions']
    axes[3].loglog([r['h'] for r in rows],[r['rings'][-1]['mean_abs_error'] for r in rows],'o-')
    axes[3].set(xlabel='Grid spacing',ylabel='Mean |angle − 120°|',title='Fixed ring radius = 6')
    fig.suptitle('Stationary equal-tension junctions; fixed physical width π')
    save(fig,'triple-junctions')

    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    colors=['#3b65ae','#b3782b','#2b8469']
    all_area=[]
    for row,color in zip(sorted(summary['long_grain_statistics'],key=lambda r:r['metrics']['seed']),colors):
        with np.load(folder/row['folder']/'fields.npz') as data:
            axes[0,0].step(data['time'],4096/data['grain_count'],where='post',color=color,label=f"seed {row['metrics']['seed']}")
            axes[0,1].scatter(data['expected_area_rate'],data['observed_area_rate'],s=5,alpha=.12,color=color,rasterized=True)
            ids=np.unique(data['final_labels']);area=data['areas'][-1,ids];area=area[area>1]
            all_area.extend((area/area.mean()).tolist())
            if row['metrics']['seed']==11:
                axes[1,0].imshow(data['final_labels'],origin='lower',extent=(-32,32,-32,32),cmap='tab20',interpolation='nearest')
    axes[0,0].set(xlabel='Time',ylabel='Domain area / active grain count',title='Finite-time mean-area growth')
    axes[0,0].legend(fontsize=8)
    axes[0,1].plot([-5,5],[-5,5],'k--',lw=1,label='von Neumann–Mullins')
    axes[0,1].set(xlabel='π (sides − 6) / 3',ylabel='Measured dA/dt',title='Large grains with stable neighbor count',xlim=(-5,5),ylim=(-5,5))
    axes[0,1].legend(fontsize=8)
    axes[1,0].set(xlabel='x',ylabel='y',title='Seed 11 at t = 160; periodic domain')
    axes[1,1].hist(all_area,bins=np.linspace(0,max(all_area)*1.01,13),density=True,color='#678a9b',edgecolor='white')
    axes[1,1].set(xlabel='Grain area / run mean area',ylabel='Sample density',title=f'Final normalized areas; {len(all_area)} surviving grains')
    fig.suptitle('Three fixed initial seeds; finite ensemble, not an asymptotic scaling certification')
    save(fig,'grain-statistics')

    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    theta=np.linspace(0,2*np.pi,129)
    for n in (33,65,97,129):
        with np.load(folder/f'precipitate-{n}'/'fields.npz') as data:
            radii=np.r_[data['contour_radii'],data['contour_radii'][0]]
            axes[0].plot(radii*np.cos(theta),radii*np.sin(theta),label=f'n = {n}')
    axes[0].set(xlabel='x',ylabel='y',title='Stationary v = 0 contours',aspect='equal');axes[0].legend(fontsize=8)
    rows=summary['precipitate_contour_errors']
    axes[1].loglog([32/r['n'] for r in rows],[r['rms'] for r in rows],'o-')
    axes[1].set(xlabel='Grid spacing',ylabel='Contour-radius RMS error',title='Reference n = 129')
    timing=summary['condensation_timing']
    for i,mode in enumerate(('fft','condensed')):
        times=timing['total_compute_runs'][mode]
        axes[2].bar(i,np.median(times),color=colors[i]);axes[2].scatter([i]*len(times),times,color='black',s=13)
    axes[2].set(xticks=[0,1],xticklabels=['Generic FFT','Exact condensation'],ylabel='Computation time (s)',title='Same-grid equilibrium solve + diagnostics')
    fig.suptitle('Homogeneous dilatational precipitate: exact special-case reduction, not a novelty claim')
    save(fig,'precipitate-convergence')

    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    pore=json.loads((folder/'pore-capillary/report.json').read_text());rows=pore['rows'];finest=[r for r in rows if r['n']==256]
    inverse=np.linspace(1/15,1/5,100);gamma=pore['surface_energy_exact']
    axes[0,0].plot(inverse,gamma*inverse,label='Energetic γ/R')
    axes[0,0].plot(inverse,1.2*gamma*inverse,'--',label='Initial zero-growth 6γ/(5R)')
    axes[0,0].scatter([1/r['radius'] for r in finest],[r['capillary_potential'] for r in finest],s=20)
    axes[0,0].scatter([1/r['radius'] for r in finest],[r['initial_neutral_potential'] for r in finest],marker='s',s=20)
    axes[0,0].set(xlabel='1 / radius',ylabel='Chemical potential',title='Different thermodynamic and transient tests');axes[0,0].legend(fontsize=8)
    for key,label in [('capillary_relative_error','Energetic potential'),('neutral_relative_error','Initial neutral potential')]:
        axes[0,1].loglog([64/n for n in (128,192,256)],
                         [max(r[key] for r in rows if r['n']==n) for n in (128,192,256)],'o-',label=label)
    axes[0,1].set(xlabel='Grid spacing',ylabel='Maximum relative error across radii',title='Fixed profile width');axes[0,1].legend(fontsize=8)
    with np.load(folder/'pore-time-0.0025/fields.npz') as data:
        for tag in ('low','high'):axes[1,0].plot(data[tag+'_time'],data[tag+'_area']-data[tag+'_area'][0],label=tag+' supersaturation')
    axes[1,0].set(xlabel='Time',ylabel='Pore-area change',title='Closed vacancy inventory');axes[1,0].legend(fontsize=8)
    for key in ('low_p','low_n','high_p','high_n'):
        rows=summary['pore_time_errors'];axes[1,1].loglog([r['dt'] for r in rows],[r['rms_errors'][key] for r in rows],'o-',label=key)
    axes[1,1].set(xlabel='Timestep',ylabel='Field RMS error',title='Reference dt = 0.0025');axes[1,1].legend(fontsize=8)
    fig.suptitle('Pore capillarity and finite-time kinetics with the original exchange mobility')
    save(fig,'pore-kinetics')
    (output/'plotting.json').write_text(json.dumps({'plotting_and_figure_io_s':perf_counter()-started},indent=2))
    print('Saved four PNG/SVG pairs in',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=ROOT/'results/microstructure-2026-09-13')
    render(parser.parse_args().input.resolve())
