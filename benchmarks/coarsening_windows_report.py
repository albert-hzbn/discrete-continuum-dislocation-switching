"""Plot saved CH coarsening windows, finite-size checks and numerical sensitivity."""
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
    summary=json.loads((folder/'summary.json').read_text())
    colors=['#396cad','#b67c29','#28836b']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for extension in ('png','svg'):fig.savefig(output/f'{name}.{extension}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for row,length in enumerate((128,256)):
        for col,seed in enumerate((1701,2901,4701)):
            with np.load(folder/f'L{length}-seed{seed}/fields.npz') as data:
                image=axes[row,col].imshow(data['field_4096'],origin='lower',extent=(-length/2,length/2,-length/2,length/2),vmin=-1.1,vmax=1.1,cmap='RdBu_r')
            axes[row,col].set(title=f'L = {length}, seed {seed}',xlabel='x',ylabel='y')
    fig.colorbar(image,ax=axes,shrink=.7,label='Conserved order parameter')
    fig.suptitle('Cahn–Hilliard morphology at t = 4096; physical interface scale √2 in every run')
    save(fig,'coarsening-morphology')

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for length,style in ((128,'--'),(256,'-')):
        for seed,color in zip((1701,2901,4701),colors):
            with np.load(folder/f'L{length}-seed{seed}/fields.npz') as data:
                history=data['history'];positive=history[:,0]>=64
                axes[0,0].loglog(history[positive,0],history[positive,3],style,color=color,label=f'L{length}, seed {seed}')
        rows=[r for r in summary['ensemble'] if r['length']==length]
        for metric,color in (('spectral',colors[0]),('energy',colors[1])):
            x=[np.sqrt(r['start']*r['end']) for r in rows]
            axes[0,1].errorbar(x,[r[metric+'_mean'] for r in rows],yerr=[r[metric+'_seed_sd'] for r in rows],
                               fmt='o'+style,color=color,capsize=3,label=f'L{length}, {metric}')
        axes[1,0].plot([np.sqrt(r['start']*r['end']) for r in rows],[r['max_length_over_box'] for r in rows],
                       'o'+style,label=f'L = {length}')
    axes[0,0].set(xlabel='Time',ylabel='Structure-factor length',title='Three seeds and two physical domain sizes')
    axes[0,0].legend(fontsize=7,ncol=2)
    axes[0,1].axhline(1/3,color='black',lw=1,label='1/3');axes[0,1].axhspan(1/3-.05,1/3+.05,color='gray',alpha=.1)
    axes[0,1].set(xscale='log',xlabel='Geometric midpoint of fit window',ylabel='Fitted exponent ± seed SD',title='Fixed windows; seed scatter is not fit uncertainty')
    axes[0,1].legend(fontsize=7,ncol=2)
    axes[1,0].axhline(.2,color='black',ls=':',label='Declared finite-box threshold')
    axes[1,0].set(xscale='log',xlabel='Window midpoint',ylabel='Maximum length / box across seeds',title='Keep finite-size failures visible')
    axes[1,0].legend(fontsize=8)
    with np.load(folder/'L256-seed1701/fields.npz') as data:
        for time,color in zip((256,512,1024,2048,4096),plt.cm.viridis(np.linspace(.1,.9,5))):
            row=data['history'][np.where(data['history'][:,0]==time)[0][0]]
            k=data['radial_k'];power=data[f'radial_{time}'];mask=(k>0)&(power>1e-20)
            axes[1,1].loglog(k[mask]*row[3],power[mask]/(row[5]*row[3]**2),color=color,label=f't = {time}')
    axes[1,1].set(xlabel='k × structure-factor length',ylabel='S(k) / (variance × length²)',title='Scaled spectra: L = 256, seed 1701')
    axes[1,1].legend(fontsize=8)
    fig.suptitle('Extended coarsening evidence; a fitted slope alone does not establish asymptotic scaling')
    save(fig,'coarsening-windows')
    (output/'plotting.json').write_text(json.dumps({'plotting_and_figure_io_s':perf_counter()-start},indent=2)+'\n')
    print('Saved two PNG/SVG figure pairs in',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=ROOT/'results/coarsening-2026-09-13')
    render(parser.parse_args().input.resolve())
