"""Render paired grain-mesh statistics without rerunning evolution."""
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
    report=json.loads((folder/'summary.json').read_text());colors=['#396cad','#b67c29','#28836b']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    with np.load(folder/'comparison.npz') as data:
        for seed,color in zip((11,29,47),colors):
            for n,style in ((96,'--'),(128,'-')):
                axes[0,0].step(data[f'time_{seed}'],data[f'mean_area_{n}_{seed}'],where='post',ls=style,color=color,label=f'seed {seed}, N={n}')
            axes[0,1].plot(data[f'time_{seed}'],100*data[f'area_error_{seed}'],color=color,label=f'seed {seed}')
        for n,color in ((96,colors[0]),(128,colors[1])):
            areas=data[f'normalized_areas_{n}']
            axes[1,0].step(areas,np.arange(1,len(areas)+1)/len(areas),where='post',color=color,label=f'N={n}; {len(areas)} grains')
    axes[0,0].set(xlabel='Time',ylabel='Domain area / active count',title='Same seeds, width, dt and physical observation times')
    axes[0,0].legend(fontsize=7,ncol=2)
    axes[0,1].set(xlabel='Time',ylabel='Summed absolute area discrepancy / domain (%)',title='Matched phase identities, including eliminated grains')
    axes[0,1].legend(fontsize=8)
    axes[1,0].set(xlabel='Grain area / run mean area',ylabel='Empirical cumulative fraction',title='Descriptive distribution comparison')
    axes[1,0].legend(fontsize=8)
    for i,row in enumerate(report['rows']):
        axes[1,1].plot([96,128],100*np.array(row['von_neumann_errors']),'o-',color=colors[i],label=f"seed {row['seed']}")
    axes[1,1].axhline(25,color='black',ls=':',label='Declared rate-error gate')
    axes[1,1].set(xticks=[96,128],xlabel='Cells per axis',ylabel='Qualified von Neumann normalized RMS error (%)',title='Fixed finite-width area-rate diagnostic')
    axes[1,1].legend(fontsize=8)
    fig.suptitle('Grain-statistics mesh sensitivity: two grids and three seeds, not an asymptotic limit')
    for extension in ('png','svg'):fig.savefig(output/f'grain-mesh.{extension}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    (output/'plotting.json').write_text(json.dumps({'plotting_and_figure_io_s':perf_counter()-start},indent=2)+'\n')
    print('Saved grain-mesh PNG/SVG pair in',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=ROOT/'results/grain-mesh-2026-09-13')
    render(parser.parse_args().input.resolve())
