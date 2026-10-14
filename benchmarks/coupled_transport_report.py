"""Plot saved Section 33 evidence without recomputing trajectories."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import argparse,csv,json
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=ROOT/'results/coupled-transport-2026-09-22')
    args=parser.parse_args();started=perf_counter();folder=args.input
    summary=json.loads((folder/'summary.json').read_text());out=folder/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for name in ('field-21','affine-21','frozen-gamma-21','frozen-fields-21'):
        with np.load(folder/name/'fields.npz') as d:
            axes[0,0].plot(d['time'],d['energy'],label=name,ls='--' if name=='affine-21' else '-')
    axes[0,0].set(xlabel='Time',ylabel='Total energy',title='Coupled and frozen-block trajectories')
    axes[0,0].legend(fontsize=8)
    for key,label in (('phase','Phase excess'),('population','Population excess'),('alpha','Nye density'),('stress','Stress')):
        axes[0,1].semilogy([summary['reports'][r['run']]['n'] for r in summary['space']],
            [100*r['errors'][key]['relative_l2'] for r in summary['space']],'o-',label=label)
        axes[1,0].semilogy([summary['reports'][r['run']]['n'] for r in summary['matched']],
            [100*r['errors'][key]['relative_l2'] for r in summary['matched']],'o-',label=label)
    axes[0,1].set(xlabel='Grid n per axis',ylabel='Relative L2 discrepancy (%)',title='Full field against n=41 spectral restriction')
    axes[1,0].set(xlabel='Grid n per axis',ylabel='Relative L2 discrepancy (%)',title='Affine gamma against matched full evolution')
    axes[0,1].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    x=np.arange(3)
    for i,kind in enumerate(('field','affine')):
        axes[1,1].bar(x+(i-.5)*.34,[summary['reports'][f'{kind}-{n}']['timings']['online_s'] for n in (13,21,29)],.32,label=kind)
    axes[1,1].set(xticks=x,xticklabels=['13','21','29'],xlabel='Grid n per axis',ylabel='Online seconds',title='Single-run cost; no repeated-timing claim')
    axes[1,1].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(out/f'coupled-verification.{ext}',dpi=160)
    plt.close(fig)
    with np.load(folder/'field-21/fields.npz') as a,np.load(folder/'affine-21/fields.npz') as b:
        fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
        arrays=[(a['phase']+1)/2,a['population']-.2,np.sqrt(np.sum(a['alpha']**2,axis=(0,1)))]
        errors=[(b['phase']-a['phase'])/2,b['population']-a['population'],np.sqrt(np.sum((b['alpha']-a['alpha'])**2,axis=(0,1)))]
        for i,label in enumerate(('Structural phase coordinate','Population excess','Nye density norm')):
            for j,values in enumerate((arrays,errors)):
                ax=axes[j,i];im=ax.imshow(values[i][:,:,10].T,origin='lower',extent=(-6,6,-6,6),cmap='viridis' if j==0 else 'magma')
                ax.set(title=label if j==0 else 'Affine − full' if i<2 else 'Affine − full tensor norm',xlabel='x',ylabel='y')
                fig.colorbar(im,ax=ax,shrink=.75)
        fig.suptitle('n=21, t=1, z=0 slice; fixed physical widths')
        for ext in ('png','svg'):fig.savefig(out/f'coupled-fields.{ext}',dpi=160)
        plt.close(fig)
    keys=['n','alpha_error_pct','phase_error_pct','population_error_pct','energy_error_pct','field_online_s','affine_online_s','gamma_coordinates','total_coordinates']
    with (folder/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in summary['matched']:
            a=summary['reports'][r['run']];b=summary['reports'][r['reference']]
            writer.writerow({'n':a['n'],'alpha_error_pct':100*r['errors']['alpha']['relative_l2'],
                'phase_error_pct':100*r['errors']['phase']['relative_l2'],'population_error_pct':100*r['errors']['population']['relative_l2'],
                'energy_error_pct':100*r['energy_relative'],'field_online_s':b['timings']['online_s'],'affine_online_s':a['timings']['online_s'],
                'gamma_coordinates':a['final_gamma_coordinates'],'total_coordinates':a['final_total_coordinates']})
    (out/'plotting.json').write_text(json.dumps({'plotting_tables_and_io_s':perf_counter()-started},indent=2)+'\n')
    print('Generated comparison CSV and two PNG/SVG figure pairs.')


if __name__=='__main__':main()
