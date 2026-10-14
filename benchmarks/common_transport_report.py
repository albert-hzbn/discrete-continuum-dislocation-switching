"""Figures from shared-mobility trajectories, including the unfavorable cost result."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import argparse,json
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'results/common-transport-2026-09-13');args=p.parse_args()
    started=perf_counter();s=json.loads((args.input/'summary.json').read_text());out=args.input/'figures';out.mkdir(exist_ok=True)
    colors={'field':'#275d8c','plain':'#a86730','adaptive':'#337650','fallback':'#888888'}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name in ('field','plain','adaptive'):
        with np.load(args.input/name/'fields.npz') as data:
            axes[0].plot(data['time'],data['energy']-data['energy'][0],label=name,color=colors[name])
            if len(data['sampled_residual']):
                axes[1].plot(data['sampled_residual'][:,0],100*data['sampled_residual'][:,1],label=name,color=colors[name])
    axes[0].set(title='Same energy and full mobility',xlabel='Time',ylabel='Energy change');axes[0].legend()
    axes[1].axhline(5,color='black',ls=':',lw=1,label='5% kinetic budget')
    axes[1].set(title='Residual of the rate used for each step',xlabel='Time',ylabel='Dissipative residual (%)');axes[1].legend()
    for ext in ('png','svg'):fig.savefig(out/f'common-kinetics.{ext}',dpi=160)
    plt.close(fig)
    names=('field','plain','adaptive','fallback');fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    solver=[s['reports'][name]['timings']['solver_s'] for name in names]
    audit=[s['reports'][name]['timings']['audit_and_enrichment_s'] for name in names]
    axes[0].bar(names,solver,label='Solver');axes[0].bar(names,audit,bottom=solver,label='Audit + enrichment')
    axes[0].set(title='Measured cost, one run each',ylabel='Elapsed seconds');axes[0].legend()
    compared=[r for r in s['comparisons'] if r['reference']=='field']
    bars=axes[1].bar([r['run'] for r in compared],[100*r['errors']['alpha']['relative_l2'] for r in compared],color=[colors[r['run']] for r in compared])
    axes[1].bar_label(bars,labels=[f"{100*r['errors']['alpha']['relative_l2']:.3g}%" for r in compared],padding=4)
    axes[1].set(title='Final Nye-field error against full transport',ylabel='Relative L2 (%)');axes[1].margins(y=.15)
    for ext in ('png','svg'):fig.savefig(out/f'common-cost-error.{ext}',dpi=160)
    plt.close(fig)
    (out/'plotting.json').write_text(json.dumps({'plotting_and_figure_io_s':perf_counter()-started},indent=2)+'\n')
    print('Rendered two PNG/SVG figure pairs.')


if __name__=='__main__':main()
