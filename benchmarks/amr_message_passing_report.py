"""Section 61 report: AMR baseline and message-passing scaling, from saved evidence."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
mark=lambda b:'PASS' if b else 'FAIL'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;a=json.loads((out/'amr/report.json').read_text());m=json.loads((out/'message-passing/report.json').read_text());f=np.load(out/'amr/fields.npz')
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    d=f['amr']-f['uniform'];im=axes[0].imshow(d,origin='lower',extent=(-64,64,-64,64),cmap='RdBu_r',vmin=-abs(d).max(),vmax=abs(d).max());fig.colorbar(im,ax=axes[0],shrink=.8)
    axes[0].contour(np.linspace(-64,64,256),np.linspace(-64,64,256),f['uniform'],levels=[0],colors='k',linewidths=.6);axes[0].set(title='AMR minus uniform φ at t=12',xlim=(-24,24),ylim=(-24,24))
    names=['uniform','focused','amr'];c=a['cost'];x=np.arange(3)
    axes[1].bar(x-.2,[c[k]['solver_plus_adaptation_s'] for k in names],.4,label='solver + adaptation (s)');ax=axes[1].twinx();ax.bar(x+.2,[c[k]['mean_cells'] for k in names],.4,color='C1',label='mean active cells')
    axes[1].set(xticks=x,xticklabels=['uniform n=256','focused tensor','quadtree AMR'],title='Case 1 cost at equal physics');axes[1].legend(loc='upper left',fontsize=8);ax.legend(loc='upper right',fontsize=8)
    P=[r['parts'] for r in m['strong']['uniform']]
    axes[2].plot(P,P,'k--',lw=.8,label='ideal')
    for k,s in (('uniform','o-'),('adaptive','s-')):axes[2].plot(P,[r['speedup'] for r in m['strong'][k]],s,label=f'{k} strong speedup')
    axes[2].plot(P,[r['efficiency'] for r in m['weak']],'^-',label='uniform weak efficiency');axes[2].set(xlabel='Processes',title='Message-passing strips (one node)',xticks=P);axes[2].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'amr-message-passing.{ext}',dpi=160)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out;g=a['gates'];acc=a['accuracy_vs_uniform'];sp=a['speedups']
    lines=['# Independent AMR baseline and message-passing scaling','',
        'September 26, 2026. Section 61 adds the two remaining section 19 measurements. The first is a speedup relative to an adaptive-mesh PF code that is independent of this project\'s focused tensor grids: a standard block-structured quadtree with 16×16-cell blocks, an interface refinement criterion, a one-block buffer, 2:1 balance and stacked vectorized updates. The second is scaling under message passing: separate processes with private memory exchange halo rows through pipes.','',
        '## AMR baseline (case 1 physics, three single-thread runs, median)','',
        '| Method | Solver + adaptation (s) | Solver (s) | Adaptation (s) | Mean active cells | Final cells |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for k,label in (('uniform','uniform n=256'),('focused','focused tensor grid (this project)'),('amr','quadtree AMR (independent baseline)')):
        lines.append(f"| {label} | {c[k]['solver_plus_adaptation_s']:.4f} | {c[k]['solver_s']:.4f} | {c[k]['adaptation_s']:.4f} | {c[k]['mean_cells']:.0f} | {c[k]['final_cells']} |")
    lines+=['',f"Speedups (solver + adaptation): uniform/AMR {sp['uniform_over_amr']:.2f}x, uniform/focused {sp['uniform_over_focused']:.2f}x, focused/AMR {sp['focused_over_amr']:.2f}x (above 1 means AMR is faster than the focused grid). Final AMR structure: {a['amr_structure']['blocks']} leaf blocks with level counts {a['amr_structure']['leaf_levels']}.",'',
        '| Gate | Outcome | Evidence |','| --- | --- | --- |',
        f"| A1 accuracy of the AMR baseline | {mark(g['A1_accuracy'])} | against uniform n=256: phase-fraction relative L2 {acc['amr']['relative_l2']:.2e}, radius difference {acc['amr']['radius_difference']:.2e} (focused grid: {acc['focused']['relative_l2']:.2e}, {acc['focused']['radius_difference']:.2e}) |",
        f"| A2 prediction: fewer cells than the focused grid | {mark(g['A2_prediction_fewer_cells'])} | {c['amr']['mean_cells']:.0f} against {c['focused']['mean_cells']:.0f} |",
        f"| A2 prediction: slower than the focused grid | {mark(g['A2_prediction_slower'])} | {c['amr']['solver_plus_adaptation_s']:.4f} s against {c['focused']['solver_plus_adaptation_s']:.4f} s |",'',
        f"Post hoc block-size sensitivity, labelled as such, because the declared 16×16 blocks are one design choice among several: "+'; '.join(f"{k}×{k} blocks {v['mean_cells']:.0f} cells, {v['solver_plus_adaptation_s']:.3f} s" for k,v in json.loads((out/'amr/post-hoc-block-size.json').read_text())['rows'].items())+f". With 8×8 blocks the quadtree uses fewer cells than the focused grid ({c['focused']['mean_cells']:.0f}), so the declared cell-count prediction holds there and fails at 16×16. At every block size the AMR baseline is 1.6–2.5x slower than the focused tensor grid at equal accuracy. The sparse ghost gather and the per-block stencil assembly cost more per cell than a single tensor-product stencil. Both solvers here are NumPy implementations. An optimized compiled AMR library (for example AMReX) has far lower per-cell overhead, so this ranking is a statement about these implementations, not about AMR in general.",'',
        '## Message-passing scaling','','| Processes | Uniform 1024² ms/step | Speedup | Efficiency | Adaptive 332² ms/step | Speedup | Uniform / adaptive | Weak grid | Weak efficiency | Bitwise equal to serial |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for i,p in enumerate(P):
        u,ad,w=m['strong']['uniform'][i],m['strong']['adaptive'][i],m['weak'][i]
        lines.append(f"| {p} | {1e3*u['seconds_per_step']:.3f} | {u['speedup']:.2f} | {u['efficiency']:.2f} | {1e3*ad['seconds_per_step']:.3f} | {ad['speedup']:.2f} | {m['adaptive_over_uniform'][str(p)]:.2f} | {w['n']}² | {w['efficiency']:.2f} | {'yes' if u['bitwise'] and ad['bitwise'] and w['bitwise'] else 'NO'} |")
    lines+=['',f"D1 bitwise equality with the serial step: {mark(m['gates']['D1_bitwise'])}. The first declared run deadlocked: both ranks sent before receiving, and 8 KB rows exceed the macOS socket-pair buffer ([log](results/amr-message-passing-2026-09-26/message-passing/run-first-attempt-deadlock.log)). The exchange now receives first on the upper side of each boundary. These processes share one node, so the operating system carries the messages; a cluster's network latency and bandwidth are not measured.",'',
        f'![AMR difference, cost comparison and message-passing scaling]({rel}/figures/amr-message-passing.png)','','```sh',f'python3 benchmarks/amr_baseline.py --output {rel}/amr',f'python3 benchmarks/message_passing.py --output {rel}/message-passing',f'python3 benchmarks/amr_message_passing_report.py --input {rel}','```','',
        f'[AMR report]({rel}/amr/report.json), [message-passing report]({rel}/message-passing/report.json), [theory Section 61](unified_multiscale_defect_framework.md).','']
    write_doc('AMR_MESSAGE_PASSING_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
