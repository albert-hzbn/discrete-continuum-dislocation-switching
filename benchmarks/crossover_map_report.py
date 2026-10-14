"""Section 67 report: where adaptive resolution stops paying, from saved evidence."""
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
    out=args.input;r=json.loads((out/'report.json').read_text());rows=r['rows'];g=r['gates']
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    a=np.array([x['active_fraction'] for x in rows]);gg=np.linspace(.2,1,200)
    axes[0].plot(gg**2,1/(gg+(1-gg)/4)**2,'k--',lw=.9,label='cell ratio (g + (1-g)/4)^-2')
    axes[0].plot(a,[x['speedup'] for x in rows],'o-',label='measured speedup');axes[0].axhline(1,color='0.6',lw=.8)
    axes[0].set(xlabel='Fine-square area fraction g²',ylabel='Uniform time / adaptive time',title='Adaptive advantage against active fraction');axes[0].legend(fontsize=8)
    axes[1].plot(a,r['efficiency'],'s-');axes[1].axhspan(.5,1.2,color='C2',alpha=.08,label='declared C3 band')
    axes[1].set(xlabel='Fine-square area fraction g²',ylabel='Speedup / cell ratio',title='Per-cell efficiency of the focused grid',ylim=(.4,1.3));axes[1].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'crossover-map.{ext}',dpi=150)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out
    lines=['# Where adaptive resolution stops paying','',
        'September 27, 2026. Section 67 measures the speedup of the focused tensor grid (Section 12) over uniform phase-field on the case 1 Allen–Cahn step as the active region grows. A circle of radius R needs a fine band of half-width R + 10, so each axis is fine over a fraction g = 2(R + 10)/L. The predicted cell ratio is (g + (1 - g)/4)^-2, and the fine square covers an area fraction g². L = 128, h = 0.5, factor-4 coarsening, dt = 0.02, 200 steps. Timing uses the NumPy solver on one thread after a 0.5 s sustained warm-up, best of three.','',
        '| Gate | Outcome |','| --- | --- |']+[f'| {k} | {mark(v)} |' for k,v in g.items()]
    lines+=['','| R | g | Area fraction g² | Adaptive cells | Cell ratio | Speedup | Speedup / cell ratio | Relative L2 vs uniform |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for x,e in zip(rows,r['efficiency']):
        lines.append(f"| {x['radius']:g} | {x['fine_axis_fraction']:.3f} | {x['active_fraction']:.3f} | {x['adaptive_cells']} | {x['cell_ratio']:.2f} | {x['speedup']:.2f} | {e:.2f} | {x['relative_l2']:.1e} |")
    lines+=['','The speedup tracks the cell ratio and falls to 1.0 when the fine band covers the domain. The per-cell efficiency rises from 0.78 to about 1.0 as the grid grows, because a small grid pays proportionally more fixed per-step overhead and the nonuniform stencil costs a little more per cell. The map gives the practical rule: on this benchmark the focused grid pays 2x or more only while the fine band covers less than about 30% of the domain area, and it pays nothing once the active region fills the domain. The uniform grid has 65,536 cells, so this map says nothing about how the constant changes in larger 3D problems.','',
        'Run record: the timing ran after all of this session\'s simulations had stopped, but the user\'s own desktop applications kept the machine at a load average of about 12 on 14 cores. Each speedup is the ratio of two single-thread timings taken back to back, and the efficiency trend is smooth, so the load does not change the conclusion. The absolute seconds in the report are not a clean benchmark.','',
        f'![Speedup and per-cell efficiency against active fraction]({rel}/figures/crossover-map.png)','','```sh',
        f'python3 benchmarks/crossover_map.py --output {rel}',f'python3 benchmarks/crossover_map_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 67](unified_multiscale_defect_framework.md).','']
    write_doc('CROSSOVER_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
