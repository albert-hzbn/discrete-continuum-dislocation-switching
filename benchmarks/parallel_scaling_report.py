"""Section 55 report: figures and markdown from the saved thread-scaling measurements."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());figures=out/'figures';figures.mkdir(exist_ok=True)
    P=[int(p) for p in r['threads']];s=r['ac_strong_scaling'];w=r['ac_weak_efficiency'];cs=r['ch_strong_scaling'];cw=r['ch_weak_efficiency']
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    axes[0].plot(P,P,'k--',lw=.8,label='ideal')
    for key,label,style in (('uniform','AC uniform PF','o-'),('adaptive','AC focused adaptive','s-')):
        axes[0].plot(P,[s[key][str(p)]['speedup'] for p in P],style,label=label)
    axes[0].plot(P,[cs[str(p)]['speedup'] for p in P],'^-',label='CH spectral uniform')
    axes[0].set(xlabel='Threads',ylabel='Speedup over one thread',title='Strong scaling',xticks=P);axes[0].legend(fontsize=8)
    axes[1].plot(P,[w[str(p)]['efficiency'] for p in P],'o-',label='AC uniform')
    axes[1].plot(P,[cw[str(p)]['efficiency'] for p in P],'^-',label='CH spectral')
    axes[1].axhline(1,color='k',ls='--',lw=.8);axes[1].set(xlabel='Threads',ylabel='Weak-scaling efficiency',title='Weak scaling (cells per thread fixed)',xticks=P,ylim=(0,1.15));axes[1].legend(fontsize=8)
    axes[2].plot(P,[r['adaptive_over_uniform_speedup'][str(p)] for p in P],'s-',color='C1')
    axes[2].set(xlabel='Threads',ylabel='Uniform time / adaptive time',title='Adaptive advantage at equal thread count',xticks=P,ylim=(0,None))
    for ext in ('png','svg'):fig.savefig(figures/f'parallel-scaling.{ext}',dpi=170)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out;g=r['grids'];ms=lambda x:f"{1e3*x:.3f}"
    lines=['# Measured thread scaling of uniform and adaptive grids','',
        f"September 26, 2026. Section 55 measures the strong and weak scaling that the brief (section 19) asks for where feasible. The Allen–Cahn finite-volume step of case 1 runs as a row-parallel Numba kernel with no reductions. It matches the NumPy solver to {r['equivalence']['uniform']['max_difference_to_numpy']:.1e} and gives bitwise-identical results for every thread count, so each speedup compares identical arithmetic. The Cahn–Hilliard spectral step uses SciPy's threaded FFT. Hardware: {r['environment']['platform']}, {r['environment']['cpu_count']} cores (10 performance, 4 efficiency), Numba threading layer '{r['environment']['threading_layer']}'. This is shared-memory threading on one machine, not distributed-memory domain decomposition.",'',
        '## Protocol','',
        f"Case 1 physics (circle radius {r['radius']:g}, h={r['h']:g} in the active region, dt={r['dt']:g}) on a larger domain of side 512, so that each thread has work. Uniform PF: {g['uniform']['shape'][0]}² = {g['uniform']['cells']:,} cells. Focused adaptive grid (fine band [-25,25]², factor-4 coarsening elsewhere, as in case 1): {g['adaptive']['shape'][0]}² = {g['adaptive']['cells']:,} cells. Both evolve the same state. Over the timed interval (t=4) they agree to relative L2 {r['accuracy_t4']['relative_l2']:.1e} (L∞ {r['accuracy_t4']['linf']:.1e}), so the ratio compares equal accuracy. Each configuration gets a 0.5 s sustained warm-up, then five repeats of about 0.6 s; the median time per step is reported. One-time JIT compilation ({r['compile_seconds']:.2f} s) is excluded, as the brief requires.",'',
        'Retained measurement failure: the first run used a fixed number of steps per repeat. Short repeats on the small grids were sometimes timed before macOS moved the thread to a performance core. The single-thread weak baseline (512²) then ran at 17 ns per cell against 3.3 ns for 1024², which produced impossible weak efficiencies above 4. A probe showed that the per-cell time of small grids alternated between the two levels from trial to trial, while long runs were stable. The protocol above was changed before the reported run; its numbers replace the first run and are not selected from it.','',
        '## Strong scaling (time per step, ms)','','| Threads | AC uniform | speedup | efficiency | AC adaptive | speedup | efficiency | Uniform / adaptive | CH spectral 1024² | speedup |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p in P:
        k=str(p);lines.append(f"| {p} | {ms(r['ac_strong']['uniform'][k]['median'])} | {s['uniform'][k]['speedup']:.2f} | {s['uniform'][k]['efficiency']:.2f} | {ms(r['ac_strong']['adaptive'][k]['median'])} | {s['adaptive'][k]['speedup']:.2f} | {s['adaptive'][k]['efficiency']:.2f} | {r['adaptive_over_uniform_speedup'][k]:.2f} | {ms(r['ch_strong'][k]['median'])} | {cs[k]['speedup']:.2f} |")
    lines+=['','## Weak scaling (cells per thread fixed)','','| Threads | AC grid | AC ns per cell | AC efficiency | CH grid | CH ms per step | CH efficiency |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p in P:
        k=str(p);a=r['ac_weak'][k];c=r['ch_weak'][k]
        lines.append(f"| {p} | {a['n']}² | {1e9*a['median']/a['cells']:.2f} | {w[k]['efficiency']:.2f} | {c['n']}² | {ms(c['median'])} | {cw[k]['efficiency']:.2f} |")
    nu=r['numpy_single_thread']
    lines+=['',f"For context, the original single-thread NumPy step takes {ms(nu['uniform']['median'])} ms (uniform) and {ms(nu['adaptive']['median'])} ms (adaptive); the compiled kernel is {nu['uniform']['median']/r['ac_strong']['uniform']['1']['median']:.1f}x faster on one thread. Earlier cost studies used the NumPy solver for both sides, so their ratios are unaffected.",'',
        f'![Strong scaling, weak scaling and the adaptive advantage by thread count]({rel}/figures/parallel-scaling.png)','',
        '## Findings','',
        f"1. The explicit AC stencil scales well until memory bandwidth saturates: the uniform grid reaches {s['uniform']['4']['speedup']:.2f}x on 4 threads and {s['uniform']['8']['speedup']:.2f}x on 8 (efficiency {s['uniform']['8']['efficiency']:.2f}). Weak-scaling efficiency is {w['4']['efficiency']:.2f} at 4 threads and {w['8']['efficiency']:.2f} at 8.",
        f"2. The adaptive grid has {g['uniform']['cells']/g['adaptive']['cells']:.1f}x fewer cells, so it runs out of parallel work sooner: {s['adaptive']['8']['speedup']:.2f}x on 8 threads. Its advantage over uniform PF at equal thread count therefore falls from {r['adaptive_over_uniform_speedup']['1']:.2f}x on one thread to {r['adaptive_over_uniform_speedup']['8']:.2f}x on eight. Adaptivity and parallelism are partly substitutes. On many cores the saving from fewer unknowns shrinks unless the active region itself is large enough to keep every core busy. This is a measured instance of the parallel caveat in theory Section 12.",
        f"3. The threaded 2D FFT of the spectral CH reference scales poorly: {cs['4']['speedup']:.2f}x on 4 threads and {cs['8']['speedup']:.2f}x on 8, with weak-scaling efficiency {cw['8']['efficiency']:.2f} at 8. On this machine, a uniform spectral CH solver gains little from threads.",
        '4. None of these runs measures distributed-memory communication. Halo exchange and the FFT transposes of theory Section 12 remain analytical estimates.','',
        '```sh','python3 benchmarks/parallel_scaling.py --output results/parallel-scaling-2026-09-26',f'python3 benchmarks/parallel_scaling_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 55](unified_multiscale_defect_framework.md).','']
    write_doc('PARALLEL_SCALING_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
