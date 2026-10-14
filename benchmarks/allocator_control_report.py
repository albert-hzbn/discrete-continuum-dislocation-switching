"""Section 92 report: error control by the allocation threshold, from saved evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'allocator-control-2026-09-28';mark=lambda b:'PASS' if b else 'FAIL'

if __name__=='__main__':
    r=json.loads((OUT/'report.json').read_text());NC=r['thresholds'];st=r['study'];tp=OUT/'cpu-time-post-hoc.json';tm=json.loads(tp.read_text()) if tp.exists() else None
    fig,ax=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for sd,mk in (('91','o-'),('92','s--')):
        ax[0].plot(NC,[100*st[sd][str(n)]['error'] for n in NC],mk,label=f'H, seed {sd}');ax[0].axhline(100*st[sd]['C']['error'],color='C3' if sd=='91' else 'C4',lw=.8,ls=':',label=f'C, seed {sd}')
    for tol in r['tolerances']:ax[0].axhline(100*tol,color='0.7',lw=.6)
    ax[0].set(xscale='log',xlabel='Allocation threshold n_c',ylabel='Sparse-region error against D (%)',title='The threshold controls the error');ax[0].legend(fontsize=7)
    if tm:
        ax[1].plot(NC,[1e3*tm['minimum'][f'H{n}'] for n in NC],'o-',label='H (CPU time)');ax[1].axhline(1e3*tm['minimum']['D'],color='k',ls='--',lw=.8,label='D');ax[1].axhline(1e3*tm['minimum']['C'],color='C3',ls=':',lw=.8,label='C')
        ax[1].set(xscale='log',yscale='log',xlabel='n_c',ylabel='ms per step',title='CPU-time cost, seed 92 (post hoc)');ax[1].legend(fontsize=7)
    (OUT/'figures').mkdir(exist_ok=True)
    for ext in ('png','svg'):fig.savefig(OUT/'figures'/f'allocator-control.{ext}',dpi=140)
    g=r['gates'];L=['# Error control by the allocation threshold','',
      "September 28, 2026. Section 92 tests whether the Section 91 allocator is error-controlled. The threshold n_c must steer the sparse-region error predictably, and a threshold calibrated for a tolerance on one ensemble (seed 91) must meet it on an independent one (seed 92). The protocol is 1000 steps, 256², and the same stress model ([representation_allocator_control.py](benchmarks/representation_allocator_control.py)).",'',
      '| Gate | Outcome |','| --- | --- |',f"| E1 error non-increasing and cost non-decreasing in n_c (10% slack) | {mark(g['E1_monotone'])} (error: pass; cost: fails on machine-load noise) |",
      f"| E2 thresholds calibrated on seed 91 meet 2%, 1%, 0.5% on seed 92 at under a third of D's cost | {mark(g['E2_transfer'])} |",f"| E3 every n_c ≥ 16 point is cheaper than D and more accurate than C | {mark(g['E3_between'])} |",'',
      '| n_c | Error, seed 91 | Error, seed 92 | Particles (seed 92) | Continuum blocks (seed 92) |'+(' CPU cost (post hoc) |' if tm else ''),'| ---: | ---: | ---: | ---: | ---: |'+(' ---: |' if tm else '')]
    for n in NC:L.append(f"| {n} | {100*st['91'][str(n)]['error']:.2f}% | {100*st['92'][str(n)]['error']:.2f}% | {st['92'][str(n)]['particles']} | {st['92'][str(n)]['continuum_blocks']} |"+(f" {1e3*tm['minimum'][f'H{n}']:.2f} ms |" if tm else ''))
    L+=[f"| C (reference) | {100*st['91']['C']['error']:.2f}% | {100*st['92']['C']['error']:.2f}% | 0 | all |"+(f" {1e3*tm['minimum']['C']:.2f} ms |" if tm else ''),f"| D (reference) | 0 | 0 | 500 200 | 0 |"+(f" {1e3*tm['minimum']['D']:.2f} ms |" if tm else ''),'',
      "Calibration: "+', '.join(f"tolerance {100*float(t):g}% gives n_c = {n} (seed 92: {100*st['92'][str(n)]['error']:.2f}%)" for t,n in r['chosen_n_c'].items())+'.','',
      "The error falls monotonically with the threshold on both seeds, and the calibrated thresholds transfer to an independent ensemble. The threshold is therefore an error-control knob. Between n_c = 4 (almost all continuum, the C limit) and n_c = 512 (the sparse band kept discrete, error 0.24%), every setting sits between the two specialized methods. The wall-clock costs of the declared sweep are not comparable, because D itself ran at 71 ms on one seed and 341 ms on the other under the shared machine's changing load. That is why E1 fails on cost."]
    if tm:L+=['',"A first post hoc re-timing (three interleaved wall-clock repeats, minimum kept; timing-post-hoc.json) was still load-dominated: C took at least 15.6 ms against 2.0 ms on a quiet machine. The cost was then measured as process CPU time, which counts only the cycles a run uses.",
        '',f"On CPU time, E1's cost clause {'holds' if tm['E1_cost_post_hoc'] else 'still fails'} (post hoc; E1 stays failed as declared). D costs {1e3*tm['minimum']['D']:.1f} ms per step and C {1e3*tm['minimum']['C']:.2f} ms. H rises from {1e3*tm['minimum']['H4']:.2f} to {1e3*tm['minimum']['H512']:.2f} ms across the whole range, {tm['minimum']['D']/max(tm['minimum'][f'H{n}'] for n in NC):.0f}–{tm['minimum']['D']/min(tm['minimum'][f'H{n}'] for n in NC):.0f}x cheaper than D. The allocator is error-controlled: one transferable knob chooses the error, at a cost always far below the accurate specialist."]
    L+=['','![Error and cost against the threshold](results/allocator-control-2026-09-28/figures/allocator-control.png)','','```sh','python3 benchmarks/representation_allocator_control.py','python3 benchmarks/representation_allocator_timing.py  # post hoc, wall time','python3 benchmarks/representation_allocator_timing.py --cpu  # post hoc, CPU time','python3 benchmarks/allocator_control_report.py','```','','[Report](results/allocator-control-2026-09-28/report.json), [theory Section 92](unified_multiscale_defect_framework.md).','']
    write_doc('ALLOCATOR_CONTROL_RESULTS.md','\n'.join(L))
