"""Section 48 report: accuracy-cost figure and markdown from saved evidence."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['c','eta','T','slip','damage']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());c=r['candidates'];figures=out/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for family,marker in (('graded','s'),('uniform','o')):
        keys=[k for k in c if c[k]['family']==family]
        worst=[max(c[k]['errors']['t0.2'][f] for f in FIELDS)*100 for k in keys]
        axes[0].loglog([c[k]['wall_seconds'] for k in keys],worst,marker+'-',label=family)
        for k,w in zip(keys,worst):axes[0].annotate(f"n{c[k]['n']}",(c[k]['wall_seconds'],w),fontsize=7)
        axes[1].loglog([c[k]['cells'] for k in keys],[c[k]['errors']['t0.2']['slip']*100 for k in keys],marker+'-',label=family)
    for t in r['targets']:axes[0].axhline(100*t,color='0.6',ls=':',lw=.8)
    axes[0].set(xlabel='Launch-to-exit wall time (s)',ylabel='Worst field error vs n128 at t=0.2 (%)',title='Accuracy against cost');axes[0].legend()
    axes[1].set(xlabel='Cells',ylabel='Slip error vs n128 (%)',title='Slip error against cell count');axes[1].legend()
    for ext in ('png','svg'):fig.savefig(figures/f'graded-cost.{ext}',dpi=170)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out;ref=r['reference']
    lines=['# Graded versus uniform notch cost at equal accuracy','',
        f"September 26, 2026. Section 48 measures the cost of static graded Q2 meshes against uniform meshes at equal accuracy for t≤0.20 (Gauss rule, Section 45 physics, preconditioned midpoint), against a common uniform n128 reference. Its own error is estimated by uniform n96 versus n128: worst field {100*max(ref['uniform96_vs_reference'].values()):.2f}%. Static grading is a conventional technique; a favorable ratio is an implementation-level result, not evidence of a novel method.",'',
        '## Errors against the common reference at t=0.20','','| Candidate | Cells | c | eta | T | slip | damage | Wall s | Solver s | Peak RSS MiB | Repeat solver s |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for k,v in c.items():
        e=v['errors']['t0.2'];rep=', '.join(f'{x:.0f}' for x in v['repeat_solver_seconds']) or '—'
        lines.append(f"| {k} | {v['cells']} | "+' | '.join(f"{100*e[f]:.3f}%" for f in FIELDS)+f" | {v['wall_seconds']:.0f} | {v['solver_seconds']:.0f} | {v['peak_rss_bytes']/2**20:.0f} | {rep} |")
    lines+=['','## Cheapest candidate meeting each target (all five fields)','','| Target | Graded | Uniform | Ratio (uniform/graded wall) | Kind |','| --- | --- | --- | ---: | --- |']
    for t,pick in r['cheapest'].items():
        ratio=r['ratios'][t];lines.append(f"| {100*float(t):g}% | {pick['graded'] or 'none'} | {pick['uniform'] or 'none'} | {ratio.get('wall',float('nan')):.2f} | {ratio['kind']} |")
    lines+=['',r.get('interpretation',''),'',
        f'![Accuracy against cost]({rel}/figures/graded-cost.png)','',
        '## Timing conditions','',"The six candidates ran concurrently as single-thread fresh processes, one per performance core, while the other studies ran. The repeats of graded n48 and n64 ran later under a different load. Their spread, which includes that load difference, is the declared threshold a ratio must exceed. Solver seconds exclude initialization and output; wall seconds are launch-to-exit.",'',
        '```sh',f'python3 benchmarks/graded_cost.py --output {rel} --timed FAMILY N REPEAT   # per candidate','python3 benchmarks/graded_cost.py --output '+str(rel),f'python3 benchmarks/graded_cost_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 48](unified_multiscale_defect_framework.md).','']
    write_doc('GRADED_COST_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
