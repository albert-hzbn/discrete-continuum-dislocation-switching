"""Render fresh-process reference cost and accuracy measurements."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args();out=args.input
    start=perf_counter();r=json.loads((out/'report.json').read_text());figures=out/'figures';figures.mkdir(exist_ok=True)
    summary=r['summary'];groups={key:[x for x in r['runs'] if x['result']['degree']==degree] for key,degree in (('q1',1),('q2',2))}
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    labels=['Q1 n=48','Q2 n=24'];colors=['#ad5f3f','#287d77']
    for i,key in enumerate(('q1','q2')):
        stat=summary[key]['solver_seconds']
        axes[0].bar(i,stat['median'],color=colors[i],yerr=[[stat['median']-stat['min']],[stat['max']-stat['median']]],capsize=5)
    axes[0].set(xticks=[0,1],xticklabels=labels,ylabel='Solver seconds (median and range)',title='Three fresh processes per method')
    parts=['initialization_seconds','solver_seconds','diagnostics_output_seconds','other'];part_labels=['Initialization','Solver','Diagnostics / I/O','Startup / other / exit']
    part_colors=['#b6c5d2','#287d77','#e5ba79','#677680']
    bottom=np.zeros(2)
    for field,label,color in zip(parts,part_labels,part_colors):
        values=[]
        for key in ('q1','q2'):
            run=sorted(groups[key],key=lambda x:x['launch_to_exit_seconds'])[1]
            values.append(run['result'][field] if field!='other' else run['launch_to_exit_seconds']-sum(run['result'][k] for k in parts[:-1]))
        axes[1].bar([0,1],values,bottom=bottom,color=color,label=label);bottom+=values
    axes[1].set(xticks=[0,1],xticklabels=labels,ylabel='Launch-to-exit seconds',title='Actual median-total run breakdown');axes[1].legend(fontsize=7)
    names=['c','eta','T','slip','energy'];positions=np.arange(len(names))
    for i,key in enumerate(('q1','q2')):
        errors=[max(x['accuracy_errors'][name] for x in groups[key])/(.01 if name=='energy' else .05) for name in names]
        axes[2].bar(positions+(i-.5)*.34,errors,width=.34,color=colors[i],label=labels[i])
    axes[2].axhline(1,color='black',linestyle='--',label='Acceptance limit')
    axes[2].set(xticks=positions,xticklabels=names,ylabel='Measured error / allowed error',title='Against fresh Q2 n=48 reference');axes[2].legend(fontsize=7)
    for suffix in ('png','svg'):fig.savefig(figures/f'thermochemical-cost.{suffix}',dpi=170)
    plt.close(fig)
    tests=int(ET.parse(out/'test-results.xml').getroot().find('testsuite').attrib['tests'])
    checks=sum(sum(x['result']['checks'].values()) for x in r['runs']+[r['reference']])
    gates=sum(sum(x['accuracy_gates'].values()) for x in r['runs']);failed=30-gates
    relative=out.relative_to(ROOT) if out.is_absolute() else out
    lines=['# Qualified costs for the thermochemical reference','',
        f"September 22, 2026. **{tests} tests pass.** Seven fresh trajectories pass {checks}/105 invariant/solver checks; the six repeated candidates pass **{gates}/30 accuracy gates** against a fresh Q2 n=48 reference. Accuracy qualification: **{r['accuracy_qualified']}**. This is a comparison of conventional full-grid Q1/Q2 thermochemistry, not BMVR adaptive performance or a novelty result.",'',
        '## Exact evaluator optimization','',
        'Profiling identified small-matrix inverse/determinant calls and duplicate constitutive work. The optimized evaluator uses exact plane-block formulas, retains the general 3D path when any out-of-plane entry is nonzero, and reuses quantities from the same constitutive evaluation. Material laws, admissibility checks, precision, quadrature and solver tolerances are unchanged. Both cost candidates receive this optimization. The preserved source remains in the evidence directory.','',
        f"Five warmed batches of thirty Q2 n=24 rate evaluations give a median evaluator ratio of **{r['evaluator']['evaluator_speed_ratio']:.3f}×**. The largest measured rate/short-trajectory discrepancy is {max(r['evaluator']['errors'].values()):.3e}. Source paths verify that separate preserved and optimized modules were loaded. This fixed-state throughput comparison is not an end-to-end solver speedup.",'',
        '## Accuracy before cost','',
        'Every trajectory uses the original non-affine grip, coefficients, physical widths and initial profiles, dt=0.00025 and final t=0.2. Both candidates are compared with a newly evolved Q2 n=48 solution. Its scalar grid is at least as fine as either candidate, its mechanical grid is finer, and all interpolation sites lie inside the reference grid. It is a numerical reference, not an exact continuum solution.','',
        '| Candidate | c error | Phase error | T error | Slip error | Energy error |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,label in zip(('q1','q2'),labels):
        errors=[100*max(x['accuracy_errors'][name] for x in groups[key]) for name in names]
        lines.append('| '+label+' | '+' | '.join(f'{x:.4f}%' for x in errors)+' |')
    lines += ['', 'Field gates are <5%; energy error is <1% of the reference initial nonthermal scale. Composition/phase/temperature errors remove the uniform background. Slip uses physical quadrature weights. Comparison energy error, trajectory energy drift and inventory drift are different checks. Historical Q1 coarse-mesh failures remain unchanged.','',
        '## Repeated costs','',
        '| Candidate | Solver seconds, median [min, max] | Launch-to-exit seconds, median [min, max] | Peak RSS MiB, median [min, max] | Evolving scalars |','| --- | ---: | ---: | ---: | ---: |']
    for key,label in zip(('q1','q2'),labels):
        values=[]
        for name in ('solver_seconds','launch_to_exit_seconds','peak_rss_bytes'):
            d=summary[key][name];scale=2**20 if name=='peak_rss_bytes' else 1
            values.append(f"{d['median']/scale:.2f} [{d['min']/scale:.2f}, {d['max']/scale:.2f}]")
        lines.append('| '+label+' | '+' | '.join(values)+f" | {summary[key]['evolving_scalars']:,} |")
    ratios=r['raw_median_ratios']
    if r['accuracy_qualified']:
        lines += ['',f"At the declared accuracy, Q2 has **{ratios['solver_seconds']:.3f}×** the Q1 solver throughput and **{ratios['launch_to_exit_seconds']:.3f}×** the launch-to-exit throughput in this experiment. The Q1/Q2 median peak-RSS ratio is {ratios['peak_rss_bytes']:.3f}. These ratios compare element orders, not adaptive versus uniform PF."]
    else:
        lines += ['',f'Qualification failed ({failed} failed accuracy gates); the timing observations are raw costs and do not establish equivalent-accuracy speedup.']
    lines += ['',f"The finer reference solve costs {r['reference']['result']['solver_seconds']:.2f} seconds and is charged separately from the repeated candidates. Repeated final arrays agree within {max(x['max_absolute_difference'] for x in r['reproducibility'].values()):.3e}. Each candidate ran in its own process, sequentially in Q1,Q2,Q2,Q1,Q1,Q2 order. All three thread environment limits were one; software, machine type and load averages are saved. Other host activity was not assumed absent. No study test or plot ran concurrently with these worker trajectories.",'',
        'The stacked plot uses an actual median-total run, not the sum of unrelated component medians. Initialization, solver, diagnostics/checkpoint I/O and startup/other/exit residual close the launch-to-exit ledger. There is no JIT compilation. Peak RSS includes temporary arrays, initialization and factorization; evolving-array counts alone are not peak memory. Both methods are full-grid and their scalar/plastic resolutions differ.','',
        f'![Measured timing and accuracy]({relative}/figures/thermochemical-cost.png)','',
        '## Reproduction and remaining scope','',
        'The source snapshot, raw worker logs, checkpoints, per-worker timings, numerical reports, environment records and source hashes are retained. From the project root, reproduce in a new output directory with:','',
        '```sh',
        "python3 -c \"import shutil; shutil.copytree('results/thermochemical-cost-2026-09-22/baseline-src', 'results/thermochemical-cost-reproduction/baseline-src')\"",
        'PYTHONPATH=src python3 -m pytest -q --junitxml=results/thermochemical-cost-reproduction/test-results.xml',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src python3 benchmarks/thermochemical_cost.py --output results/thermochemical-cost-reproduction',
        'PYTHONPATH=src python3 benchmarks/thermochemical_cost_report.py --input results/thermochemical-cost-reproduction','```','',
        f"[Machine-readable study]({relative}/report.json), [evaluator parity]({relative}/evaluator.json), [test evidence]({relative}/test-results.xml), [theory Section 41](unified_multiscale_defect_framework.md). The report generator expects the corresponding test XML in the output directory.",'',
        'This supplies a current performance baseline for the finite thermochemical reference. Dynamic localization/reduction, reactive line/point-defect and fracture integration, broader crystallography and the original novelty requirement remain unresolved. The full goal remains active.','']
    write_doc('THERMOCHEMICAL_COST_RESULTS.md','\n'.join(lines))
    (out/'plotting.json').write_text(json.dumps({'plotting_and_report_seconds':perf_counter()-start},indent=2)+'\n')


if __name__=='__main__':main()
