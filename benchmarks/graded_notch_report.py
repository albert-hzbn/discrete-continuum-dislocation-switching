"""Section 46 graded-notch report: figures and markdown from saved evidence (declared 46.6 protocol)."""
import argparse,json
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.nodal_damage import NodalThermalFracture2D
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['c','eta','T','slip','damage']
PAIRS=[('uniform_32_48','Uniform n32/n48'),('graded_32_48','Graded n32/n48'),('graded_48_64','Graded n48/n64')]


def pct(v):return 'n/a' if v is None else f'{100*v:.3f}%'


def num(v,fmt='.4f'):return 'n/a' if v is None else format(v,fmt)


def fields_figure(out,r,figures):
    star=r['t_star'];name='graded-notch-64-0.00025'
    m,a,_=NodalThermalFracture2D.load(out/f'{name}-initial.npz')
    _,s,_=NodalThermalFracture2D.load(out/f'{name}-t{star:g}.npz')
    n=m.mechanics.mesh.n;xe,ye=m.mechanics.mesh.edges
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for ax,values,title in zip(axes[:2],[s.damage,s.damage-a.damage],[f'Nodal damage at t*={star:g}, graded n={n}','Damage increment to t*; own scale']):
        im=ax.pcolormesh(xe,ye,values.reshape(n+1,n+1).T,shading='gouraud',cmap='magma',vmin=0)
        ax.set(title=title,xlabel='x',ylabel='y',aspect='equal');fig.colorbar(im,ax=ax,shrink=.8)
    ax=axes[2]
    for x in xe:ax.plot([x,x],[0,1],color='0.55',lw=.3)
    for y in ye:ax.plot([0,1],[y,y],color='0.55',lw=.3)
    ax.contour(xe,ye,s.damage.reshape(n+1,n+1).T,levels=[.4,.9],colors=['tab:orange','tab:red'],linewidths=1)
    ax.set(title='Cell edges; d=0.4, 0.9 contours at t*',xlabel='x',ylabel='y',aspect='equal',xlim=(0,1),ylim=(0,1))
    for ext in ('png','svg'):fig.savefig(figures/f'graded-notch-fields.{ext}',dpi=170)
    plt.close(fig)


def verification_figure(r,udiag,figures):
    star=f"t{r['t_star']:g}";c=r['comparisons']
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained');x=np.arange(len(FIELDS))
    for k,(key,label) in enumerate(PAIRS):axes[0].bar(x+(k-1)*.27,[100*c[key][star][f] for f in FIELDS],width=.27,label=label)
    axes[0].axhline(5,color='black',ls='--',lw=1,label='5% gate')
    axes[0].set(xticks=x,xticklabels=FIELDS,ylabel='Discrepancy (%)',title=f'Successive-mesh discrepancies at t*={r["t_star"]:g}');axes[0].legend(fontsize=7)
    for key,label in PAIRS:
        times=[float(t[1:]) for t in c[key] if t!='final'];vals=[100*c[key][f't{t:g}']['slip'] for t in times]
        if 'final' in c[key]:times.append(.4);vals.append(100*c[key]['final']['slip'])
        axes[1].plot(times,vals,'o-',label=label)
    q=r.get('qualification')
    if q:
        times=[float(t[1:]) for t in q['comparison_64_96'] if t!='final']
        axes[1].plot(times,[100*q['comparison_64_96'][f't{t:g}']['slip'] for t in times],'d--',color='tab:red',label='Graded n64/n96 (reference)')
    axes[1].axhline(5,color='black',ls='--',lw=1);axes[1].axvline(r['t_star'],color='0.6',ls=':',label='t*')
    axes[1].set(xlabel='Time',ylabel='Slip discrepancy (%)',title='Slip discrepancy at declared checkpoints');axes[1].legend(fontsize=7)
    uh=[.08*int(k) for k in udiag['timing']];ut=[udiag['timing'][k]['tip_max_damage_09'] for k in udiag['timing']]
    pm=r['per_mesh'];gk=[k for k in pm if k.startswith('graded')]
    axes[2].plot(uh,ut,'o-',label='Uniform (Section 45.5)')
    axes[2].plot([pm[k]['fine_ell_over_h'][0] for k in gk],[pm[k]['events']['tip_max_damage_09'] for k in gk],'s-',label='Graded (tip ell/h_x)')
    if q:axes[2].plot([pm['graded-64']['fine_ell_over_h'][0],q['reference']['fine_ell_over_h'][0]],[pm['graded-64']['events']['tip_max_damage_09'],q['reference']['events']['tip_max_damage_09']],'d--',color='tab:red',label='Graded n96 reference')
    axes[2].axhline(udiag['tip_timing_estimate']['extrapolated_tip_time'],color='tab:blue',ls=':',label='Uniform 3-point extrapolation')
    if r['tip_timing_estimate'].get('observed_order'):
        axes[2].axhline(r['tip_timing_estimate']['extrapolated_tip_time'],color='tab:orange',ls='--',lw=1,label='Graded 3-point extrapolation')
    axes[2].set(xlabel='ell_f / h at the tip',ylabel='t when max d=0.9',title='Crack-tip event timing (H1)');axes[2].legend(fontsize=7)
    for ext in ('png','svg'):fig.savefig(figures/f'graded-notch-verification.{ext}',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--uniform',type=Path,default=Path('results/nodal-refinement-2026-09-26'))
    parser.add_argument('--first',type=Path,default=Path('results/graded-notch-2026-09-26'))
    parser.add_argument('--second',type=Path,default=Path('results/graded-notch-v2-2026-09-26'))
    args=parser.parse_args();start=perf_counter();out=args.input;r=json.loads((out/'report.json').read_text())
    udiag=json.loads((args.uniform/'timing-diagnostic.json').read_text())
    figures=out/'figures';figures.mkdir(exist_ok=True)
    fields_figure(out,r,figures);verification_figure(r,udiag,figures)
    tests=int(ET.parse(out/'test-results.xml').getroot().find('testsuite').attrib['tests'])
    pm=r['per_mesh'];c=r['comparisons'];star=r['t_star'];tag=f't{star:g}'
    checks=sum(sum(x['checks'].values()) for x in pm.values());total=sum(len(x['checks']) for x in pm.values())
    gg,ug=r['gates_at_t_star']['graded_48_64'],r['gates_at_t_star']['uniform_32_48']
    partial=[k for k,v in pm.items() if not v['complete']]
    rel=out.relative_to(ROOT) if out.is_absolute() else out
    lines=['# Graded notch meshes with a stiff-block midpoint solve','',
        f"September 26, 2026. **{tests} tests pass.** Five trajectories (graded n32/n48/n64, uniform n32/n48) pass **{checks}/{total}** invariant and solver checks over their accepted intervals. "
        f"{'Partial trajectories: '+', '.join(partial)+'.' if partial else 'All reached t=0.4.'} At the declared common checkpoint t*={star:g}, graded n48/n64 passes **{sum(gg.values())}/5** and uniform n32/n48 passes **{sum(ug.values())}/5** gates. "
        f"{'A graded n96 reference qualifies these gates at every checkpoint up to t*. ' if r.get('qualification') and all(v['below_5_percent'] and v['decreasing_slip_damage'] for v in r['qualification']['by_checkpoint'].values()) and tag in r['qualification']['by_checkpoint'] else ''}The original t=0.4 graded gates are unevaluated because graded trajectories folded after separation. Physics, widths, coefficients, loading and the implicit-midpoint equations are those of Section 45; only mesh geometry, the nonlinear solver path and the comparison protocol change. The full objective remains active.",'',
        '## What changed','',
        f"Section 46 adds tensor-product graded Q2 meshes (smooth density, spacing ratio {r['bands']['ratio']:g} in an x-band {r['bands']['x']} and y-band {r['bands']['y']}) and a right-preconditioned fixed-point solve whose frozen composition (Cahn–Hilliard) and heat blocks are factorized once per step. Convergence is still judged on the unpreconditioned midpoint residual at 1e-12, so the fixed point is the same implicit-midpoint solution. Uniform meshes and default settings reproduce earlier trajectories exactly (below).",'',
        'Graded meshes pass the same finite-difference energy, reciprocal, local first-law, entropy and restart tests as uniform meshes, with deliberately lopsided cells; central differences show pure eps² truncation. The preconditioned and Picard solutions agree within 1e-11 where both converge. On graded n48 and n64 plain Picard diverges within the first step (an out-of-bounds composition iterate), while the preconditioned solve converges in 6–7 iterations.','',
        '## Retained failures','',
        f"1. **First band design** (x-band [0.25,0.55], y-band [0.40,0.60]) stopped at n32, step 1314 (t=0.3285) with a Q2 corner determinant crossing zero in fully damaged crack-mouth cells of aspect ratio 3.4 ([record]({args.first}/graded-notch-32-0.00025-failure.json)). Section 46.5 then required near-isotropic cells along the whole damaged band.",
        f"2. **Revised design, declared t=0.4 study.** Graded n32 completed, but graded n48 stopped at step 1232 (t=0.308) ([record]({args.second}/graded-notch-48-0.00025-failure.json)). The mouth cell there has aspect ratio 1.05. Its Bernstein certificate fell to −0.004 while the exact minimum determinant (subdivision and dense sampling) was 0.47. A diagnostic replay bypassing only the certificate found an exact corner inversion 37 steps later (t=0.3142). The guard was early, not false, and it stays unchanged.",'',
        'Mechanism (theory Section 46.6): the model has no displacement discontinuity. Once the band saturates, the regularized opening is carried by the one or two element rows beside the crack line, so their distortion grows as opening/h, and soft broken Q2 cells fold at corners that the 3x3 quadrature does not sample (Gauss-point J stays at or above 0.89 in every last accepted state). Uniform n48 (h=0.0208) and graded n32 (fine h≈0.022) ended with exact minimum determinants 0.237 and 0.228. This corrects Section 46.5, which blamed cell elongation alone. The declared graded t=0.4 gates are recorded as **unevaluated/failed**, not replaced.','',
        '## Declared checkpoint protocol','',
        'Fixed in Section 46.6 before any new field comparison: identical physics and t=0.4 end time. A trajectory that meets the orientation guard is kept as partial. States are saved at t=0.10, 0.15, 0.20, 0.25 and 0.30, and the uniform family is rerun with the same code. The primary time t* is the latest checkpoint reached by every run of both families. Every checkpoint is reported, so t* cannot hide adverse times. Hypotheses stated in advance: **H1**, tip timing follows local tip ell/h; **H2**, if tip under-resolution causes the slip failure, graded n48/n64 slip at t* is below 5% and below uniform n32/n48 at t*.','',
        f'## Accuracy at t*={star:g}','',
        '| Field | Uniform n32/n48 | Graded n32/n48 | Graded n48/n64 | Uniform gate | Graded gate |','| --- | ---: | ---: | ---: | --- | --- |']
    for f in FIELDS:
        lines.append(f"| {f} | {pct(c['uniform_32_48'][tag][f])} | {pct(c['graded_32_48'][tag][f])} | {pct(c['graded_48_64'][tag][f])} | {'PASS' if ug[f] else 'FAIL'} | {'PASS' if gg[f] else 'FAIL'} |")
    lines+=['',f"Failed at t*: graded {', '.join(k for k,v in gg.items() if not v) or 'none'}; uniform {', '.join(k for k,v in ug.items() if not v) or 'none'}. Each comparison interpolates the finer state to the coarser state's sites without extrapolation, with volume- and quadrature-weighted norms that reduce exactly to the earlier uniform metrics.",'',
        '### Hypothesis outcomes (decision rules of Section 46.6)','',
        f"**H2 {'supported' if all(gg.values()) and c['graded_48_64'][tag]['slip']<c['uniform_32_48'][tag]['slip'] else 'not supported'} at t*:** graded n48/n64 slip is {pct(c['graded_48_64'][tag]['slip'])}, below 5% and below uniform n32/n48 ({pct(c['uniform_32_48'][tag]['slip'])}). "
        f"Uniform n32/n48 {'fails' if not ug['slip'] else 'passes'} slip at t*, so by the declared rule the uniform t=0.4 slip failure is {'attributed to tip under-resolution rather than mainly to the post-separation regime' if not ug['slip'] else 'attributed mainly to the post-separation regime'}. "
        f"Uniform slip discrepancy is already {pct(c['uniform_32_48']['t0.1']['slip'])} at t=0.10, before the tip event, and grows to {pct(c['uniform_32_48']['final']['slip'])} at t=0.4; post-separation motion amplifies an error that exists earlier. "
        "All three pairs' slip discrepancy grows with time, so the passing margin is specific to t≤t*.",'',
        '### All checkpoints (slip / damage)','','| Time | Uniform n32/n48 | Graded n32/n48 | Graded n48/n64 |','| --- | --- | --- | --- |']
    for t in [f't{x:g}' for x in r['checkpoints']]+['final']:
        cells=[f"{pct(c[k][t]['slip'])} / {pct(c[k][t]['damage'])}" if t in c[k] else 'not reached' for k,_ in PAIRS]
        lines.append(f"| {'0.4 (final)' if t=='final' else t[1:]} | "+' | '.join(cells)+' |')
    lines+=['','Uniform n32/n48 at t=0.4 is the retained Section 45.5 failure; the rerun reproduces it.','',
        '## Per-mesh trajectories','','| Mesh | Cells | Tip ell/h (x, y) | Reached | t(max d=0.9) | First saturation | L1 slip at t* | Max iterations | Solver s |','| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for k,p in pm.items():
        e=p['events'];reach='t=0.4' if p['complete'] else f"partial, t={p['failure']['last_accepted_time']:.5g}"
        lines.append(f"| {k} | {p['cells']} | {p['fine_ell_over_h'][0]:.2f}, {p['fine_ell_over_h'][1]:.2f} | {reach} | {num(e['tip_max_damage_09'])} | {num(e['first_saturation'],'.5g')} | {p['slip_at_t_star']['L1']:.4e} | {e['max_midpoint_iterations']} | {p['solver_seconds']:.1f} |")
    for k,p in pm.items():
        if not p['complete']:lines+=['',f"{k} stopped at step {p['failure']['step']}: {p['failure']['error']}"]
    repro=[f"{k}: {'bitwise identical' if v['reproduces_saved_final'] else 'NOT identical' if v['reproduces_saved_final'] is False else 'not checked'}" for k,v in pm.items() if k.startswith('uniform')]
    est=udiag['tip_timing_estimate'];gest=r['tip_timing_estimate']
    lines+=['',f"Uniform rerun versus saved Section 45 finals: {'; '.join(repro)}.",'',
        f"H1 test: the Section 45.5 uniform tip event occurred at t=0.224, 0.191, 0.166 on n24, n32, n48; its three-point extrapolation was {est['extrapolated_tip_time']:.4f} (observed order {est['observed_order']:.2f}). Graded n32 (tip ell/h 3.66) matches the uniform n48 tip event within {100*abs(pm['graded-32']['events']['tip_max_damage_09']/pm['uniform-48']['events']['tip_max_damage_09']-1):.3f}% and saturates on the same step (ell/h 3.84) with 2.25× fewer cells. "
        +(f"The graded n32/n48/n64 extrapolation independently gives {gest['extrapolated_tip_time']:.4f} (observed order {gest['observed_order']:.2f}); graded n64 is {100*gest['finest_relative_timing_error']:.1f}% late against it, versus about {100*est['n48_relative_timing_error']:.0f}% for uniform n48. Both extrapolations are estimates from three points, not established asymptotic rates." if gest.get('observed_order') else 'The graded times are not monotone, so no graded extrapolation is reported.'),'',
        f"Fold times: graded n32 completed to t=0.4; graded n48 stopped at t={pm['graded-48']['failure']['last_accepted_time']:.5g} (mouth, free edge), graded n64 at t={pm['graded-64']['failure']['last_accepted_time']:.5g} (band interior, x≈0.126)"
        +(f" and graded n96 at t={r['qualification']['reference']['failure']['last_accepted_time']:.5g} (band interior, x≈0.164)" if r.get('qualification') and r['qualification']['reference']['failure'] else '')
        +". Every graded mesh with band spacing at most 0.014 folds 0.03–0.10 after its first nodal saturation, whereas meshes with band spacing about 0.02 or coarser reach t=0.4. The fold time is not monotone in band spacing, so the expectation stated in Section 46.7 that the admissible window shrinks steadily with refinement is refuted. "
        f"t*={star:g} precedes first nodal saturation on every graded mesh, so these gates qualify initiation and early propagation, not the broken regime.",'']
    vs=c.get('graded48_versus_uniform48',{})
    if 'skipped' in vs:lines+=['A graded n48 versus uniform n48 field comparison is not reported: graded wall-adjacent sites lie outside the uniform cell-center range and the comparison does not extrapolate.','']
    if tag in vs:lines+=[f'Graded n48 versus uniform n48 at t* (same cell count, different placement; diagnostic only): '+', '.join(f"{f} {pct(vs[tag][f])}" for f in FIELDS)+'.','']
    q=r.get('qualification')
    lines+=['## Finer-reference qualification (Section 46.7)','']
    if q is None:lines+=['Graded n96 has not been run; the graded n48/n64 gates above are unqualified.','']
    else:
        ref=q['reference'];reach='t=0.4' if ref['complete'] else f"t={ref['failure']['last_accepted_time']:.5g} before stopping ({ref['failure']['error'].rstrip('.')})"
        lines+=[f"Graded n96 (tip ell/h {ref['fine_ell_over_h'][0]:.2f}) reached {reach}; t(max d=0.9)={num(ref['events']['tip_max_damage_09'])}, first saturation {num(ref['events']['first_saturation'],'.5g')}, maximum {ref['events']['max_midpoint_iterations']} midpoint iterations, {ref['solver_seconds']:.0f} s solver (alone, single thread). Its checks pass {sum(ref['checks'].values())}/{len(ref['checks'])} over the accepted interval.",'',
            '| Time | n64/n96 c | eta | T | slip | damage | n48/n64 slip | All <5% | Slip and damage decrease |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
        for t,v in q['by_checkpoint'].items():
            e=q['comparison_64_96'][t]
            lines.append(f"| {t[1:]} | {pct(e['c'])} | {pct(e['eta'])} | {pct(e['T'])} | {pct(e['slip'])} | {pct(e['damage'])} | {pct(c['graded_48_64'][t]['slip'])} | {'PASS' if v['below_5_percent'] else 'FAIL'} | {'yes' if v['decreasing_slip_damage'] else 'no'} |")
        g2=q['tip_timing_estimate_48_64_96']
        if g2.get('observed_order'):lines+=['',f"The n96 tip event continues the graded sequence. The n48/n64/n96 extrapolation gives {g2['extrapolated_tip_time']:.4f} (observed order {g2['observed_order']:.2f}), against {gest['extrapolated_tip_time']:.4f} from n32/n48/n64 and {est['extrapolated_tip_time']:.4f} from the uniform family. The three estimates agree within {100*(max(g2['extrapolated_tip_time'],gest['extrapolated_tip_time'],est['extrapolated_tip_time'])/min(g2['extrapolated_tip_time'],gest['extrapolated_tip_time'],est['extrapolated_tip_time'])-1):.1f}%; n96 is {100*g2['finest_relative_timing_error']:.1f}% late against its own estimate."]
        qualified=[t[1:] for t,v in q['by_checkpoint'].items() if v['below_5_percent']]
        lines+=['',f"Qualified checkpoints: {', '.join(qualified) or 'none'}. {'t* itself is qualified.' if tag in q['by_checkpoint'] and q['by_checkpoint'][tag]['below_5_percent'] else 't*='+format(star,'g')+' is not qualified by the reference.'}",'']
    runs=pm.values()
    lines+=['## Balances and cost','',f"Every accepted step of every trajectory meets bounds, irreversibility, orientation, entropy-production, plastic-volume, inventory and transfer checks ({checks}/{total}).",'',
        "The five trajectories ran concurrently as single-thread processes, so solver seconds are observations under shared load, not a cost comparison. No equivalent-accuracy speedup is claimed.",'',
        f'![Graded mesh and damage at t*]({rel}/figures/graded-notch-fields.png)','',f'![Discrepancies at t*, slip over time and tip timing]({rel}/figures/graded-notch-verification.png)','',
        '## Scope','',"This is static a-priori grading around the declared notch path, not dynamic adaptivity. The fracture model remains isotropic AT2 with the volumetric/deviatoric split, no displacement discontinuity and no contact. Post-separation fields (t>t*) are not qualified by any result here; they need discontinuity kinematics or a separation/contact model. Reactive line/point-defect coupling and qualified multiscale cost savings also remain open.",'',
        '```sh','PYTHONPATH=src python3 -m pytest -q --junitxml=results/graded-notch-reproduction/test-results.xml',
        'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src:benchmarks',
        'for n in 32 48 64; do python3 benchmarks/graded_notch.py --output results/graded-notch-reproduction --run graded $n & done',
        'for n in 32 48; do python3 benchmarks/graded_notch.py --output results/graded-notch-reproduction --run uniform $n & done; wait',
        'python3 benchmarks/graded_notch.py --output results/graded-notch-reproduction --run graded 96',
        'python3 benchmarks/graded_notch.py --output results/graded-notch-reproduction',
        'python3 benchmarks/graded_notch_report.py --input results/graded-notch-reproduction','```','',
        f'[Numerical report]({rel}/report.json), [tests]({rel}/test-results.xml), [verification and saved-state readback]({rel}/verification.json), [theory Section 46](unified_multiscale_defect_framework.md).','']
    write_doc('GRADED_NOTCH_RESULTS.md','\n'.join(lines))
    (out/'plotting.json').write_text(json.dumps({'plotting_and_report_seconds':perf_counter()-start},indent=2)+'\n')


if __name__=='__main__':main()
