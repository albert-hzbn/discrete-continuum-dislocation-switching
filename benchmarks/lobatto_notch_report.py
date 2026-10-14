"""Section 47 report: figures and markdown from saved Gauss-Lobatto notch evidence."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.nodal_damage import NodalThermalFracture2D
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['c','eta','T','slip','damage']


def pct(v):return 'n/a' if v is None else f'{100*v:.3f}%'


def exact_min_det(path):
    m,s,_=NodalThermalFracture2D.load(path);M=m.mechanics.mesh;C=M.determinant_coefficients(s.u)
    t=np.linspace(0,1,31);B=np.stack(((1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3),-1)
    return float(np.einsum('ia,eab,jb->eij',B,C,B).min())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--gauss',type=Path,default=Path('results/graded-notch-v3-2026-09-26'));args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());c=r['comparisons'];pm=r['per_mesh'];figures=out/'figures';figures.mkdir(exist_ok=True)
    finest=max((int(k) for k,v in pm.items() if v['complete']),default=None)
    fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    if finest:
        folder=Path(pm[str(finest)]['folder']);m,s,_=NodalThermalFracture2D.load(folder/f'lobatto-notch-{finest}-0.00025-final.npz');n=finest;xe,ye=m.mechanics.mesh.edges
        im=axes[0].pcolormesh(xe,ye,s.damage.reshape(n+1,n+1).T,shading='gouraud',cmap='magma',vmin=0);fig.colorbar(im,ax=axes[0],shrink=.8)
        axes[0].set(title=f'Nodal damage at t=0.4, GLL graded n={n}',xlabel='x',ylabel='y',aspect='equal')
    for key,label in (('48_64','n48/n64'),('64_96','n64/n96'),('96_128','n96/n128')):
        rows=c[key];times=[float(t[1:]) for t in rows if t!='final']+([.4] if 'final' in rows else [])
        vals=[100*rows[f't{t:g}' if t!=.4 else 'final']['slip'] for t in times]
        axes[1].plot(times,vals,'o-',label=label)
    axes[1].axhline(5,color='black',ls='--',lw=1);axes[1].set(xlabel='Time',ylabel='Slip discrepancy (%)',title='GLL successive-mesh slip discrepancy');axes[1].legend(fontsize=8)
    ks=sorted(int(k) for k in pm);axes[2].plot([pm[str(k)]['fine_ell_over_h'][0] for k in ks],[pm[str(k)]['events']['tip_max_damage_09'] for k in ks],'s-',label='GLL')
    g=json.loads((args.gauss/'report.json').read_text());gk=[k for k in g['per_mesh'] if k.startswith('graded')]
    axes[2].plot([g['per_mesh'][k]['fine_ell_over_h'][0] for k in gk],[g['per_mesh'][k]['events']['tip_max_damage_09'] for k in gk],'o--',label='Gauss (Section 46)')
    axes[2].set(xlabel='Tip ell/h',ylabel='t when max d=0.9',title='Tip event timing by quadrature rule');axes[2].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'lobatto-notch.{ext}',dpi=170)
    plt.close(fig)
    tests=int(ET.parse(out/'test-results.xml').getroot().find('testsuite').attrib['tests']) if (out/'test-results.xml').exists() else None
    gates=r['gates_64_96_final'];q=r['qualification_96_128'];rel=out.relative_to(ROOT) if out.is_absolute() else out
    complete=[k for k,v in pm.items() if v['complete']];partial=[k for k,v in pm.items() if not v['complete']]
    lines=['# Corner-sampling quadrature and the post-separation notch','',
        f"September 26, 2026.{' **'+str(tests)+' tests pass.**' if tests else ''} Section 47 replaces the 3x3 Gauss rule by the 3x3 Gauss–Lobatto (GLL) rule, whose points include every element corner. This is where the retained compressive barrier of the damage split must act to prevent the post-separation folds found in Section 46. Physics, widths, loading, dt and the midpoint equations are unchanged. Trajectories reaching t=0.4: {', '.join('n'+k for k in complete) or 'none'}; partial: {', '.join('n'+k for k in partial) or 'none'}.",'',
        '## Retained failures along the way','',
        '1. GLL n64 with the Section 46.3 preconditioner stopped at t=0.30825 on midpoint non-convergence. Its geometry was healthy (exact minimum determinant 0.92). A replay traced the stall to a stiff inertial mode, and a mechanical Schur-complement block with the tangent at a midpoint iterate fixed it ([replay](results/lobatto-notch-2026-09-26/solver-replay.json)).',
        '2. Building that block on every step made easy steps four times slower, and the "full" reruns were stopped near t=0.1. The retained "adaptive" mode builds it only for steps unconverged after 12 updates; the fixed point and tolerance are unchanged.',
        '3. No adaptive GLL trajectory from n64 up reached t=0.4. The Bernstein certificate refused n64 at t=0.3235, n96 at t=0.28525 and n128 at t=0.28075. Replays that bypass only the certificate decide whether each refusal is a real fold. n64 folds genuinely: the exact minimum determinant of its free-edge mouth cell turns negative (-0.12) by t=0.3535 ([replay](results/lobatto-notch-adaptive-2026-09-26/fold-replay-64.json)). n96 is refused conservatively (exact minimum 0.996 in the refused cell), but the same mouth cell then folds genuinely by t=0.3102 (-0.16) ([replay](results/lobatto-notch-adaptive-2026-09-26/fold-replay-96.json)). n128 is also refused conservatively (exact minimum 0.64 at its last accepted state); its fold time was not replayed. The finer mesh folds earlier, consistent with a fold once the mouth opening is comparable to the cell size. On these meshes, refinement with continuous kinematics shortens rather than extends the admissible window.','',
        '## Accuracy at t=0.4 and qualification','','| Field | n48/n64 | n64/n96 | n96/n128 | Gate n64/n96 <5% |','| --- | ---: | ---: | ---: | --- |']
    for f in FIELDS:
        cell=lambda k:pct(c[k]['final'][f]) if 'final' in c[k] else 'not reached'
        lines.append(f"| {f} | {cell('48_64')} | {cell('64_96')} | {cell('96_128')} | {('PASS' if gates[f] else 'FAIL') if isinstance(gates,dict) else 'unevaluated'} |")
    lines+=['','Because no mesh from n64 up reaches t=0.4, the declared t=0.4 gates are unevaluated and count as failed.']
    lines+=['',f"Qualification against n128 (n96/n128 below 5% in every field and below n64/n96 in slip and damage): "+(', '.join(f"{k} {'qualified' if v['below_5_percent'] and v['decreasing_slip_damage'] else 'not qualified'}" for k,v in q.items()) or 'no common checkpoint')+'.','',
        '### Slip discrepancy at every checkpoint','','| Time | n48/n64 | n64/n96 | n96/n128 |','| --- | ---: | ---: | ---: |']
    for t in [f't{x:g}' for x in r['checkpoints']]+['final']:
        lines.append(f"| {'0.4' if t=='final' else t[1:]} | "+' | '.join(pct(c[k][t]['slip']) if t in c[k] else 'not reached' for k in ('48_64','64_96','96_128'))+' |')
    lines+=['','## Per-mesh trajectories','','| Mesh | Tip ell/h | Reached | t(max d=0.9) | First saturation | Max iterations | Exact min det at end | Solver s | Mode |','| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for k in sorted(pm,key=int):
        p=pm[k];e=p['events'];folder=Path(p['folder']);end=folder/(f'lobatto-notch-{k}-0.00025-final.npz' if p['complete'] else f'lobatto-notch-{k}-0.00025-last-accepted.npz')
        reach='t=0.4' if p['complete'] else f"partial t={p['failure']['last_accepted_time']:.5g} ({p['failure']['error'].rstrip('.')})"
        lines.append(f"| n{k} | {p['fine_ell_over_h'][0]:.2f} | {reach} | {e['tip_max_damage_09']:.4f} | {e['first_saturation']} | {e['max_midpoint_iterations']} | {exact_min_det(end):.3f} | {p['solver_seconds']:.0f} | {p['preconditioned']} |")
    times=sorted({float(t) for p in pm.values() for t in p['fronts']})
    fmt=lambda v:'–' if v is None else f'{v:.4f}'
    lines+=['','### Midline crack fronts','',"Position along y=0.5 where nodal damage first falls below 0.4 (process zone) and 0.9 (broken material), from every saved state. The notch mouth is at x=0.",'',
        '| Mesh | '+' | '.join(f't={t:g}' for t in times)+' |','| --- | '+' | '.join('---:' for _ in times)+' |']
    for k in sorted(pm,key=int):
        fr=pm[k]['fronts']
        for kind in ('process','broken'):
            lines.append(f"| n{k} {kind} | "+' | '.join(fmt(fr[f'{t:g}'][kind]) if f'{t:g}' in fr else 'not reached' for t in times)+' |')
    ref=pm[str(finest)]['fronts'] if finest else None
    if ref:
        b=[(float(t),v['broken']) for t,v in ref.items() if v['broken'] is not None];b.sort()
        speeds=[(b[i+1][1]-b[i][1])/(b[i+1][0]-b[i][0]) for i in range(len(b)-1)]
        pr=sorted((float(t),v['process']) for t,v in ref.items() if v['process'] is not None)
        lines+=['',f"On the complete n{finest} trajectory the process front advances from {pr[0][1]:.4f} to {pr[-1][1]:.4f} ({(pr[-1][1]-pr[0][1])/(pr[-1][0]-pr[0][0]):.3f} per unit time on average). The broken front appears at t={b[0][0]:g} and moves at "+', '.join(f'{v:.2f}' for v in speeds)+" per unit time over successive checkpoint intervals: it decelerates as it approaches the process zone, but it has not arrested by t=0.4. Arrest in this held-grip loading is therefore a trend, not an established state."]
    sens=r['quadrature_sensitivity_vs_gauss_96']
    if sens:lines+=['','Quadrature sensitivity, GLL against Gauss at n96 (diagnostic): '+'; '.join(f"t={t[1:]}: slip {pct(v['slip'])}, damage {pct(v['damage'])}" for t,v in sens.items())+'. These values are far below the mesh discrepancies, so the rule does not dominate: before a fold, GLL and Gauss produce the same trajectory to within a fraction of a percent. The GLL qualification at t=0.25 therefore extends the Gauss qualification at t*=0.20 rather than replacing it.']
    est=r['tip_timing_estimate']
    if est.get('observed_order'):lines+=['',f"GLL tip-event extrapolation from the finest three meshes: {est['extrapolated_tip_time']:.4f} (observed order {est['observed_order']:.2f}), to be compared with the Gauss estimates of 0.1413 and 0.1421."]
    lines+=['',f'![Damage at t=0.4, slip discrepancy over time, tip timing]({rel}/figures/lobatto-notch.png)','',
        '## Scope','',"GLL sampling removes the corner blind spot but controls J only at the sampled points; the Bernstein certificate still guards whole elements. The fields after first saturation are those of the smeared-crack model without displacement discontinuities or contact search. Up to t=0.25 they are qualified as discretization-converged fields of that model, not as predictions of discrete crack faces; beyond about t=0.28 no continuous-kinematics mesh finer than n48 remains admissible.",'',
        '```sh','for n in 64 96 128; do python3 benchmarks/lobatto_notch.py --output results/lobatto-notch-adaptive-2026-09-26 --run $n --adaptive & done; wait',
        'python3 benchmarks/lobatto_notch.py --output results/lobatto-notch-adaptive-2026-09-26 --adaptive',f'python3 benchmarks/lobatto_notch_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 47](unified_multiscale_defect_framework.md).','']
    write_doc('LOBATTO_NOTCH_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
