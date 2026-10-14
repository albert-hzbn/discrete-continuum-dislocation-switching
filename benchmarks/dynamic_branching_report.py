"""Sections 66 and 81 report: dynamic crack branching, isothermal and with fracture heat, from saved evidence."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
mark=lambda b:'PASS' if b else 'FAIL'
LOADS=(.03,.06,.09)
C_R=.93  # Rayleigh speed, as in dynamic_branching.py


def posthoc(out,rel):
    """Labelled post hoc runs (Section 66.3 to 66.5): stronger loads, and the 2 x 1 wide domain."""
    runs=[(d,json.loads(f.read_text())) for d in ('post-hoc-strong','post-hoc-wide') for f in sorted((out/d).glob('sigma-*.json'))]
    if not runs:return []
    fig,axes=plt.subplots(1,len(runs),figsize=(6*len(runs),4.2),layout='constrained',squeeze=False)
    for ax,(d,x) in zip(axes[0],runs):
        w=x.get('width',1);n=200;q=np.load(out/d/f"sigma-{x['sigma']:g}.npz")['final'].reshape(n*w+1,n+1)
        ax.imshow(q.T,origin='lower',extent=(0,w,0,1),cmap='magma',vmin=0,vmax=1);ax.set(title=f"post hoc σ={x['sigma']:g}, {w}×1, t={x['end_time']:.2f}",xticks=[],yticks=[])
    for ext in ('png','svg'):fig.savefig(out/'figures'/f'dynamic-branching-post-hoc.{ext}',dpi=140)
    plt.close(fig)
    lines=['## Post hoc runs','','These runs were chosen after seeing the declared outcome and are labelled post hoc; theory Section 66.3 to 66.5 interprets them.','',
        '| Run | Domain | End time | Bands crossing the counting lines (max, while the tip is short of the stop) | Wide-domain gate K4 | Energy drift |','| --- | --- | ---: | --- | --- | ---: |']
    for d,x in runs:
        k4=x.get('K4_branch_away_from_edge');lines.append(f"| σ={x['sigma']:g} | {x.get('width',1)} × 1 | {x['end_time']:.3f} | {', '.join(f'{a}: {b}' for a,b in x['max_bands_before_edge'].items())} | {'–' if k4 is None else mark(k4)} | {x['energy_drift_fraction']:.1e} |")
    for d,x in runs:
        b=x.get('first_early_branch')
        if b:lines+=['',f"In the {x['width']} × 1 domain at σ={x['sigma']:g} the crack first splits into two bands at t={b['time']:.2f} with the tip at x={b['tip']:.2f}, at least {x['width']-b['tip']:.2f} from the free edge. The square-domain split at the same time and place therefore comes from branching, not from the edge. Later rows show secondary branching, with up to {max(max(r.values()) for r in [x['max_bands_before_edge']])} bands on one line."]
    return lines+['',f'![Post hoc final damage]({rel}/figures/dynamic-branching-post-hoc.png)','']


def thermal(out,rel):
    """Section 81: the 2 x 1 wide run with fracture heat and thermal softening, compared with the isothermal wide run."""
    path=out/'post-hoc-thermal'/'sigma-0.09.json'
    if not path.exists():return []
    x=json.loads(path.read_text());iso=json.loads((out/'post-hoc-wide'/'sigma-0.09.json').read_text())
    z=np.load(out/'post-hoc-thermal'/'sigma-0.09.npz');n=200;nx=2*n;T=z['T'].reshape(nx,n);q=z['final'].reshape(nx+1,n+1)
    qc=(q[:-1,:-1]+q[1:,:-1]+q[:-1,1:]+q[1:,1:])/4;xc=(np.arange(nx)+.5)/n;band=qc>=.9
    peak={f'{a:g}–{b:g}':float(T[band&((xc>=a)&(xc<b))[:,None]].max()-x['min_T']) for a,b in ((0,.5),(.5,.9),(.9,2))}
    gates=dict(H1_energy=x['energy_drift_fraction']<=.05,H2_bounds=x['min_increment']>=0 and x['q_bounds'][0]>=0 and x['q_bounds'][1]<=1 and x['min_T']>=1-1e-9,H3_branch_away=bool(x['K4_branch_away_from_edge']))
    fig,axes=plt.subplots(1,2,figsize=(12,3.6),layout='constrained')
    im=axes[0].imshow(q.T,origin='lower',extent=(0,2,0,1),cmap='magma',vmin=0,vmax=1);axes[0].set(title=f"Damage, thermal run, t={x['end_time']:.2f}",xticks=[],yticks=[]);fig.colorbar(im,ax=axes[0],shrink=.8)
    im=axes[1].imshow((T-x['min_T']).T,origin='lower',extent=(0,2,0,1),cmap='inferno');axes[1].set(title='Temperature rise T - T₀',xticks=[],yticks=[]);fig.colorbar(im,ax=axes[1],shrink=.8)
    for ext in ('png','svg'):fig.savefig(out/'figures'/f'dynamic-branching-thermal.{ext}',dpi=140)
    plt.close(fig);b=x['first_early_branch'];bi=iso['first_early_branch']
    (out/'post-hoc-thermal'/'gates.json').write_text(json.dumps(dict(gates=gates,peak_rise_in_band=peak,onset=dict(thermal=b,isothermal=bi)),indent=1)+'\n')
    return ['## Fracture heat and thermal softening (Section 81)','',
        'The wide 2 × 1 case at σ = 0.09 is rerun with the thermochemical model\'s fracture heat: dissipated fracture energy becomes heat in proportion to each cell\'s damage growth (c_v = 0.1), heat conducts (κ = 1e-4), and toughness softens as G_c(T) = G_c0 max(1 - (T - T₀), 0.1). These gates were declared before the run.','',
        '| Gate | Outcome |','| --- | --- |']+[f'| {k} | {mark(v)} |' for k,v in gates.items()]+['',
        '| Run | First two-band time | Tip at onset | End time | Energy drift | Peak T - T₀ on the branches |','| --- | ---: | ---: | ---: | ---: | ---: |',
        f"| isothermal (Section 66.4) | {bi['time']:.2f} | {bi['tip']:.2f} | {iso['end_time']:.2f} | {iso['energy_drift_fraction']:.1e} | – |",
        f"| thermal | {b['time']:.2f} | {b['tip']:.2f} | {x['end_time']:.2f} | {x['energy_drift_fraction']:.1e} | {peak['0.9–2']:.3f} |",'',
        f"Softening advances the branching onset by {bi['time']-b['time']:.2f} and leaves its position unchanged. In cells with damage of at least 0.9, the peak rise is "+', '.join(f'{v:.2f} for x in {k}' for k,v in peak.items())+'. The hottest material is on the branches, where G_c falls by up to a third. Thermal expansion and composition are not part of this model.','',
        f'![Damage and temperature rise, thermal run]({rel}/figures/dynamic-branching-thermal.png)','']


def refinement(out,rel):
    """Section 83: the isothermal 2 x 1 case at n = 150, 200 (Section 66.4) and 300 cells per unit height."""
    paths={150:out/'refinement-n150'/'sigma-0.09.json',200:out/'post-hoc-wide'/'sigma-0.09.json',300:out/'refinement-n300'/'sigma-0.09.json'}
    if not all(p.exists() for p in paths.values()):return []
    x={n:json.loads(p.read_text()) for n,p in paths.items()};on={n:v['first_early_branch'] for n,v in x.items()}
    line=lambda b:min(float(k) for k,v in b['branches'].items() if v>=2) if b else None
    ok1=lambda v:v['min_increment']>=-1e-12 and v['q_bounds'][0]>=0 and v['q_bounds'][1]<=1 and v['max_speed']<=C_R and v['energy_drift_fraction']<=.05
    gates=dict(B1_bounds_energy=all(ok1(x[n]) for n in (150,300)),
        B2_same_line_and_time=all(on[n] is not None and line(on[n])==.9 for n in x) and abs(on[300]['time']-on[200]['time'])<=.05,
        B3_converging=all(on[n] is not None for n in x) and abs(on[300]['time']-on[200]['time'])<=abs(on[200]['time']-on[150]['time']))
    fig,axes=plt.subplots(1,3,figsize=(18,3.6),layout='constrained')
    for ax,n in zip(axes,(150,200,300)):
        d=out/('post-hoc-wide' if n==200 else f'refinement-n{n}');q=np.load(d/'sigma-0.09.npz')['final'].reshape(2*n+1,n+1)
        ax.imshow(q.T,origin='lower',extent=(0,2,0,1),cmap='magma',vmin=0,vmax=1);ax.set(title=f"n = {n} (ℓ/h = {.015*n:g}), t = {x[n]['end_time']:.2f}",xticks=[],yticks=[])
    for ext in ('png','svg'):fig.savefig(out/'figures'/f'dynamic-branching-refinement.{ext}',dpi=140)
    plt.close(fig)
    (out/'refinement-gates.json').write_text(json.dumps(dict(gates=gates,onset={n:on[n] for n in x},max_speed={n:x[n]['max_speed'] for n in x},drift={n:x[n]['energy_drift_fraction'] for n in x}),indent=1)+'\n')
    lines=['## Mesh refinement (Section 83)','','The isothermal 2 × 1 case (σ = 0.09, dt = 0.001, ℓ = 0.015) at three resolutions, with all physical parameters unchanged. These gates were declared before the runs.','',
        '| Gate | Outcome |','| --- | --- |']+[f'| {k} | {mark(v)} |' for k,v in gates.items()]+['',
        '| n | ℓ/h | First two-band time | Tip at onset | Bands on x = 0.9–1.4 (max before the stop) | Max tip speed | Energy drift | Seconds |','| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    for n in (150,200,300):
        v=x[n];b=on[n];lines.append(f"| {n} | {.015*n:g} | {'–' if b is None else '%.2f'%b['time']} | {'–' if b is None else '%.2f'%b['tip']} | {', '.join(str(c) for c in v['max_bands_before_edge'].values())} | {v['max_speed']:.2f} | {v['energy_drift_fraction']:.1e} | {v['seconds']:.0f} |")
    d1,d2=on[200]['time']-on[150]['time'],on[300]['time']-on[200]['time']
    lines+=['',f"The onset converges: {on[150]['time']:.2f}, {on[200]['time']:.2f} and {on[300]['time']:.2f} at ℓ/h = 2.25, 3 and 4.5, always on the x = 0.9 line with the tip at 0.90. The change is {d1:+.2f} and then {d2:+.2f}. By the amendment declared before the result, n = 300 ends at t = {x[300]['end_time']:.2f} with the tip at x ≈ 1.1, so its zero bands at x ≥ 1.2 and its lower maximum speed reflect the shorter run, not the physics. Up to x = 1.1 all three resolutions show the same two bands. The secondary branching (four bands at x = 1.4) is common to n = 150 and 200."]
    return lines+['',f'![Final damage at three resolutions]({rel}/figures/dynamic-branching-refinement.png)','']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());g=r['gates']
    runs={s:json.loads((out/f'sigma-{s:g}.json').read_text()) for s in LOADS}
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,5,figsize=(24,4.6),layout='constrained')
    for ax,s in zip(axes[:3],LOADS):
        q=np.load(out/f'sigma-{s:g}.npz')['final'];n=r['n'];ax.imshow(q.reshape(n+1,n+1).T,origin='lower',extent=(0,1,0,1),cmap='magma',vmin=0,vmax=1)
        ax.set(title=f"σ={s:g}, t={runs[s]['end_time']:.2f}",xticks=[],yticks=[])
    for s in LOADS:
        rows=runs[s]['rows'];t=np.array([x['time'] for x in rows]);tip=np.array([x['tip'] for x in rows])
        axes[3].plot(t,tip,label=f'σ={s:g}')
        if len(t)>5:axes[4].plot(t[:-5],[(tip[i+5]-tip[i])/(t[i+5]-t[i]) for i in range(len(t)-5)],label=f'σ={s:g}')
    axes[3].set(xlabel='Time',ylabel='Rightmost x with d ≥ 0.8',title='Crack tip');axes[3].legend(fontsize=8)
    axes[4].axhline(r['c_R'],color='k',ls='--',lw=.8,label='c_R');axes[4].set(xlabel='Time',ylabel='Tip speed',title='Crack speed');axes[4].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'dynamic-branching.{ext}',dpi=140)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out
    lines=['# Dynamic crack branching','',
        'September 27, 2026. Section 66 tests dynamic crack branching with the Section 36 AT2 energy under inertia: explicit elastodynamics with a lumped mass, and after each step a staggered, irreversible damage minimization. The notched unit square is pulled by sudden tractions σ on its top and bottom edges (c_s = 1, c_R ≈ 0.93, static critical traction about 0.016). Parameters are synthetic and nondimensional.','',
        '| Gate | Outcome |','| --- | --- |']+[f'| {k} | {mark(v)} |' for k,v in g.items()]
    lines+=['','| σ | End time | Max tip speed | Max bands (x = 0.65, 0.75, 0.85) before the edge | First two-band time | Energy drift (fraction of peak) |','| ---: | ---: | ---: | --- | ---: | ---: |']
    for s in LOADS:
        x=runs[s];lines.append(f"| {s:g} | {x['end_time']:.3f} | {x['max_speed']:.3f} | {', '.join(str(v) for v in x['max_bands_before_edge'].values())} | {'–' if x['first_branch_time'] is None else '%.3f'%x['first_branch_time']} | {x['energy_drift_fraction']:.3f} |")
    lines+=['',f'![Final damage for three loads, tip position and speed]({rel}/figures/dynamic-branching.png)','']
    lines+=posthoc(out,rel)+thermal(out,rel)+refinement(out,rel)
    lines+=['```sh',
        f'for s in 0.03 0.06 0.09; do python3 benchmarks/dynamic_branching.py --output {rel} --sigma $s & done; wait',f'python3 benchmarks/dynamic_branching.py --output {rel}',f'for s in 0.15 0.25; do python3 benchmarks/dynamic_branching.py --output {rel}/post-hoc-strong --sigma $s & done  # post hoc',f'python3 benchmarks/dynamic_branching.py --output {rel}/post-hoc-wide --sigma 0.09 --width 2  # post hoc',f'python3 benchmarks/dynamic_branching.py --output {rel}/post-hoc-thermal --sigma 0.09 --width 2 --thermal  # Section 81',f'for n in 150 300; do python3 benchmarks/dynamic_branching.py --output {rel}/refinement-n$n --sigma 0.09 --width 2 --n $n & done  # Section 83',f'python3 benchmarks/dynamic_branching_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Sections 66 and 81](unified_multiscale_defect_framework.md).','']
    write_doc('DYNAMIC_BRANCHING_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
