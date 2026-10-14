"""Section 90 report: branching in the finite-strain thermochemical model, declared runs and post hoc closures, from saved evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'thermochemical-branching-2026-09-28';mark=lambda b:'PASS' if b else 'FAIL'
VARIANTS=(('','declared'),('-erode','90.2 reactive erosion (post hoc)'),('-erode-band0.9','90.4 band deletion (post hoc)'))


def gates(r):
    ok=all(r['checks'].values()) and r['energy_drift']<=1e-4
    return dict(H1=ok,H2=r['H2_off_line_max']<=.01,H3=r['first_branch'] is not None)


if __name__=='__main__':
    runs={(n,v):json.loads((OUT/f'n{n}{v}.json').read_text()) for v,_ in VARIANTS for n in (64,80) if (OUT/f'n{n}{v}.json').exists()}
    L=['# Dynamic branching in the finite-strain thermochemical model','',
       "September 28, 2026. Section 90 tries to branch a crack in the full finite-strain thermochemical model (NodalThermalFracture2D): Q2 mechanics with inertia, heat, thermal and compositional expansion, evolving composition and phase, and nodal AT1 damage with the exact energy ledger. Plastic flow is switched off (the brittle regime). The unit square is held in fixed grips at strain 0.09 (ψ ≈ 0.7 of the AT1 bulk threshold), with a notch along y = 0.5, ℓ = 0.03, G_c = 0.002 and damage mobility 5000 ([thermochemical_branching.py](benchmarks/thermochemical_branching.py)). Smoke tests showed that AT2 damages the whole bulk at this overdrive, so AT1 is used.",'',
       '| Run | Mesh | H1 checks and drift | H2 bulk intact | H3 branches | End | Tip | Stop reason | Eroded / deleted cells | Energy drift |','| --- | ---: | --- | --- | --- | ---: | ---: | --- | ---: | ---: |']
    for (n,v),r in runs.items():
        g=gates(r);label=dict(VARIANTS)[v]
        L.append(f"| {label} | {n} | {mark(g['H1'])} | {mark(g['H2'])} ({r['H2_off_line_max']:.1e}) | {mark(g['H3'])} | {r['end_time']:.3f} | {r['rows'][-1]['tip']:.3f} | {r['failure']['error'] if r['failure'] else 'tip past 0.85'} | {r.get('eroded_cells',0)} / {r.get('deleted_cells',0)} | {r['energy_drift']:.1e} |")
    fb={k:r['first_branch'] for k,r in runs.items() if r['first_branch']}
    H4=len([k for k in fb if k[1]==VARIANTS[2][0]])==2 and len({fb[(n,VARIANTS[2][0])]['line'] for n in (64,80)})==1 and abs(fb[(64,VARIANTS[2][0])]['time']-fb[(80,VARIANTS[2][0])]['time'])<=.1 if all((n,VARIANTS[2][0]) in fb for n in (64,80)) else False
    L+=['',f"H4 (the band-deletion meshes agree on the first branching line and time within 0.1): {mark(H4)}."]
    if fb:L+=['','First branching: '+'; '.join(f"n = {k[0]} {dict(VARIANTS)[k[1]]}: t = {b['time']:.3f}, line x = {b['line']:g}, tip {b['tip']:.3f}" for k,b in fb.items())+'.']
    fig,axes=plt.subplots(1,len(runs),figsize=(4*len(runs),4),layout='constrained',squeeze=False)
    for ax,((n,v),r) in zip(axes[0],runs.items()):
        z=np.load(OUT/f'n{n}{v}.npz');ax.imshow(z['final'].reshape(n+1,n+1).T,origin='lower',extent=(0,1,0,1),cmap='magma',vmin=0,vmax=1);ax.set(title=f"n={n} {v or 'declared'}, t={r['end_time']:.2f}",xticks=[],yticks=[])
    (OUT/'figures').mkdir(exist_ok=True)
    for ext in ('png','svg'):fig.savefig(OUT/'figures'/f'thermochemical-branching-final.{ext}',dpi=120)
    (OUT/'report.json').write_text(json.dumps(dict(runs={f'n{n}{v}':dict(gates=gates(r),**{k:x for k,x in r.items() if k not in ('rows',)}) for (n,v),r in runs.items()},H4=H4),indent=1,default=float)+'\n')
    L+=['',"H1 passes on every run: energy drift at most 1.8e-6, with every check at every accepted step. The declared runs stop at t = 0.248 on both meshes, on an element inversion beside the separated crack (Sections 64–65). Reactive erosion (90.2) and band deletion (90.4, cells with all four damage nodes at least 0.9) are both path-independent, and they carry the runs to t ≈ 0.40–0.53, with the tip at x = 0.46–0.56. Band deletion does so consistently on both meshes. Before every stop the tip grows a bulb of heavy damage about 0.1 wide, the widening that precedes branching, and the finite-strain Q2 elements fold inside it. H2 fails on a damage spot that nucleates at the bottom grip, not on a branch.",
        '',"Decision (theory 90.5): branching inside the full finite-strain thermochemical model is not reached. The cause is identified. Finite-strain elements with orientation certification cannot carry the pre-branching damage zone of a smeared crack, and the path-independent closures stop at the same widening. The split would need discontinuity kinematics for arbitrary, bifurcating paths (XFEM, cohesive remeshing), a specialized-method regime. The framework's branching model is the small-strain elastodynamic one with fracture heat (Sections 81, 83), which branches because linear kinematics cannot fold."]
    L+=['','![Final damage of every run](results/thermochemical-branching-2026-09-28/figures/thermochemical-branching-final.png)','','```sh','for n in 64 80; do python3 benchmarks/thermochemical_branching.py --n $n & done; wait','for n in 64 80; do python3 benchmarks/thermochemical_branching.py --n $n --erode & done; wait  # 90.2','for n in 64 80; do python3 benchmarks/thermochemical_branching.py --n $n --erode --band 0.9 & done; wait  # 90.4','python3 benchmarks/thermochemical_branching_report.py','```','','[Report](results/thermochemical-branching-2026-09-28/report.json), [theory Section 90](unified_multiscale_defect_framework.md).','']
    write_doc('THERMOCHEMICAL_BRANCHING_RESULTS.md','\n'.join(L));print('\n'.join(L[4:4+len(runs)+2]))
