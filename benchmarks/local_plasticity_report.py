"""Section 59 report: figures and markdown from saved local-plasticity evidence."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.grid import Grid
from bmvr.adaptivity import transfer
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
mark=lambda b:'PASS' if b else 'FAIL'
POST_HOC={'32':(.015512,.2001),'64':(.006417,.0828)}  # uniform h=2 and h=1 gamma error against h=0.5 (post hoc run recorded in the outcome)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());f=np.load(out/'fields.npz');g=r['gates'];u,a=r['runs']['uniform'],r['runs']['adaptive'];av=r['adaptive_vs_uniform']
    ph1=json.loads((out/'post-hoc-linear-interpolation/report.json').read_text());ph2=json.loads((out/'post-hoc-volume-weights/report.json').read_text())
    gu=Grid(f['uniform_xe'],f['uniform_ye'],periodic=True);ga=Grid(f['adaptive_xe'],f['adaptive_ye'],periodic=True);gamma=f['uniform_gamma'];til=gamma-gu.integrate(gamma)/gu.area
    diff=transfer(ga,gu,f['adaptive_gamma'])-gamma
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    im=axes[0].pcolormesh(gu.xe,gu.ye,til,cmap='RdBu_r',vmin=-abs(til).max(),vmax=abs(til).max());fig.colorbar(im,ax=axes[0],shrink=.8)
    axes[0].contour(gu.x,gu.y,f['uniform_v'],levels=[0],colors='k',linewidths=.8);axes[0].plot(*np.array(u['final_positions']).T,'ko',ms=4)
    axes[0].set(title='Slip fluctuation γ - ⟨γ⟩ at t=8 (uniform), inclusion and screws',aspect='equal')
    im=axes[1].pcolormesh(gu.xe,gu.ye,diff,cmap='RdBu_r',vmin=-abs(diff).max(),vmax=abs(diff).max());fig.colorbar(im,ax=axes[1],shrink=.8)
    for x in (-24,24):axes[1].axvline(x,color='k',lw=.5,ls='--');axes[1].axhline(x,color='k',lw=.5,ls='--')
    axes[1].set(title='Adaptive minus uniform γ (dashed: fine band)',aspect='equal')
    t=[x['time'] for x in u['rows']];axes[2].plot(t,[x['mean_stress'] for x in u['rows']],label='mean stress')
    axes[2].plot(t,[x['mean_gamma'] for x in u['rows']],label='⟨γ⟩ (continuum slip)');axes[2].plot(t,[100*x['plastic_screws'] for x in u['rows']],label='100 × screw plastic shear')
    axes[2].set(xlabel='Time',title='Plastic relaxation by both mechanisms');axes[2].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'local-plasticity.{ext}',dpi=160)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out;h=r['homogeneous'];first,last=u['rows'][0],u['rows'][-1]
    lines=['# Case 9 with local crystal plasticity','',
        'September 26, 2026. Section 59 replaces case 9\'s single analytical mean-shear mode with a local continuum slip field γ(x). The slip field has periodic antiplane elasticity, hardening and a gradient term. It is coupled to the diffuse inclusion, the defect population and two screw dislocations through one energy, in one adaptive calculation. The energy separates the smeared-distortion elasticity E_el, the screw–slip cross term and the analytic screw pair energy, so nothing is counted twice. For γ = 0 it reduces exactly to case 9. Automated tests check every force, including the screw glide forces from the cross term, against finite differences of the discrete energy, on uniform and nonuniform periodic grids.','',
        '## Gates','','| Gate | Outcome | Evidence |','| --- | --- | --- |',
        f"| P1 energy, inventory, Burgers | {mark(g['P1_consistency'])} | largest energy increase {max(u['max_energy_increase'],a['max_energy_increase']):.1e}; inventory drift ≤ {max(u['inventory_drift'],a['inventory_drift']):.1e}; Burgers content 0 |",
        f"| P2 homogeneous slip | {mark(g['P2_homogeneous'])} | recurrence error ≤ {max(x['recurrence_error'] for x in h.values()):.1e}; continuous error {h['0.02']['continuous_error']:.2e} → {h['0.01']['continuous_error']:.2e} on halving dt (ratio {h['0.02']['continuous_error']/h['0.01']['continuous_error']:.2f}) |",
        f"| P3 local plasticity | {mark(g['P3_local'])} | max abs(γ̃)/max abs(γ) = {u['locality']['max_tilde']/u['locality']['max_gamma']:.2f} (≥ 0.25 passes); fraction of ∫abs(γ̃) within 8 of the screws {u['locality']['near_fraction']:.2f} uniform, {a['locality']['near_fraction']:.2f} adaptive (≥ 0.5 fails) |",
        f"| P4 plastic relaxation | {mark(g['P4_relaxation'])} | mean stress {first['mean_stress']:.4f} → {last['mean_stress']:.4f}; ⟨γ⟩ = {last['mean_gamma']:.4f} and screw plastic shear {last['plastic_screws']:.2e} both contribute |",
        f"| P5 adaptive against uniform | {mark(g['P5_adaptive'])} | relative L2: v {av['relative_l2']['v']:.1e}, c {av['relative_l2']['c']:.1e}, γ {av['relative_l2']['gamma']:.2%} (> 1% fails); screws {av['max_position_difference']:.1e}; speedup {av['solver_ratio_including_adaptation']:.2f}x with {av['dof_ratio']:.2f}x fewer unknowns |",'',
        '## Why P3 fails','',
        f"The slip field responds to the screws' 1/r stress and to its own long-range elastic interactions, so its fluctuation is not a compact plastic zone. {u['locality']['near_fraction']:.0%} of it lies within 8 of the screws, and the rest spreads across the cell (left panel). The declared radius assumed a localization that this physics does not have. The failure stands.",'',
        '## Why P5 fails, and what fixed it (post hoc)','',
        f"The difference map (middle panel) puts the declared run's γ error of {av['relative_l2']['gamma']:.2%} in thin stripes at the y = ±24 fine/coarse transitions: positive on the fine side, negative on the coarse side, and nearly absent at x = ±24. γ enters the elasticity only on y-faces. The declared operator averaged it to each face with weights 1/2. Where the spacing jumps from 0.5 to 2, that gives the fine cell next to the transition 1.75 times the local stress, a spurious force dipole. My first written diagnosis blamed coarse-region resolution, citing uniform h = 2 and h = 1 errors of {POST_HOC['32'][0]:.2%} and {POST_HOC['64'][0]:.2%}. That was wrong: the error pattern is a transition artifact, not a resolution error.",'',
        f"Post hoc attempt 1: linear interpolation to the face. It gives the nearer (fine) cell weight 0.8 and a stress factor of 2.5, and the γ error rose to {ph1['adaptive_vs_uniform']['relative_l2']['gamma']:.2%}. Post hoc attempt 2 derives the weights. For cellwise-constant γ, exact quadrature of (w_y - γ)² over the face's dual cell gives the half-cell-volume average γ̄ = (dy_lo γ_lo + dy_hi γ_hi)/(dy_lo + dy_hi), up to a jump term that the declared scheme also omits. With these weights every cell receives exactly its own volume of face stress (checked to 1.0), and uniform grids are bitwise unchanged. Adaptive against uniform then agrees to γ {ph2['adaptive_vs_uniform']['relative_l2']['gamma']:.3%}, v {ph2['adaptive_vs_uniform']['relative_l2']['v']:.1e}, c {ph2['adaptive_vs_uniform']['relative_l2']['c']:.1e} and screw positions {ph2['adaptive_vs_uniform']['max_position_difference']:.1e}, at {ph2['adaptive_vs_uniform']['solver_ratio_including_adaptation']:.2f}x speed with {ph2['adaptive_vs_uniform']['dof_ratio']:.2f}x fewer unknowns. P5 passes post hoc. The declared failure is retained as an implementation error, and the corrected operator is the one kept in the code.",'',
        f'![Slip fluctuation, adaptive error, and relaxation by both mechanisms]({rel}/figures/local-plasticity.png)','','```sh',f'python3 benchmarks/local_plasticity.py --output {rel} --declared-interpolation',f'python3 benchmarks/local_plasticity.py --output {rel}/post-hoc-volume-weights',f'python3 benchmarks/local_plasticity_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 59](unified_multiscale_defect_framework.md).','']
    write_doc('LOCAL_PLASTICITY_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
