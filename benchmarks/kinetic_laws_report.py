"""Section 56 report: figures and markdown from saved kinetic-law evidence."""
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
    out=args.input;r=json.loads((out/'report.json').read_text());c,a,s=r['curvature'],r['anisotropic'],r['stress'];figures=out/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    for name,run in c['runs'].items():
        k=np.array(run['kappa']);axes[0].plot(k,-np.array(run['V'])/k,'.',ms=3,label=f'R0={name}, h=0.5')
    k=np.array(c['fine']['kappa']);axes[0].plot(k,-np.array(c['fine']['V'])/k,'k.',ms=3,label='R0=12, h=0.25')
    kk=np.linspace(.04,.2,50)
    for key,style in (('cubic_fit_h05','C0--'),('cubic_fit_h025','k--')):
        f=c['post_hoc'][key];axes[0].plot(kk,f['m']+f['a']*kk**2,style,lw=.8)
    axes[0].axhline(1,color='gray',lw=.6);axes[0].set(xlabel='Curvature κ',ylabel='-V/κ',title='Velocity–curvature: V/κ against κ');axes[0].legend(fontsize=7)
    th=np.linspace(0,2*np.pi,360,endpoint=False)
    for key,run in a['runs'].items():
        rr=np.array(run['radii']);axes[1].plot(np.degrees(th),rr/run['a0']-1,label=f"δ={key}")
        if key!='0':axes[1].plot(np.degrees(th),-float(key)/14*np.cos(4*th)+a['control_lattice_amplitude']*np.cos(4*th),'k:',lw=.8)
    axes[1].set(xlabel='Polar angle (deg)',ylabel='r/a0 - 1',title='Anisotropic mobility: shape at R=12 (dotted: -δ/14 + lattice)');axes[1].legend(fontsize=8)
    taus=np.array(s['taus'])
    for N,row in s['rows'].items():axes[2].plot(taus,row['velocities'],'o-',label=f'N={N}')
    axes[2].plot(taus,-.2*taus+s['rows']['48']['intercept'],'k--',lw=.8,label='-M b τ + v0')
    axes[2].set(xlabel='Applied shear τ',ylabel='Mean radial velocity',title='Loop velocity–stress through the PK pipeline');axes[2].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'kinetic-laws.{ext}',dpi=170)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out;g=c['gates'];ph=c['post_hoc']
    lines=['# Kinetic laws: velocity–curvature, anisotropic mobility, velocity–stress','',
        'September 26, 2026. Section 56 derives three kinetic laws that the brief names explicitly (sections 2.4 and 18) and tests each against gates declared before running. Earlier evidence was indirect: case 1 fitted only the radius-squared slope, anisotropic mobility was not implemented, and case 6 checked its velocity–stress law by integrating the prescribed drag directly. That check never called the force computation, so it tested nothing beyond arithmetic, and it is superseded here.','',
        '## 1. Velocity–curvature (Allen–Cahn)','',
        f"Prediction V = -L k κ. Circles with R0 = 8, 12, 16 and 24 (h=0.5) give a pooled fit of V = -m κ + b with m = {c['pooled_slope']:.4f} and b = {c['intercept']:.1e}.",'',
        '| Gate | Declared criterion | Measured | Outcome |','| --- | --- | --- | --- |',
        f"| G1 | abs(m-1) ≤ 0.02 | {abs(c['pooled_slope']-1):.4f} | {mark(g['G1_slope'])} |",
        f"| G2 | abs(b) ≤ 0.02 max abs(V) | {abs(c['intercept'])/c['max_speed']:.4f} | {mark(g['G2_intercept'])} |",
        f"| G3 | per-R0 slopes within 1% | "+', '.join(f"{k}: {v:.4f}" for k,v in c['per_radius_slope'].items())+f" | {mark(g['G3_history'])} |",
        f"| G4 | h=0.25 changes m by ≤ 0.01 and is no farther from 1 | {c['fine_h025_slope']:.4f} vs {c['per_radius_slope']['12']:.4f} | {mark(g['G4_resolution'])} |",'',
        f"Diagnosis, added after G3 and G4 failed (not a declared gate). The declared law is only the leading order. V/κ rises from 0.99 at R=24 to 1.015 at R=5, because of a finite-width correction. Fitting a free intercept over different curvature windows then gives different slopes; that is why G3 fails. The dynamics do not depend on history. At equal radius, V/κ is identical across all four runs to {ph['max_equal_radius_spread']:.1e}, so V is a function of κ alone. With the next term, V = -m κ - a κ³: m = {ph['cubic_fit_h05']['m']:.4f} at h=0.5 and {ph['cubic_fit_h025']['m']:.4f} at h=0.25, so the leading coefficient converges to 1 with an error ratio of {ph['m_error_ratio']:.2f} per halving (close to second order). Meanwhile a = {ph['cubic_fit_h05']['a']:.3f} and {ph['cubic_fit_h025']['a']:.3f} is resolution independent: it is a property of the diffuse interface, about 0.36 ε² for ε = √2. G4 fails because the h=0.5 grid error partly cancels this physical correction, so the coarse linear fit happened to lie closer to 1. The recorded energy never increases.",'',
        '## 2. Anisotropic mobility','',
        "L(n) = L0 (1 + δ cos 4θ_n), R0 = 24 to R = 12. The two predictions are an area rate dA/dt = -2π L0 k, exactly and for every δ, and an asymptotic shape r = R(1 - (δ/14) cos 4θ).",'',
        '| δ | dA/dt | a4/a0 at R=12 | minus control | predicted -δ/14 |','| ---: | ---: | ---: | ---: | ---: |']
    for key,run in a['runs'].items():
        d=a['shape_amplitude_minus_control'].get(key);p=a['predicted'].get(key)
        lines.append(f"| {key} | {run['area_rate']:.4f} | {run['a4_over_a0']:.5f} | {'–' if d is None else f'{d:.5f}'} | {'–' if p is None else f'{p:.5f}'} |")
    ag=a['gates']
    lines+=['',f"Gates: A1 area rate {mark(ag['A1_area_rate'])}. The ±0.3 runs are within 0.08% of the control, and the control is 1.1% from -2π, a finite-width offset like G1's. A2 shape {mark(ag['A2_shape'])}: the measured amplitudes match -δ/14 within 2.5% for both signs, against a 20% gate. A3 energy {mark(ag['A3_energy'])}. The δ=0 control carries a lattice amplitude of {a['control_lattice_amplitude']:.1e} from the five-point Laplacian; it is subtracted, and it is 24x smaller than the physical effect.",'',
        '## 3. Velocity–stress through the Peach–Koehler pipeline','',
        'A circular glide loop (R=3, b=0.2, M=1) in periodic linear elasticity, loaded by a uniform applied shear σ_xz = τ through the prescribed mean strain. The velocity comes from the full model: Fourier stress, core filtering, Galerkin nodal Peach–Koehler force and nodal mobility.','',
        '| Nodes N | Slope | M b cos(π/N) | Relative slope error (sign corrected) | Linear residual | Equilibrium stress τ_c |','| ---: | ---: | ---: | ---: | ---: | ---: |']
    for N,row in s['rows'].items():
        lines.append(f"| {N} | {row['slope']:.12f} | {row['predicted_slope']:.12f} | {row['corrected_slope_relative_error']:.1e} | {row['residual']:.1e} | {row['equilibrium_stress']:.5f} |")
    sg=s['gates'];sc=s['sign_corrected_post_hoc']
    lines+=['',f"As declared, S1 {mark(sg['S1_slope'])}, S2 {mark(sg['S2_linear'])}, S3 {mark(sg['S3_order'])}. The declaration omitted an orientation sign. For this counterclockwise loop with b = +e_x, the Peach–Koehler force (σ b) × ξ under positive σ_xz points inward, so v_r = -M b cos(π/N) τ, and the declared positive slope was wrong. The measured magnitude agrees with M b cos(π/N) to 2e-15 at every N. With the sign corrected, S1 {mark(sc['S1_slope'])}, and S3 {mark(sc['S3_order'])} with error ratios {sc['order_ratios'][0]:.2f} and {sc['order_ratios'][1]:.2f} per doubling of N, the predicted second order. The declared S1 and S3 failures are retained as a theory error. The equilibrium stress τ_c converges with N, to -0.0275 at N=48; this is the stress that holds this loop against its line tension.",'',
        f'![Velocity–curvature, anisotropic shapes and loop velocity–stress]({rel}/figures/kinetic-laws.png)','',
        '```sh',f'for part in curvature anisotropic stress; do python3 benchmarks/kinetic_laws.py --output {rel} --part $part; done',f'python3 benchmarks/kinetic_laws.py --output {rel} --part assemble',f'python3 benchmarks/kinetic_laws_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 56](unified_multiscale_defect_framework.md).','']
    write_doc('KINETIC_LAWS_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
