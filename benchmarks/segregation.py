"""Section 57: McLean isotherm and Gibbs adsorption equation in the diffuse interface, with declared gates."""
import argparse,json,platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.segregation import Segregation1D
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
MEANS=np.geomspace(.001,.1,12)


def solve(model,mean,dt):
    eta,c=model.initial(mean);eta,c,info=model.equilibrate(eta,c,dt);ex=model.excess(eta,c)
    return eta,c,dict(mean_solute=float(mean),**ex,**{k:v for k,v in info.items()},
        isotherm_error=float(np.max(np.abs(c-model.isotherm(eta,ex['mu'])))),energy_scale=abs(model.energy(eta,c)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);model=Segregation1D();rows=[];profiles={}
    for mean in MEANS:
        eta,c,row=solve(model,mean,.02);rows.append(row);profiles[f'{mean:.4g}']=(eta,c)
    mu=np.array([r['mu'] for r in rows]);gamma=np.array([r['gamma'] for r in rows]);Gamma=np.array([r['adsorption'] for r in rows])
    slope=np.gradient(gamma,mu,edge_order=2);interior=slice(1,-1);gibbs=np.abs(slope[interior]/(-Gamma[interior])-1)
    fine=Segregation1D(n=320);mid=float(MEANS[len(MEANS)//2]);_,_,fine_row=solve(fine,mid,.005);coarse_mid=rows[len(MEANS)//2]
    gamma0=2*np.sqrt(2)/3
    # Post hoc, after Z3 failed: close pairs isolate the finite-difference truncation of the declared spacing,
    # whose central-difference error for Gamma ~ exp(mu/theta) is about (d mu/theta)^2/6.
    pairs=[]
    for i in (3,6,9):
        lo=solve(model,MEANS[i]*.995,.02)[2];hi=solve(model,MEANS[i]*1.005,.02)[2]
        d=(hi['gamma']-lo['gamma'])/(hi['mu']-lo['mu']);pairs.append(dict(mean=float(MEANS[i]),slope=d,adsorption=rows[i]['adsorption'],relative_error=abs(d/(-rows[i]['adsorption'])-1),dmu=hi['mu']-lo['mu']))
    spacing=float(np.mean(np.diff(mu)));predicted_truncation=(spacing/model.theta)**2/6
    gates=dict(Z1_isotherm=max(r['isotherm_error'] for r in rows)<=1e-8,
        Z2_conservation=max(r['mass_drift'] for r in rows)<=1e-12 and all(r['max_energy_increase']<=1e-12*r['energy_scale'] for r in rows),
        Z3_gibbs=bool(np.max(gibbs)<=.01),Z4_monotone_and_limit=bool(np.all(np.diff(gamma)<0) and abs(gamma[0]/gamma0-1)<=.005),
        Z5_resolution=abs(fine_row['adsorption']/coarse_mid['adsorption']-1)<=.01)
    report=dict(scope='Section 57 segregation to a diffuse planar interface; declared gates.',parameters=dict(model.__dict__),means=MEANS.tolist(),rows=rows,
        gibbs_slope=slope.tolist(),gibbs_relative_error_interior=gibbs.tolist(),gamma0=gamma0,post_hoc_close_pairs=pairs,predicted_declared_truncation=predicted_truncation,fine_median=fine_row,coarse_median=coarse_mid,gates=gates,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()))
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item())+'\n')
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for key in ('0.001','0.01','0.1'):
        k=min(profiles,key=lambda s:abs(float(s)-float(key)));eta,c=profiles[k]
        axes[0].semilogy(model.x,c,label=f'mean {float(k):.3g}');axes[0].semilogy(model.x,model.isotherm(eta,float(np.mean(model.potential(eta,c)))),'k:',lw=.8)
    axes[0].set(xlim=(-8,8),xlabel='x',ylabel='Solute fraction c',title='Equilibrium profiles (dotted: McLean isotherm)');axes[0].legend(fontsize=8)
    axes[1].plot(mu,gamma,'o-');axes[1].axhline(gamma0,color='k',ls='--',lw=.8,label='γ0 (no solute)');axes[1].set(xlabel='Chemical potential μ',ylabel='Interface tension γ',title='γ(μ)');axes[1].legend(fontsize=8)
    axes[2].plot(mu,Gamma,'o',label='Adsorption Γ');axes[2].plot(mu,-slope,'-',label='-dγ/dμ');axes[2].set(xlabel='Chemical potential μ',ylabel='Γ',title='Gibbs adsorption equation');axes[2].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'segregation.{ext}',dpi=170)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out;mark=lambda b:'PASS' if b else 'FAIL'
    lines=['# Interface segregation and adsorption','',
        'September 26, 2026. Section 57 tests segregation thermodynamics in the diffuse interface, as brief section 2.4 requires. Solute bound to a planar AC interface (E_b/θ = 3) relaxes by conservative diffusion and phase relaxation to equilibrium. Two exact results are tested there: the McLean–Langmuir isotherm, pointwise inside the diffuse interface, and the Gibbs adsorption equation dγ/dμ = -Γ, which holds for the discrete energy through the envelope theorem.','',
        '| μ | Bulk solute | Adsorption Γ | -dγ/dμ | Relative error | γ | Isotherm error | Steps |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for i,r in enumerate(rows):
        err='–' if i in (0,len(rows)-1) else f'{gibbs[i-1]:.1e}'
        lines.append(f"| {r['mu']:.4f} | {r['bulk_solute']:.2e} | {r['adsorption']:.5f} | {-slope[i]:.5f} | {err} | {r['gamma']:.5f} | {r['isotherm_error']:.1e} | {r['steps']} |")
    lines+=['',f"Gates: Z1 isotherm {mark(gates['Z1_isotherm'])} (max error {max(r['isotherm_error'] for r in rows):.1e}); Z2 conservation {mark(gates['Z2_conservation'])} (mass drift at most {max(r['mass_drift'] for r in rows):.1e}); Z3 Gibbs {mark(gates['Z3_gibbs'])} (largest interior error {np.max(gibbs):.1e}); Z4 monotone γ and solute-free limit {mark(gates['Z4_monotone_and_limit'])} (γ at the lowest solute {gamma[0]:.5f} against γ0 = {gamma0:.5f}); Z5 resolution {mark(gates['Z5_resolution'])} (Γ {coarse_mid['adsorption']:.5f} at h=0.25 against {fine_row['adsorption']:.5f} at h=0.125).",'',
        f"Z3 fails as declared. Its error is almost uniform, 2.5–2.9% at every interior point, which is the truncation of a central difference across the declared spacing: Γ grows like exp(μ/θ), the spacing is Δμ/θ ≈ {spacing/model.theta:.2f}, and (Δμ/θ)²/6 = {predicted_truncation:.3f}. The declared twelve-point spacing was too coarse for a 1% gate, which is a test-design error. Post hoc close pairs (mean solute ±0.5%, Δμ ≈ {np.mean([q['dmu'] for q in pairs]):.4f}) give -dγ/dμ = Γ within "+', '.join(f"{q['relative_error']:.1e}" for q in pairs)+" at three means, confirming the identity. The declared failure is retained.",'',
        f"Segregation lowers the interface tension from {gamma[0]:.4f} to {gamma[-1]:.4f} across this range. The Gibbs equation is exact for the discrete model, so the finite-difference error reflects only the μ spacing. The binding energy, mixing scale and mobility are synthetic; a calibrated segregation energy would come from DFT or atom-probe data (PARAMETER_REGISTRY.md).",'',
        f'![Profiles, γ(μ) and the Gibbs adsorption equation]({rel}/figures/segregation.png)','','```sh',f'python3 benchmarks/segregation.py --output {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 57](unified_multiscale_defect_framework.md).','']
    write_doc('SEGREGATION_RESULTS.md','\n'.join(lines));print(gates,float(np.max(gibbs)),flush=True)


if __name__=='__main__':main()
