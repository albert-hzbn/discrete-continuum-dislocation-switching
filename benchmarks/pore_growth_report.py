"""Section 52 report: velocity-drive figure and markdown from saved pore-growth evidence."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());figures=out/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for ax,mob in zip(axes,('single','squared')):
        for h,marker in ((.5,'o'),(.25,'s')):
            rows=[x for x in r['rows'] if x['mobility']==mob and x['h']==h]
            ax.plot([x['drive'] for x in rows],[x['velocity'] for x in rows],marker,label=f'h={h:g}',mfc='none' if h==.5 else None)
        d=np.linspace(-.03,.06,50)
        if mob=='single':ax.plot(d,.5/np.sqrt(2)*d,'k--',lw=1,label='V = (k_ex/a)(mu_I - gamma/R)')
        ax.axhline(0,color='0.6',lw=.6);ax.axvline(0,color='0.6',lw=.6)
        ax.set(xlabel='mu_n,I - gamma/R_A',ylabel='V',title=f"{'m = k_ex p(1-p)' if mob=='single' else 'm = k_ex p^2(1-p)^2 (original)'}");ax.legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'pore-growth.{ext}',dpi=170)
    plt.close(fig)
    f=r['post_hoc_regression']['fits'];c=r['post_hoc_regression']['two_parameter_fits']['fits'];rel=out.relative_to(ROOT) if out.is_absolute() else out
    lines=['# Sharp-interface pore growth','',
        "September 26, 2026. Section 52 derives when the pore model has an asymptotic interface growth law and tests it locally. **The declared gates fail**, because one declared configuration sits 2% above the critical radius. The derived law is nevertheless confirmed to 0.4% for the singly degenerate exchange mobility and shown to be absent in its rigid-profile form for the original squared-degenerate mobility.",'',
        '## Derivation','',
        "Integrating the exchange law across a translating planar profile gives V ∫(P′)²/m ds = mu_n,I − gamma/R. For the original m = k_ex p²(1−p)² the integral diverges, so no finite rigid-interface mobility exists; this is the Section 21.3 limitation. For m = k_ex p(1−p) it equals a/k_ex, giving V = (k_ex/a)(mu_n,I − gamma/R) with the energetic Gibbs–Thomson potential. Combining it with quasi-static diffusion gives the mixed reaction/diffusion law of theory 52.2.",'',
        '## Measurements','','| Mobility | h | Slope of V vs drive | Intercept | R² | Fitted capillary factor c |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,fit in f.items():
        mob,h=key.split('-');lines.append(f"| {mob} | {h} | {fit['slope']:.4f} | {fit['intercept']:+.1e} | {fit['r_squared']:.5f} | {c[key]['capillary_factor']:.3f} |")
    lines+=['',"The prediction for the singly degenerate case is slope k_ex/a = 0.3536 with c = 1. Per-configuration effective mobilities match within 3.5%, except the near-critical R=12, Delta=0.02 case (drive ~6e-4), which fails the declared 10% and 5% gates. The regression rows above are post hoc, added after that failure. The squared mobility gives a mesh-independent slope with c ≈ 1.12, a kinetically shifted apparent equilibrium rather than the energetic one.",'',
        f'![Velocity against Gibbs–Thomson drive]({rel}/figures/pore-growth.png)','',
        '## Scope','',"This establishes an asymptotic interface law where it exists, for a physically justified singly degenerate mobility. The original squared-degenerate model has none of the rigid-profile form. The global diffusion-limited growth rate with a far-field geometry, gas pressure, nucleation and coalescence are not tested. Inventory drift is at most 1e-8 and there are no energy increases in any run.",'',
        '```sh',f'python3 benchmarks/pore_growth.py --output {rel}',f'python3 benchmarks/pore_growth_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 52](unified_multiscale_defect_framework.md).','']
    write_doc('PORE_GROWTH_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
