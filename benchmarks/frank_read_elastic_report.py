"""Section 77 report: Frank-Read critical stress with non-singular elastic self-interaction, from saved evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'frank-read-elastic-2026-09-28'
mark=lambda b:'PASS' if b else 'FAIL'
NAMES={'F1_declared':'F1 parallel-line force matches -K/d within 2% (declared)','F1_exact_post_hoc':'F1 against the exact finite non-singular segments, 1e-10 (post hoc)','F2_foreman_log':'F2 τ_c L/(μb) linear in ln(L/a), R² ≥ 0.99'}

if __name__=='__main__':
    r=json.loads((OUT/'report.json').read_text());c=r['critical'];fit=r['fit'];nu=.324
    L=np.array([x['L'] for x in c]);y=np.array([x['scaled'] for x in c])
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(np.log(L),y,'o',label='bisection τ_c');xx=np.linspace(np.log(40),np.log(500));axes[0].plot(xx,fit['slope']*xx+fit['intercept'],'-',label=f"fit, slope {fit['slope']:.3f}")
    for k,lab in ((1/(2*np.pi),'screw 1/2π'),(1/(2*np.pi*(1-nu)),'edge 1/2π(1-ν)')):axes[0].plot(xx,k*(xx-np.log(L[0]))+y[0],':',lw=.9,label=f'slope {lab}')
    axes[0].set(xlabel='ln(L/a)',ylabel='τ_c L/(μb)',title='Critical stress with elastic self-interaction');axes[0].legend(fontsize=8)
    axes[1].plot(L,[x['ratio'] for x in c],'o-');axes[1].set(xscale='log',xlabel='L/a',ylabel='τ_c / (2Γ_screw/bL)',title='Ratio to the line-tension estimate')
    (OUT/'figures').mkdir(exist_ok=True)
    for ext in ('png','svg'):fig.savefig(OUT/'figures'/f'frank-read-elastic.{ext}',dpi=150)
    plt.close(fig)
    lines=['# Frank–Read sources with elastic self-interaction','',
        'September 28, 2026. Section 60 bowed Frank–Read sources under a constant line tension. Section 77 gives the bowing line its complete isotropic elastic self-energy in the non-singular form of Cai et al. (2006) ([elastic_line.py](src/bmvr/elastic_line.py), [frank_read_elastic.py](benchmarks/frank_read_elastic.py)). Node forces are the automatic-differentiation gradient of W - τbA. The critical stress τ_c is the largest stress with a stable bowed equilibrium, found by bisection (μ = 1, ν = 0.324, a = 1).','',
        '| Gate | Outcome |','| --- | --- |']+[f'| {NAMES[k]} | {mark(v)} |' for k,v in r['gates'].items()]
    lines+=['','| d/a | Declared -K/d error | Exact-segment error |','| ---: | ---: | ---: |']
    for d in (5.,10.,20.):
        e=[x for x in r['F1'] if x['d']==d];lines.append(f"| {d:g} | {100*max(x['declared_rel'] for x in e):.2f}% | {max(x['exact_rel'] for x in e):.1e} |")
    lines+=['','The errors are the same at 0°, 45° and 90°. The declared reference -K/d is the limit of infinite, singular lines. The test lines have length 400a and core a, and the two corrections, about a²/d² and d/L, give the measured pattern: 4% + 1.25% at d = 5, 1% + 2.5% at d = 10, and 0.25% + 5% at d = 20. The exact interaction of two finite non-singular segments, K[L asinh(L/D) - (L² + D²)^{1/2} + D] with D = (d² + a²)^{1/2}, is matched to 2e-13 (post hoc). F1 stays failed as declared.',
        '','| L/a | τ_c/μ | τ_c L/(μb) | Ratio to 2Γ_screw/(bL) |','| ---: | ---: | ---: | ---: |']+[f"| {x['L']:g} | {x['tau_c']:.5f} | {x['scaled']:.4f} | {x['ratio']:.3f} |" for x in c]
    lines+=['',f"τ_c L/(μb) is linear in ln(L/a) with slope {fit['slope']:.3f} and R² = {fit['r2']:.4f}. That is Foreman's logarithm, which the L-independent line tension of Section 60 cannot produce. The slope lies between the screw and edge prefactors, 1/2π = 0.159 and 1/2π(1 - ν) = 0.235, and close to the edge value. The screw line-tension estimate underpredicts τ_c by a factor 1.36–1.40, and the factor grows slowly with L. The last increment (0.169 per doubling, against 0.157–0.158) is slightly larger.",
        '','![Critical stress against ln(L/a), and the ratio to line tension](results/frank-read-elastic-2026-09-28/figures/frank-read-elastic.png)','',
        '```sh','python3 benchmarks/frank_read_elastic.py','python3 benchmarks/frank_read_elastic_report.py','```','','[Report](results/frank-read-elastic-2026-09-28/report.json), [theory Section 77](unified_multiscale_defect_framework.md).','']
    write_doc('FRANK_READ_ELASTIC_RESULTS.md','\n'.join(lines))
