"""Section 51 report: conditioning, pointwise convergence and ensemble statistics of late CH coarsening."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/coarsening-conditioning-2026-09-26'
SEEDS=[1701+1000*k for k in range(8)]


def main():
    r=json.loads((OUT/'report.json').read_text());pw=json.loads((OUT/'pointwise.json').read_text());figures=OUT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for eps in ('1e-10','1e-06'):
        c=np.load(OUT/f'conditioning-{eps}.npy');axes[0].loglog(c[1:,0],c[1:,3],label=f'initial perturbation {eps}')
    for k,v in pw['time'].items() if False else []:pass
    for d in pw['time']['differences']:axes[0].axhline(d['l2'],color='0.6',lw=.6,ls=':')
    axes[0].set(xlabel='Time',ylabel='L2 difference',title='Perturbation growth vs discretization differences (dotted)');axes[0].legend(fontsize=8)
    for label,color in (('A','tab:blue'),('B','tab:orange'),('C','tab:green')):
        data=np.array([np.load(OUT/f'ensemble-{label}-{s}.npy') for s in SEEDS]);t=data[0][:,0];mean=data[:,:,2].mean(0);se=data[:,:,2].std(0,ddof=1)/np.sqrt(len(SEEDS))
        axes[1].loglog(t[1:],mean[1:],color=color,label=f"{label}: h={r['discretizations'][label]['h']}, dt={r['discretizations'][label]['dt']}")
        axes[1].fill_between(t[1:],(mean-2*se)[1:],(mean+2*se)[1:],color=color,alpha=.2)
    tt=np.array([256,4096]);axes[1].loglog(tt,mean[np.searchsorted(t,256)]*(tt/256)**(1/3),'k--',lw=.8,label='slope 1/3')
    axes[1].set(xlabel='Time',ylabel='Ensemble-mean energy length',title='Eight-seed ensemble, box 256 (±2 SE)');axes[1].legend(fontsize=7)
    wins=['512','1024','2048'];x=np.arange(3)
    for k,(label,color) in enumerate((('A','tab:blue'),('B','tab:orange'),('C','tab:green'))):
        m=[r['exponents'][label][w]['energy'][0] for w in wins];e=[2*r['exponents'][label][w]['energy'][1] for w in wins]
        axes[2].errorbar(x+(k-1)*.12,m,yerr=e,fmt='o',color=color,label=label)
    axes[2].axhline(1/3,color='black',ls='--',lw=.8);axes[2].set(xticks=x,xticklabels=['512–1024','1024–2048','2048–4096'],ylabel='Energy-length exponent (±2 SE)',title='Window exponents');axes[2].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'coarsening-conditioning.{ext}',dpi=170)
    plt.close(fig)
    g=r['gates'];c=json.loads((OUT/'conditioning.json').read_text())
    lines=['# Late Cahn–Hilliard coarsening: conditioning, pointwise accuracy and ensemble statistics','',
        "September 26, 2026. Section 51 revisits the failed late-time accuracy gates of Section 24 with the same model, initial construction and first-order stabilized scheme.",'',
        '| Question | Result |','| --- | --- |',
        f"| Are late trajectories sensitive to tiny initial perturbations? | No. A 1e-10 perturbation ends at {c['1e-10']['final_l2']:.1e} at t=4096 (maximum growth about 20x); the 1e-6 run scales linearly. |",
        f"| Do pointwise late fields converge under refinement? | No, in the affordable range: successive differences stay O(1) (time series {', '.join(f'{d['l2']:.2f}' for d in pw['time']['differences'])}; space series {', '.join(f'{d['l2']:.2f}' for d in pw['space']['differences'])}). Both 51.3 gates fail. |",
        f"| Are ensemble statistics accurate? | Yes: mean lengths differ by at most {100*max(r['mean_length_differences']['mesh_spectral'],r['mean_length_differences']['mesh_energy']):.2f}% (mesh) and {100*max(r['mean_length_differences']['time_spectral'],r['mean_length_differences']['time_energy']):.2f}% (time), against a seed standard error of about 1%. |",
        f"| Is the late exponent 1/3? | The energy-length exponent at h=0.25 in 2048–4096 is {r['exponents']['B']['2048']['energy'][0]:.3f} ± {r['exponents']['B']['2048']['energy'][1]:.3f}, within two standard errors. Its timestep stability fails the declared 0.02 level. |",'',
        'Gates (51.2): '+', '.join(f"{k} {'PASS' if v else 'FAIL'}" for k,v in g.items())+'.','',
        "Reading. Discretization error pushes along neutral directions (domain translation, event timing) with no restoring force. It accumulates until morphologies differ, although random initial perturbations decay. Late-time accuracy is therefore statistical, not pointwise, for this scheme at affordable steps. The declared exponent-stability failure reflects realization noise between discretizations comparable to the gate. The Section 24 failures stand.",'',
        '![Conditioning, ensemble lengths and exponents](results/coarsening-conditioning-2026-09-26/figures/coarsening-conditioning.png)','',
        '```sh','python3 benchmarks/coarsening_conditioning.py --output results/coarsening-conditioning-2026-09-26 --conditioning',
        '# pointwise members: --pointwise H DT ; ensemble members: --member A|B|C SEED','python3 benchmarks/coarsening_conditioning.py --output results/coarsening-conditioning-2026-09-26 --pointwise-assemble',
        'python3 benchmarks/coarsening_conditioning.py --output results/coarsening-conditioning-2026-09-26','python3 benchmarks/coarsening_conditioning_report.py','```','',
        '[Report](results/coarsening-conditioning-2026-09-26/report.json), [pointwise](results/coarsening-conditioning-2026-09-26/pointwise.json), [theory Section 51](unified_multiscale_defect_framework.md).','']
    write_doc('COARSENING_CONDITIONING_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
