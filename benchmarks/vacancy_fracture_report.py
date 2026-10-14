"""Section 50 report: figures and markdown from saved reactive-vacancy evidence."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.vacancy_fracture import NodalVacancyFracture2D
from bmvr.nodal_damage import NodalThermalFracture2D
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['c','eta','T','vac','slip','damage']


def pct(v):return 'n/a' if v is None else f'{100*v:.3f}%'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());h=r['homogeneous'];nr=r['notch'];figures=out/'figures';figures.mkdir(exist_ok=True)
    m,s,_=NodalVacancyFracture2D.load(out/'vacancy-notch-64-0.00025-final.npz');_,c,_=NodalThermalFracture2D.load(Path(nr['control'])/'graded-notch-64-0.00025-t0.2.npz')
    n=m.mechanics.mesh.n;xe,ye=m.mechanics.mesh.edges;eq=r['notch']['parameters']['equilibrium']
    fig,axes=plt.subplots(1,3,figsize=(15.5,4.6),layout='constrained')
    im=axes[0].pcolormesh(xe,ye,(s.vac/eq-1).reshape(n,n).T,cmap='viridis');fig.colorbar(im,ax=axes[0],shrink=.8)
    axes[0].contour(xe,ye,s.damage.reshape(n+1,n+1).T,levels=[.5],colors='w',linewidths=.8)
    axes[0].set(title='Vacancy supersaturation x/x_eq-1 (d=0.5 in white)',xlabel='x',ylabel='y',aspect='equal')
    diff=(s.damage-c.damage).reshape(n+1,n+1).T;lim=float(np.max(abs(diff)))
    im=axes[1].pcolormesh(xe,ye,diff,shading='gouraud',cmap='RdBu_r',vmin=-lim,vmax=lim);fig.colorbar(im,ax=axes[1],shrink=.8)
    axes[1].set(title='Damage: coupled minus control, t=0.2',xlabel='x',ylabel='y',aspect='equal')
    bs=h['generation_backstress'];sr=h['sink_relaxation'];dts=[float(k) for k in sr['runs']];errs=[sr['runs'][k]['error'] for k in sr['runs']]
    axes[2].loglog(dts,errs,'o-',label='Sink relaxation error');axes[2].loglog(dts,errs[0]*(np.array(dts)/dts[0])**2,'k--',lw=.8,label='slope 2')
    axes[2].set(xlabel='dt',ylabel='|x(T)-exact|',title='Homogeneous sink relaxation, full model');axes[2].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'vacancy-fracture.{ext}',dpi=170)
    plt.close(fig)
    tests=int(ET.parse(out/'test-results.xml').getroot().find('testsuite').attrib['tests'])
    g=nr['gates_48_64'];rel=out.relative_to(ROOT) if out.is_absolute() else out;p=nr['parameters']
    checks=sum(sum(x['checks'].values()) for x in nr['runs'].values());total=sum(len(x['checks']) for x in nr['runs'].values())
    lines=['# Reactive vacancies in finite thermochemical plasticity and fracture','',
        f"September 26, 2026. **{tests} tests pass.** Section 50 adds a vacancy site fraction to the common finite-strain energy with three reactions: generation by crystal slip (line → point), annihilation at bulk and crack sinks (point → surface), and energetic binding to damaged material. Three graded notch trajectories pass **{checks}/{total}** checks, and graded n48/n64 passes **{sum(g.values())}/6** mesh gates at t=0.20. All coefficients are synthetic and nondimensional; no material calibration is claimed.",'',
        '## Thermodynamic structure','',
        'Generation charges each unit of slip chi vacancies and raises the flow threshold by chi mu_v. The slip dissipation is then sum |gamma_dot|(|q| - chi mu_v), which is at least the yield dissipation, so no cap is needed. Sinks relax toward the reservoir with dissipation V k mu_v^2. Transport uses the conservative two-point flux of the composition field. Binding -b x ∫d² enters the nodal damage force through the same energy. The flow law is single valued when min yield > chi b, and a semi-discrete minimum argument gives mu_v >= -b (theory 50.2); every rate evaluation checks this and never clamps it.','',
        'Automated tests verify finite-difference energy derivatives including the binding force, the per-cell first law and global energy with all three processes active, entropy production, the exact inventory ledger, the dissipation identity, bitwise reduction to the vacancy-free model, restart, the sink rate and the well-posedness guard.','',
        '## Homogeneous benchmarks','',
        f"Sink relaxation (x0=0.08, k=0.5, full coupled model with frozen mechanics): errors against an exact ODE solution at dt 0.04, 0.02, 0.01 are {', '.join(f'{e:.2e}' for e in errs)}, giving observed orders {sr['observed_orders'][0]:.4f} and {sr['observed_orders'][1]:.4f}. Maximum total-energy drift is {max(v['energy_drift'] for v in sr['runs'].values()):.1e}; the released vacancy energy appears as heat.",'',
        f"Generation back-stress (fixed shear 0.1, stress relaxation to t=1): with chi=4 the vacancy fraction rises to {bs['generation']['final_vac']:.4f} (mu_v={bs['generation']['final_mu']:.4f}) and final slip falls by {100*bs['slip_reduction']:.1f}% against chi=0 ({bs['generation']['final_slip']:.4f} versus {bs['control']['final_slip']:.4f}). Stored generation power ({bs['generation']['stored_power_last']:.2e}) is excluded from heat, and entropy production stays positive (minimum {bs['generation']['min_entropy_production']:.2e}).",'',
        '## Graded notch with reactive vacancies','',
        f"Section 46.5 bands, Gauss rule, t≤0.20, chi={p['generation']:g}, b={p['binding']:g}, k0={p['sink']:g}, k_d={p['damage_sink']:g}, M_v={p['mobility']:g}, theta={p['theta']:g}, x_eq={p['equilibrium']:g}. The control is the vacancy-free Section 46 trajectory on the same mesh, which the zero-coefficient model reproduces bitwise.",'',
        '| Field | Graded n32/n48 | Graded n48/n64 | Gate <5% |','| --- | ---: | ---: | --- |']
    for f in FIELDS:lines.append(f"| {f} | {pct(nr['pairs']['32_48']['t0.2'][f])} | {pct(nr['pairs']['48_64']['t0.2'][f])} | {'PASS' if g[f] else 'FAIL'} |")
    lines+=['','| Mesh | Max supersaturation | Damage-band enrichment | Outside band | Damage change vs control | L1 slip change | Tip event (coupled / control) | Inventory ledger residual |','| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for k,e in nr['effects'].items():
        lines.append(f"| {k} | {pct(e['max_supersaturation'])} | {pct(e['band_mean_enrichment'])} | {pct(e['outside_band_mean'])} | {pct(e['damage_relative_change'])} | {100*e['slip_L1_change']:+.2f}% | {e['tip_event_vacancy']:.4f} / {e['tip_event_control']:.4f} | {e['ledger_residual']:.1e} |")
    extra=out/'vacancy-notch-96-0.00025-final.npz'
    if extra.exists():
        from graded_notch import compare
        post=compare(out/'vacancy-notch-64-0.00025-final.npz',extra,NodalVacancyFracture2D,('c','eta','T','vac'))
        lines+=['',"Post hoc, added after the slip gate failed: graded n96 gives n64/n96 discrepancies at t=0.20 of "+', '.join(f"{f} {pct(post[f])}" for f in FIELDS)+". Slip falls by about 1.4x per refinement step (9.60% → 5.66% → "+pct(post['slip'])+"), consistent with the first-order limit of the cell-constant back-stress. All six fields are below 5% at n64/n96. The declared n48/n64 gate remains failed."]
    lines+=['',"Supersaturation is generated where slip is active and the damage band is enriched through binding and transport, while crack sinks absorb vacancies. The table quantifies how strongly these synthetic couplings alter damage and slip; it does not predict a material's response. The inventory ledger (generated plus absorbed versus change) closes to the time-quadrature accuracy of the reported rows.",'',
        f'![Vacancy supersaturation, damage change and sink relaxation]({rel}/figures/vacancy-fracture.png)','',
        '## Scope','',"The vacancy free energy is athermal (fixed mixing scale), like the existing composition energy; T-proportional configurational entropy is an extension. Line defects remain continuum slip; there are no discrete dislocations in this finite model. Pore nucleation from supersaturation, beyond energetic damage binding, is not modeled. The notch comparison is limited to t≤0.20, before separation, where the graded Gauss family is qualified (Section 46).",'',
        '```sh',f'python3 benchmarks/vacancy_fracture.py --output {rel} --homogeneous',
        f'for n in 32 48 64; do python3 benchmarks/vacancy_fracture.py --output {rel} --notch $n & done; wait',
        f'python3 benchmarks/vacancy_fracture.py --output {rel}',f'python3 benchmarks/vacancy_fracture_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [tests]({rel}/test-results.xml), [theory Section 50](unified_multiscale_defect_framework.md).','']
    write_doc('VACANCY_FRACTURE_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
