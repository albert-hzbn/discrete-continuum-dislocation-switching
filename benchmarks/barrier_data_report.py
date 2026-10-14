"""Sections 72-73 and 85 report: computed embryo energetics and junction strengths for copper, from saved evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1];K_B=8.617333262e-5
mark=lambda b:'PASS' if b else 'FAIL'


def mishin_section(e):
    """Section 85: the Section 72 protocol with the Mishin (2001) potential, and the Section 76 rates recomputed from it."""
    p=ROOT/'results/embryo-atomistics-mishin-2026-09-28/report.json'
    if not p.exists():return []
    m=json.loads(p.read_text());ne=json.loads((ROOT/'results/nucleation-rates-2026-09-28/report.json').read_text())
    nm=json.loads((ROOT/'results/nucleation-rates-mishin-2026-09-28/report.json').read_text());m3=json.loads((ROOT/'results/nucleation-rates-mishin-2026-09-28/m3.json').read_text())
    Ee={int(k):v for k,v in e['E_f_eV'].items()};Em={int(k):v for k,v in m['E_f_eV'].items()};n=np.array(sorted(Em),float)
    fig,ax=plt.subplots(figsize=(6.4,4.2),layout='constrained')
    for E,lab in ((Ee,'EMT'),(Em,'Mishin 2001')):ax.plot(n,[E[int(k)]/k**(2/3) for k in n],'o-',ms=3,label=lab)
    ax.set(xscale='log',xlabel='Vacancies in cluster n',ylabel='E_f(n) / n^2/3 (eV)',title='Void-embryo energies: EMT against Mishin');ax.legend(fontsize=8)
    d=ROOT/'results/embryo-atomistics-mishin-2026-09-28/figures';d.mkdir(exist_ok=True)
    for ext in ('png','svg'):fig.savefig(d/f'embryo-emt-mishin.{ext}',dpi=150)
    plt.close(fig);g=dict(m['gates']);g['M3_full_law_cnt']=m3['M3_full_law_cnt']
    names={'M1_fidelity':'M1 a₀ within 0.1%, E_f(1) within 5% of 1.287 eV','M2_capillarity':'M2 capillary fit R² ≥ 0.999 (n ≥ 13)','M3_full_law_cnt':'M3 full-law CNT within 1.5x of Becker–Döring wherever n* ≥ 19'}
    L=['## Embryos and nucleation rates with the Mishin potential (Section 85)','',
       'The Section 72 protocol is rerun unchanged with the tabulated Mishin et al. (2001) potential (theory 82.6, downloaded from NIST with the user\'s permission), and the Section 76 rates are recomputed from these energies. The gates were declared before the run; the A-gates of Section 72 are repeated in the report for comparison.','',
       '| Gate | Outcome |','| --- | --- |']+[f'| {names[k]} | {mark(g[k])} |' for k in names]
    L+=['','| Quantity | EMT | Mishin | Measured |','| --- | ---: | ---: | ---: |',
        f"| E_f(1) (eV) | {Ee[1]:.3f} | {Em[1]:.3f} | 1.27–1.29 |",
        f"| Surface (111), (100), (110) (J/m²) | "+', '.join(f'{v:.2f}' for v in e['surfaces_J_m2'].values())+' | '+', '.join(f'{v:.2f}' for v in m['surfaces_J_m2'].values())+' | about 1.75 (mean) |',
        f"| γ_eff from the cluster fit (J/m²) | {e['fit']['gamma_eff_J_m2']:.3f} | {m['fit']['gamma_eff_J_m2']:.3f} | |",
        f"| Fit R² (n ≥ 13) | {e['fit']['r2']:.4f} | {m['fit']['r2']:.4f} | |"]
    for S in ('10','100','1000','10000'):
        a,b=e['barriers'][S]['post_hoc_discrete'],m['barriers'][S]['post_hoc_discrete'];L.append(f"| ΔG* at S = {S}, 800 K (eV) | {a['barrier_eV']:.2f} (n* {a['n_star']:.0f}) | {b['barrier_eV']:.2f} (n* {b['n_star']:.0f}) | |")
    L.append('| Onset S_on (J_BD = 1 m⁻³ s⁻¹) at 600, 700, 800, 900, 1000 K | '+', '.join(f'{v:.0f}' for v in ne['onset_S'].values())+' | '+', '.join(f'{v:.0f}' for v in nm['onset_S'].values())+' | |')
    rows=nm['post_hoc_full_capillary_cnt'];L+=['','Full-law CNT against Becker–Döring with the Mishin energies: '+'; '.join(f"{r['T']:.0f} K, S = {r['S']:g}: n* = {r['n_star']:.1f}, ratio {r['ratio']:.3g}" for r in rows)+'.',
        '',f"The Mishin potential reduces the surface-energy deficit from about 35% (EMT) to 23% and matches the monovacancy formation energy to 1%. Barriers rise by about 1.6x and the onset supersaturation by 3–4.6x, which is the calibration effect of the surface energy. M2 fails for the same reason as A2: non-magic cluster shapes scatter about the smooth capillary law. M3 fails at one point only. At 700 K and S = 10⁴, n* = 19.2 sits at the declared threshold, and the ratio is 2.93. Every point with n* ≥ 21 agrees within 0.70–1.00. The boundary of capillary validity found with EMT (n* ≈ 19) moves to about n* ≈ 21 with this potential. The discrete Becker–Döring flux remains the rate to use below it.",
        '','![Embryo energies, EMT against Mishin](results/embryo-atomistics-mishin-2026-09-28/figures/embryo-emt-mishin.png)','']
    bp=ROOT/'results/surface-energy-band-2026-09-28/band.json'
    if bp.exists():
        bd=json.loads(bp.read_text());Ts=list(bd['onset_atomistic'])
        L+=['## The surface-energy band (Section 89)','',f"The Mishin surfaces lie about 23% below the commonly quoted 1.75 J/m², which is itself extrapolated from liquid surface tension; semilocal DFT is reported near the embedded-atom values. A second closure rescales every embryo's surface term to 1.75 J/m² (factor {bd['scale']:.3f}) and recomputes the Becker–Döring onset. The two closures bracket the criterion:",'',
            '| T (K) | '+' | '.join(Ts)+' |','| --- |'+' ---: |'*len(Ts),'| S_on, atomistic γ | '+' | '.join(f'{v:.3g}' for v in bd['onset_atomistic'].values())+' |','| S_on, γ = 1.75 J/m² | '+' | '.join(f'{v:.3g}' for v in bd['onset_extrapolated'].values())+' |',
            '',"At 800 K the barriers rise by a factor of 2.3–2.8 (the γ³ estimate is 2.4). The band, a factor of 12 at 1000 K and 230 at 600 K, is the stated uncertainty of the nucleation closure.",'','```sh','python3 benchmarks/surface_energy_band.py','```','']
    return L


def main():
    e=json.loads((ROOT/'results/embryo-atomistics-2026-09-27/report.json').read_text());j=json.loads((ROOT/'results/junction-strengths-2026-09-27/report.json').read_text())
    fig_dir=ROOT/'results/embryo-atomistics-2026-09-27/figures';fig_dir.mkdir(exist_ok=True)
    Ef={int(k):v for k,v in e['E_f_eV'].items()};n=np.array(sorted(Ef),float);E=np.array([Ef[int(k)] for k in n]);f=e['fit'];kT=K_B*e['T']
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained');nn=np.linspace(13,150,300)
    axes[0].plot(n,E,'ko',ms=4,label='EMT molecular statics');axes[0].plot(nn,f['A']*nn**(2/3)+f['B']*nn**(1/3)+f['C'],'C0-',label='A n^2/3 + B n^1/3 + C (n ≥ 13)')
    axes[0].set(xlabel='Vacancies in cluster n',ylabel='E_f(n) (eV)',title='Void-embryo formation energy, EMT copper');axes[0].legend(fontsize=8)
    for i,S in enumerate(('1000','10000')):
        axes[1].plot(n,E-n*kT*np.log(float(S)),'o',color=f'C{i}',ms=4,label=f'S = {S}, atomistic points')
        g=e['fit']['gamma_eff_J_m2']/(1.602176634e-19*1e20);om=e['omega_A3'];m=np.linspace(1,60,300)
        axes[1].plot(m,(36*np.pi)**(1/3)*om**(2/3)*g*m**(2/3)-m*kT*np.log(float(S)),'--',color=f'C{i}',label=f'S = {S}, CNT with γ_eff')
    axes[1].set(xlabel='n',ylabel='ΔG(n) = E_f(n) - n kT ln S (eV)',title='Embryo free energy at 800 K',xlim=(0,60),ylim=(-2,8));axes[1].legend(fontsize=7)
    for ext in ('png','svg'):fig.savefig(fig_dir/f'embryo-energetics.{ext}',dpi=150)
    plt.close(fig)
    g=e['gates'];lines=['# Computed barrier data for copper: void embryos and dislocation junctions','',
        'September 27, 2026. Sections 72 and 73 supply the barrier data that Section 69 left open.',
        '','## Void embryos by molecular statics (Section 72)','',
        'Potential: EMT copper (ASE). Clusters: compact voids of n vacancies in a 2048-atom supercell, relaxed to 0.01 eV/Å.','',
        '| Gate | Outcome |','| --- | --- |']+[f'| {k} | {mark(v)} |' for k,v in g.items()]
    lines+=['',f"The lattice constant is {e['a0_angstrom']:.4f} Å (measured 3.615 Å) and E_f(1) is {Ef[1]:.3f} eV (Section 69: 1.287 eV). Relaxed flat-surface energies are "+', '.join(f"({k}) {v:.3f}" for k,v in e['surfaces_J_m2'].items())+f" J/m², mean {e['surface_mean_J_m2']:.3f}, against a measured value of about 1.75. EMT underestimates surface energy by about 35%, a known limitation of the potential.",
        '',f"The capillary fit for n ≥ 13 is E_f = {f['A']:.3f} n^2/3 + {f['B']:.3f} n^1/3 {f['C']:+.3f} eV, with R² = {f['r2']:.4f}, and it gives γ_eff = {f['gamma_eff_J_m2']:.3f} J/m². A2 fails on R², because the non-magic cluster shapes scatter about the smooth law, although γ_eff lies within 4% of the flat-surface mean.",
        '','| S (800 K) | n* | ΔG* declared (fit curve), eV | ΔG* with computed n ≤ 24 (post hoc), eV | ΔG*/kT (post hoc) | CNT(γ_eff) / atomistic (post hoc) |','| ---: | ---: | ---: | ---: | ---: | ---: |']
    for S,b in e['barriers'].items():
        p=b['post_hoc_discrete'];lines.append(f"| {S} | {p['n_star']:.0f} | {b['barrier_eV']:.2f} | {p['barrier_eV']:.2f} | {p['barrier_over_kT']:.1f} | {p['cnt_ratio']:.2f} |")
    lines+=['','The declared A3 fails at S = 10⁴ because the declared method evaluated the n ≥ 13 fit at n* ≈ 9, outside its range. With the directly computed clusters n = 1–24, a labelled post hoc evaluation, continuum CNT with γ_eff reproduces the atomistic barrier within 6% at S = 10³ and 10⁴. So capillary theory is adequate down to about 10 vacancies once γ is taken from the same potential. For copper itself, the barrier scales as γ³: with the measured γ ≈ 1.75 J/m², barriers are about (1.75/1.09)³ ≈ 4.1 times the EMT values. The Section 69 CNT numbers are the experimental-γ estimate, and these atomistic numbers bound the potential dependence.',
        '','These values calibrate the Section 58 closure: θ_T = k_B T, the embryo radius R_e from n* (R* in the report), and ΔF_ins from ΔG*. A first run with a sign error in ΔG(n) (the n E_f(1) term counted twice) was caught before interpretation, and it is kept under first-pass/.',
        '',f'![Embryo formation energy and free energy](results/embryo-atomistics-2026-09-27/figures/embryo-energetics.png)',
        '']+mishin_section(e)+['## Junction strengths (Section 73)','','| Gate | Outcome |','| --- | --- |',f"| J1: the model's Frank-criterion ranking matches the DDD ranking | {mark(j['gates']['J1_ranking'])} |",
        '','| Reaction | DDD a_ij (Kubin, Devincre, Hoc 2008) | Model Frank gain G (a²) | Taylor τ at ρ = 10¹² m⁻² (MPa) |','| --- | ---: | ---: | ---: |']
    for k in ('collinear','glissile','Lomer','Hirth','self','coplanar'):
        lines.append(f"| {k} | {j['a_ij'][k]} | {j['frank_gain_a2'].get(k,'–')} | {j['taylor_MPa_by_forest_density']['1e+12'][k]:.2f} |")
    lines+=['',"The model's core line energy κ|b|² (Section 54) ranks the reactions collinear > Lomer = glissile > Hirth, as DDD does. It cannot separate Lomer from glissile, which have the same junction |b|; that distinction needs anisotropic line tension, and DDD separates the two by only 12%. The junction closure is therefore calibrated to the DDD coefficients, with the copper DDD parameters μ = 54.6 GPa, ν = 0.324, b = 0.255 nm and B = 1.56e-5 Pa·s from the same study.",
        '','Sources: Kubin, Devincre and Hoc, Acta Mater. 56, 6040 (2008), via Akhondzadeh, Sills, Bertin and Cai, [arXiv:2006.08847](https://arxiv.org/abs/2006.08847), Table 1; Madec et al., [Science 301, 1879 (2003)](https://www.science.org/doi/10.1126/science.1085477).',
        '','## Interface events: slip transmission at a copper Σ3 twin boundary (Section 73.2)','',
        '| Quantity | Value | Source |','| --- | --- | --- |',
        '| Transmission stress, pristine boundary (micropillars) | about 17 MPa | Malyar, Grabowski, Dehm, Kirchlechner, Acta Mater. 161, 412 (2018), [ADS](https://ui.adsabs.harvard.edu/abs/2018AcMat.161..412M/abstract) |',
        '| Strain-rate sensitivity of ideal transmission | 0.015 ± 0.009 | same study |',
        '| Transmission stress, perfect boundary (MD) | more than 200 MPa (MD strain-rate upper bound) | compiled in [arXiv:2601.19543](https://arxiv.org/abs/2601.19543) |',
        '','The transmission closure is calibrated in critical-stress form, τ_tr between 17 and 200 MPa, with the measured rate sensitivity. The zero-stress activation energy of Zhu et al. ([PNAS 104, 3031, 2007](https://www.pnas.org/doi/10.1073/pnas.0611097104)) could not be read from any source reachable here without bypassing a paywall or CAPTCHA. Section 73.3 derives it instead from the thermally activated flow relation, with every assumed input carried as a range ([twin_activation.py](benchmarks/twin_activation.py)): ΔG at the operating stress is 0.47–0.70 eV, ΔG₀ is 0.49–0.83 eV, and the activation volume is 16–73 b³. That volume is cross-slip-like, consistent with the measured interpretation.',
        '','```sh','python3 benchmarks/embryo_atomistics.py --output results/embryo-atomistics-2026-09-27','python3 benchmarks/junction_strengths.py --output results/junction-strengths-2026-09-27','python3 benchmarks/embryo_atomistics.py --output results/embryo-atomistics-mishin-2026-09-28 --potential mishin  # Section 85','python3 benchmarks/nucleation_rates.py --embryo results/embryo-atomistics-mishin-2026-09-28/report.json --output results/nucleation-rates-mishin-2026-09-28  # Section 85','python3 benchmarks/barrier_data_report.py','```','']
    write_doc('BARRIER_DATA_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
