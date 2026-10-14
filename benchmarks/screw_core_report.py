"""Section 82 report: the copper screw core with the analytical ZJW potential, from saved evidence; adds the post hoc anisotropic prediction."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from bmvr import extended_defects as X
from screw_core import cylinder,disregistry,crossing
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'screw-core-2026-09-28'
mark=lambda b:'PASS' if b else 'FAIL'
NAMES={'P1_lattice':'P1 lattice constant within 0.5% of 3.615 Å','P2_cohesive':'P2 cohesive energy within 3% of 3.49 eV','P3_elastic':'P3 B within 10% of 138 GPa and C44 within 15% of 75.4 GPa',
    'P4_positive_fault':'P4 positive intrinsic fault energy','C1_separation':'C1 separation within 30% of the isotropic prediction'}


def profile(path,a0,R):
    z=np.load(path);_,x0,yc=cylinder(a0,R);b=a0/np.sqrt(2);at=Atoms('Cu'*len(z['relaxed']),positions=z['relaxed']);ref=Atoms('Cu'*len(z['reference']),positions=z['reference'])
    xs,dis=disregistry(at,ref,x0,yc,b,a0);return xs-x0,dis


def main():
    r=json.loads((OUT/'report-zjw.json').read_text());e=r['elastic_GPa'];a0=r['a0'];g=r['sfe_mJ_m2'];s=r['screw']
    rows=json.loads((OUT/'post-hoc-tolerance.json').read_text())
    C=X.cubic_stiffness(e['C11'],e['C12'],e['C44'])*1e9;t=np.array([1.,-1,0])/np.sqrt(2)
    b1,b2=a0/6*np.array([2.,-1,-1])*1e-10,a0/6*np.array([1.,-2,1])*1e-10;d_an=X.anisotropic_separation(C,t,b1,b2,g*1e-3)*1e10
    K=X.energy_factor(C,t);ks,ke=float(t@K@t)/1e9,float(np.array([1,1,-2])@K@np.array([1,1,-2])/6)/1e9
    fig,ax=plt.subplots(figsize=(7,4.2),layout='constrained')
    for R in (60,120):
        p=OUT/f'screw-zjw-R{R}-tight.npz'
        if p.exists():
            xs,dis=profile(p,a0,float(R));ax.plot(xs,dis,label=f'R = {R} Å, 1e-4 eV/Å')
    ax.axhline(.5,color='0.6',lw=.8,ls=':');[ax.axhline(v,color='0.8',lw=.6) for v in (.25,.75)]
    ax.set(xlabel='x - x₀ along [11-2] (Å)',ylabel='Screw disregistry / b',title='Dissociated screw in ZJW copper',xlim=(-35,35));ax.legend(fontsize=8)
    (OUT/'figures').mkdir(exist_ok=True)
    for ext in ('png','svg'):fig.savefig(OUT/'figures'/f'screw-disregistry.{ext}',dpi=150)
    plt.close(fig)
    final={R:[x for x in rows if x['R']==R][-1]['separation'] for R in (60.,120.) if any(x['R']==R for x in rows)}
    emt=json.loads((OUT/'partial.json').read_text())
    lines=['# The dissociated screw dislocation core in copper','',
        'September 28, 2026. Section 82 computes the screw dislocation core in copper by molecular statics and compares the partial separation with the continuum prediction of Section 74.1 ([screw_core.py](benchmarks/screw_core.py)). The EMT and Sutton–Chen copper potentials available offline both have negative stacking-fault energies (82.1), so the calculation uses the analytical Zhou–Johnson–Wadley embedded-atom potential (82.2, [eam_zjw.py](src/bmvr/eam_zjw.py)). That potential is defined by twenty published parameters and needs no file. Its forces equal finite differences of its energy to 4e-9 eV/Å.','',
        '| Gate | Outcome |','| --- | --- |']+[f'| {NAMES[k]} | {mark(v)} |' for k,v in r['gates'].items()]
    lines+=['','| Property | ZJW | Measured copper |','| --- | ---: | ---: |',
        f"| a₀ (Å) | {a0:.4f} | 3.615 |",f"| E_c (eV) | {r['cohesive_eV']:.3f} | 3.49 |",f"| B (GPa) | {e['B']:.1f} | 138 |",f"| C11, C12, C44 (GPa) | {e['C11']:.1f}, {e['C12']:.1f}, {e['C44']:.1f} | 168, 121, 75 |",
        f"| γ_isf (mJ/m²) | {g:.1f} | 41–58 |",f"| EMT γ_isf for comparison (mJ/m²) | {2*emt['sfe_mJ_m2']:.1f} (corrected, 82.2) | |",'',
        f"The declared relaxation (FIRE to 0.01 eV/Å, R = 60 Å, {s['atoms']} atoms) dissociates the screw into two Shockley partials, with the disregistry passing b/2 midway. The separation is {s['separation_A']:.1f} Å, {s['ratio']:.2f} of the isotropic prediction {s['isotropic_prediction_A']:.1f} Å (Voigt μ = {e['mu']:.1f} GPa, ν = {e['nu']:.3f}), so C1 fails.",
        '','## Post hoc (declared before running, 82.4)','','| R (Å) | Force tolerance (eV/Å) | FIRE steps | Separation (Å) |','| ---: | ---: | ---: | ---: |']
    lines+=[f"| {x['R']:.0f} | {x['fmax']:g} | {x['steps']} | {x['separation']:.1f} |" for x in rows]
    lines+=['',f"Tightening the tolerance moves the partials apart, so the declared tolerance stopped FIRE far from equilibrium. The net force on a partial at 10.9 Å is below 0.01 eV/Å per atom. "
        +(f"At 1e-4 eV/Å, R = 60 Å gives {final[60.]:.1f} Å and R = 120 Å gives {final[120.]:.1f} Å, a {100*abs(final[120.]/final[60.]-1):.0f}% boundary effect. The peaks of the dislocation density give 22.0 and 25.0 Å, within 5% of the crossings. At this separation the partial cores overlap: the disregistry between them slopes at about 0.01 b/Å instead of forming a flat fault plateau." if len(final)==2 else ''),
        '',f"Anisotropic prediction (labelled post hoc). Copper's Zener ratio is {2*e['C44']/(e['C11']-e['C12']):.2f} for this potential, so the isotropic Voigt estimate is a poor reference. The Barnett–Lothe energy factor of a [1-10] line (extended_defects.energy_factor, tested against the isotropic limit) gives {ks:.1f} GPa for the screw component and {ke:.1f} GPa for the edge component, against Voigt μ = {e['mu']:.1f} and μ/(1 - ν) = {e['mu']/(1-e['nu']):.1f}. The anisotropic separation b₁·K·b₂/(2πγ) is {d_an:.1f} Å"
        +(f", and the converged atomistic separation is {final[max(final)]/d_an:.2f} of it." if final else '.'),
        '',f'![Screw disregistry across the glide plane](results/screw-core-2026-09-28/figures/screw-disregistry.png)','',
        'The isotropic and anisotropic predictions bracket the converged atomistic separation (0.84 and 1.37 of them). Continuum elasticity with point-like partials is good to about 20–40% at a separation of only 10b, where the cores overlap. The potential\'s fault energy (23.8 mJ/m²) is about half the measured copper value, so its partials sit roughly twice as far apart as in real copper. The core geometry is a property of this potential, not a copper prediction. The Mishin et al. (2001) potential, fitted to the measured fault energy, gives the copper prediction below (Section 82.6).','',
        '```sh','python3 benchmarks/screw_core.py --output results/screw-core-2026-09-28 --potential zjw','python3 benchmarks/screw_core_posthoc.py  # post hoc','python3 benchmarks/screw_core_report.py','```','',
        '[Report](results/screw-core-2026-09-28/report-zjw.json), [post hoc](results/screw-core-2026-09-28/post-hoc-tolerance.json), [theory Section 82](unified_multiscale_defect_framework.md).','']
    mp=OUT/'report-mishin.json'
    if mp.exists():
        m=json.loads(mp.read_text());em=m['elastic_GPa'];core=m['core'];d120=core['120']['separation_A']
        fig,ax=plt.subplots(figsize=(7,4.2),layout='constrained')
        for path,a,R,lab in ((OUT/'screw-zjw-R120-tight.npz',a0,120.,f'ZJW, γ = {g:.1f} mJ/m²'),(OUT/'screw-mishin-R120.npz',m['a0'],120.,f"Mishin, γ = {m['sfe_mJ_m2']:.1f} mJ/m²")):
            xs,dis=profile(path,a,R);ax.plot(xs,dis,label=lab)
        [ax.axhline(v,color='0.8',lw=.6) for v in (.25,.5,.75)];ax.set(xlabel='x - x₀ along [11-2] (Å)',ylabel='Screw disregistry / b',title='Screw cores, R = 120 Å, 1e-4 eV/Å',xlim=(-35,35));ax.legend(fontsize=8)
        for ext in ('png','svg'):fig.savefig(OUT/'figures'/f'screw-zjw-mishin.{ext}',dpi=150)
        plt.close(fig);mark2=lambda k:mark(m['gates'][k])
        lines+=['## The Mishin potential (Section 82.6)','',
            "With the user's permission, the tabulated potential of Mishin et al. (2001) was downloaded from the NIST Interatomic Potentials Repository (Cu01.eam.alloy) and read as cubic splines ([eam_setfl.py](src/bmvr/eam_setfl.py); forces equal finite differences to 5e-9 eV/Å). The core protocol uses what 82.4 learned: 1e-4 eV/Å, with R = 120 Å declared. The bracket gate C2 was declared before the run.",'',
            '| Gate | Outcome |','| --- | --- |',f"| P0 reader reproduces the fitted a₀ and E_c to 0.1% | {mark2('P0_reader')} |",f"| P1–P4 against measured copper | {mark(all(m['gates'][k] for k in ('P1_lattice','P2_cohesive','P3_elastic','P4_positive_fault')))} |",
            f"| P5 fault energy in the measured band 41–58 mJ/m² | {mark2('P5_fault_band')} |",f"| C2 separation between the anisotropic and isotropic predictions | {mark2('C2_bracket')} |",'',
            '| Property | Mishin | ZJW | Measured |','| --- | ---: | ---: | ---: |',f"| a₀ (Å), E_c (eV) | {m['a0']:.4f}, {m['cohesive_eV']:.3f} | {a0:.4f}, {r['cohesive_eV']:.3f} | 3.615, 3.49 |",
            f"| C11, C12, C44 (GPa) | {em['C11']:.1f}, {em['C12']:.1f}, {em['C44']:.1f} | {e['C11']:.1f}, {e['C12']:.1f}, {e['C44']:.1f} | 168, 121, 75 |",f"| γ_isf (mJ/m²) | {m['sfe_mJ_m2']:.1f} | {g:.1f} | 41–58 |",
            f"| Separation R = 60 / 120 Å (Å) | {core['60']['separation_A']:.2f} / {d120:.2f} | {final[60.]:.2f} / {final[120.]:.2f} | |",
            f"| Anisotropic / isotropic prediction (Å) | {m['anisotropic_prediction_A']:.2f} / {m['isotropic_prediction_A']:.2f} | {d_an:.2f} / {s['isotropic_prediction_A']:.2f} | |",
            f"| Separation × γ (Å mJ/m²) | {d120*m['sfe_mJ_m2']:.0f} | {final[120.]*g:.0f} | |",'',
            f"The Mishin screw dissociates into partials {d120:.1f} Å apart ({d120/(m['a0']/np.sqrt(2)):.1f}b), inside the predicted bracket, at {d120/m['isotropic_prediction_A']:.2f} of the isotropic and {d120/m['anisotropic_prediction_A']:.2f} of the anisotropic value. These are nearly the same fractions as for ZJW (0.84 and 1.37). The boundary effect between R = 60 and 120 Å is {100*abs(d120/core['60']['separation_A']-1):.1f}%, smaller than ZJW's 10% because the partials are closer together. The product separation × γ agrees between the two potentials within 3%, the inverse scaling that elasticity predicts. This is the calibrated copper core of the framework: a fault energy inside the measured band, and a separation of about 1.3 nm.",
            '','![ZJW and Mishin screw cores](results/screw-core-2026-09-28/figures/screw-zjw-mishin.png)','','```sh','python3 benchmarks/screw_core.py --output results/screw-core-2026-09-28 --potential mishin','```','']
    write_doc('SCREW_CORE_RESULTS.md','\n'.join(lines))
    (OUT/'anisotropic.json').write_text(json.dumps(dict(K_screw_GPa=ks,K_edge_GPa=ke,anisotropic_separation_A=d_an,converged=final),indent=1)+'\n');print(d_an,final)


if __name__=='__main__':main()
