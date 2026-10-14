"""Worked calibration of the Section 50 vacancy closure for copper, with uncertainty propagation."""
import json
from pathlib import Path
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
K_B=8.617333262e-5          # eV/K (CODATA 2018)
EV=1.602176634e-19          # J/eV (exact SI)
# Verified: J. Phys. F: Met. Phys. 6 (1976), doi:10.1088/0305-4608/6/8/003 (quenched copper).
E_F,DE_F=1.29,.02           # eV, positron annihilation value quoted there
E_F_RES,DE_F_RES=1.27,.05   # eV, quenched-in resistivity in the same paper
E_M,DE_M=.74,.08            # eV, first-stage vacancy migration
# Commonly cited room-temperature constants (secondary source: Wikipedia "Copper"); temperature softening is an assumed band.
LATTICE=.36149e-9           # m
SHEAR=48e9                  # Pa
SHEAR_BAND=(.75,1.)         # assumed high-temperature softening factor range, not sourced
ENTROPY_BAND=(1.,2.)        # assumed S_f/k_B range, not sourced here
ATTEMPT=1e13                # s^-1, assumed order of magnitude
SYNTHETIC=dict(theta=.02,equilibrium=.05)  # Section 50 values in units of the model shear modulus


def row(T):
    kT=K_B*T;omega=LATTICE**3/4;theta_over_mu=kT*EV/(omega*SHEAR)
    reduced=E_F/kT;x_low=np.exp(ENTROPY_BAND[0]-reduced);x_high=np.exp(ENTROPY_BAND[1]-reduced)
    diffusivity=LATTICE**2/2*ATTEMPT*np.exp(-E_M/kT)
    return dict(T=T,kT_eV=kT,theta_over_mu=theta_over_mu,theta_over_mu_softened=[theta_over_mu/SHEAR_BAND[1],theta_over_mu/SHEAR_BAND[0]],
        formation_over_theta=[reduced-ENTROPY_BAND[1],reduced-ENTROPY_BAND[0]],x_eq=[x_low,x_high],
        x_eq_factor_from_E_f=float(np.exp(DE_F/kT)),x_eq_ratio_resistivity_vs_positron=float(np.exp((E_F-E_F_RES)/kT)),
        diffusivity_m2_s=diffusivity,diffusivity_factor_from_E_m=float(np.exp(DE_M/kT)))


def main():
    rows=[row(T) for T in (800.,1000.,1200.)];omega=LATTICE**3/4
    equivalent_T=SYNTHETIC['theta']*omega*SHEAR/(K_B*EV)
    synthetic_ratio=np.log((1-SYNTHETIC['equilibrium'])/SYNTHETIC['equilibrium'])
    at=row(equivalent_T)
    out=dict(sources=dict(formation_migration='J. Phys. F: Met. Phys. 6 (1976), doi:10.1088/0305-4608/6/8/003',constants='Wikipedia "Copper" (lattice constant, shear modulus), secondary',
        assumed=dict(entropy_over_k=ENTROPY_BAND,shear_softening=SHEAR_BAND,attempt_frequency=ATTEMPT)),
        atomic_volume_m3=omega,rows=rows,synthetic=dict(SYNTHETIC,formation_over_theta=float(synthetic_ratio),
            equivalent_temperature_K=float(equivalent_T),physical_formation_over_theta_at_equivalent_T=at['formation_over_theta'],
            physical_x_eq_at_equivalent_T=at['x_eq']))
    folder=ROOT/'results/calibration-example-2026-09-26';folder.mkdir(parents=True,exist_ok=True)
    (folder/'calibration.json').write_text(json.dumps(out,indent=1)+'\n')
    lines=['# Worked calibration of the vacancy closure for copper','',
        "September 26, 2026. This worked example maps published copper vacancy data onto the nondimensional Section 50 closure psi_v = e_f x + theta[x ln x + (1-x)ln(1-x)], with energies in units of the shear modulus mu (the model's mu=1), and propagates the stated uncertainties. It demonstrates the calibration procedure of theory Section 14 and PARAMETER_REGISTRY.md; the Section 50 runs are not recalibrated.",'',
        '## Inputs','','| Quantity | Value | Status |','| --- | --- | --- |',
        f'| Vacancy formation energy E_f | {E_F} ± {DE_F} eV (positron); {E_F_RES} ± {DE_F_RES} eV (resistivity) | verified: J. Phys. F 6 (1976), [doi:10.1088/0305-4608/6/8/003](https://iopscience.iop.org/article/10.1088/0305-4608/6/8/003) |',
        f'| Migration energy E_m | {E_M} ± {DE_M} eV | verified: same paper |',
        f'| Lattice parameter, shear modulus | {LATTICE*1e9:.5f} nm, {SHEAR/1e9:.0f} GPa (room temperature) | secondary: [Wikipedia, Copper](https://en.wikipedia.org/wiki/Copper) |',
        f'| Formation entropy S_f/k_B | {ENTROPY_BAND[0]}–{ENTROPY_BAND[1]} | assumed band, not sourced |',
        f'| High-temperature shear softening | factor {SHEAR_BAND[0]}–{SHEAR_BAND[1]} | assumed band |',
        f'| Attempt frequency | {ATTEMPT:.0e} s^-1 | assumed order of magnitude |','',
        '## Mapping','',"With atomic volume Omega = a³/4, the closure's parameters are theta = k_B T/Omega and e_f = (E_f - T S_f)/Omega, giving x_eq = exp(S_f/k_B - E_f/(k_B T)) in the dilute limit. In model units theta/mu = k_B T/(Omega mu).",'',
        '| T (K) | theta/mu (room-T mu) | theta/mu (softened mu) | e_f/theta | x_eq | x_eq factor from ±δE_f | D_v (m²/s) | D_v factor from ±δE_m |','| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['T']:.0f} | {r['theta_over_mu']:.4f} | {r['theta_over_mu_softened'][0]:.4f}–{r['theta_over_mu_softened'][1]:.4f} | {r['formation_over_theta'][0]:.2f}–{r['formation_over_theta'][1]:.2f} | {r['x_eq'][0]:.1e}–{r['x_eq'][1]:.1e} | {r['x_eq_factor_from_E_f']:.2f} | {r['diffusivity_m2_s']:.1e} | {r['diffusivity_factor_from_E_m']:.1f} |")
    lines+=['','## What this says about the Section 50 synthetic values','',
        f"The synthetic mixing scale theta = {SYNTHETIC['theta']} mu corresponds to copper at about {equivalent_T:.0f} K. The mechanical-to-configurational energy ratio is therefore physically plausible. The synthetic equilibrium fraction x_eq = {SYNTHETIC['equilibrium']} (e_f/theta = {synthetic_ratio:.2f}) is not: at that temperature copper has e_f/theta between {at['formation_over_theta'][0]:.1f} and {at['formation_over_theta'][1]:.1f} and x_eq of {at['x_eq'][0]:.0e} to {at['x_eq'][1]:.0e}, lower by roughly six orders of magnitude. The synthetic closure inflates the vacancy population so that coupling effects are visible at nondimensional scale and the sink relaxation is not stiff. The Section 50 notch effects (slip -30%, supersaturation 12–13%) must therefore not be read as copper predictions.",'',
        "Uncertainty ranking: the unsourced formation-entropy band changes x_eq by a factor of e (2.7). The ±0.02 eV formation-energy uncertainty changes it by a factor of 1.2–1.3. The ±0.08 eV migration-energy uncertainty changes D_v by a factor of 2–3. The 0.02 eV difference between the resistivity and positron values is within their stated errors. A predictive calibration would additionally need the generation coefficient chi (vacancies per slip, from jog-dragging data), sink strengths and the binding b, none of which is available here.",'',
        '```sh','python3 benchmarks/calibration_example.py','```','','[calibration.json](results/calibration-example-2026-09-26/calibration.json)','']
    write_doc('CALIBRATION_EXAMPLE.md','\n'.join(lines));print(json.dumps(out['synthetic']),flush=True)


if __name__=='__main__':main()
