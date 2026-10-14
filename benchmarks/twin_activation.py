"""Section 73.3: activation energy and volume of slip transmission at a copper Σ3 twin boundary, derived from measured and
MD stresses by the thermally activated flow relation, with every assumed input carried as a range."""
import itertools,json
from pathlib import Path
import numpy as np

K_B=8.617333262e-5;EV=1.602176634e-19;B=.2556e-9;T=293.
TAU_EXP=17e6            # transmission stress, pristine Σ3 boundary, room temperature (Malyar et al. 2018)
TAU_ATH=(200e6,300e6)   # athermal bound from MD (more than 200 MPa); upper value assumed
RATE0=(1e6,1e8)         # attempt strain rate, s^-1 (assumed range)
RATE=(1e-4,1e-2)        # laboratory strain rate, s^-1 (assumed range)
Q=(1.,2.)               # Kocks exponent q in dG = dG0 (1 - tau/tau_ath)^q (p = 1)

if __name__=='__main__':
    rows=[]
    for tau_ath,r0,r,q in itertools.product(TAU_ATH,RATE0,RATE,Q):
        dG=K_B*T*np.log(r0/r);dG0=dG/(1-TAU_EXP/tau_ath)**q
        V=q*dG0*EV*(1-TAU_EXP/tau_ath)**(q-1)/tau_ath/B**3
        rows.append(dict(tau_ath_MPa=tau_ath/1e6,rate0=r0,rate=r,q=q,dG_at_17MPa_eV=float(dG),dG0_eV=float(dG0),activation_volume_b3=float(V)))
    g0=[x['dG0_eV'] for x in rows];v=[x['activation_volume_b3'] for x in rows];g=[x['dG_at_17MPa_eV'] for x in rows]
    report=dict(scope='Section 73.3: derived, not computed; every assumed input is a range.',T_K=T,tau_exp_MPa=TAU_EXP/1e6,
        dG_at_operating_stress_eV=[min(g),max(g)],dG0_eV=[min(g0),max(g0)],activation_volume_b3=[min(v),max(v)],rows=rows)
    out=Path('results/twin-activation-2026-09-27');out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(report,indent=1)+'\n')
    print({k:[round(x,3) for x in report[k]] for k in ('dG_at_operating_stress_eV','dG0_eV','activation_volume_b3')})
