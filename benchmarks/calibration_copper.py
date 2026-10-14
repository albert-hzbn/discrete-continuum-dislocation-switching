"""Section 69: copper calibration of the closure families and an independent self-diffusion validation (declared gates)."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
K_B=8.617333262e-5          # eV/K (CODATA 2018)
EV=1.602176634e-19          # J/eV
# Vacancy data: J. Phys. F 6 (1976), doi:10.1088/0305-4608/6/8/003 (positron 1.29 +- 0.02, resistivity 1.27 +- 0.05; E_m 0.74 +- 0.08).
E_F_VALUES=((1.29,.02),(1.27,.05));E_M,DE_M=.74,.08
LATTICE=.36149e-9;SHEAR=48e9;POISSON=.34      # secondary (as in Section 50)
S_F=(1.,2.);S_M=(0.,2.);NU=(5e12,2e13)         # assumed bands, not sourced
CORRELATION=.7815                              # fcc vacancy tracer correlation factor
# Independent validation target: Maier (1977), low-temperature tracer self-diffusion of copper.
MAIER=dict(D0=.68e-4,Q=2.17,dQ=.02)
T_VALIDATION=(700.,800.,900.,1000.)
# Interfaces: Cu triple-line study (Acta Mater., 2010): surface 1.75 J/m2 and grain boundary about 0.6 J/m2.
SIGMA_GB=.6;SIGMA_S=1.75
# Stacking fault energy band for pure Cu from the SFE literature (41-58 mJ/m2 quoted across studies).
SFE=(.041,.058)
# Phonon drag in typical metals at room temperature (Zbib, Hiratani, Shehadeh): 30-80 micro Pa s.
DRAG=(30e-6,80e-6)
SAMPLES=20000


def weighted_formation():
    v=np.array(E_F_VALUES);w=1/v[:,1]**2;return float(np.sum(w*v[:,0])/w.sum()),float(1/np.sqrt(w.sum()))


def diffusion_samples(T,rng):
    ef,def_=weighted_formation()
    Ef=rng.normal(ef,def_,SAMPLES);Em=rng.normal(E_M,DE_M,SAMPLES);Sf=rng.uniform(*S_F,SAMPLES);Sm=rng.uniform(*S_M,SAMPLES)
    nu=np.exp(rng.uniform(np.log(NU[0]),np.log(NU[1]),SAMPLES));kT=K_B*T
    x_eq=np.exp(Sf-Ef/kT);D_v=LATTICE**2*nu*np.exp(Sm)*np.exp(-Em/kT)
    return CORRELATION*x_eq*D_v,x_eq,D_v


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(20260927)
    ef,def_=weighted_formation();omega=LATTICE**3/4;b=LATTICE/np.sqrt(2)
    rows=[]
    for T in np.arange(600.,1351.,50.):
        D,x,Dv=diffusion_samples(T,rng);meas=MAIER['D0']*np.exp(-MAIER['Q']/(K_B*T))
        lo,med,hi=np.percentile(D,[2.5,50,97.5]);rows.append(dict(T=float(T),band=[float(lo),float(med),float(hi)],measured=float(meas),inside=bool(lo<=meas<=hi),
            x_eq_median=float(np.median(x)),D_v_median=float(np.median(Dv)),theta_over_mu=float(K_B*T*EV/(omega*SHEAR))))
    q_model=ef+E_M;dq_model=float(np.hypot(def_,DE_M));z=abs(q_model-MAIER['Q'])/np.hypot(dq_model,MAIER['dQ'])
    gates=dict(V1_band=all(r['inside'] for r in rows if r['T'] in T_VALIDATION),V2_activation=bool(z<=2))
    # Mapped closures (reported, no gates).
    xi=1e-9;k_gb=3*SIGMA_GB*xi/(2*np.sqrt(2));W_gb=3*SIGMA_GB/(2*np.sqrt(2)*xi)
    gamma_line=SHEAR*b**2/2
    mapped=dict(
        vacancy=dict(E_f_eV=[ef,def_],E_m_eV=[E_M,DE_M],theta_over_mu_at_821K=float(K_B*821*EV/(omega*SHEAR)),synthetic_theta_over_mu=.02,
            x_eq_at_821K=[float(np.exp(S_F[0]-ef/(K_B*821))),float(np.exp(S_F[1]-ef/(K_B*821)))],synthetic_x_eq=.05,
            note='D_v = a^2 nu exp(S_m) exp(-E_m/kT) for fcc; the Section 50 example used a^2/2 (a factor of 2 low).'),
        grain_boundary=dict(sigma_J_m2=SIGMA_GB,chosen_xi_m=xi,k_J_m=k_gb,W_J_m3=W_gb,W_over_mu=W_gb/SHEAR,
            note='Case 1 uses W = k = 1, so its model length unit is xi = 1 nm and its energy-density unit is W for this copper boundary.'),
        pore_surface=dict(sigma_J_m2=SIGMA_S,note='sigma = sqrt(2kW)/6 for the pore double well; at xi_p = 1 nm, W_p = 3 sigma/(sqrt(2) xi_p)',W_J_m3=3*SIGMA_S/(np.sqrt(2)*xi)),
        dislocation=dict(b_m=b,line_tension_J_m=gamma_line,frank_read_MPa_at_L={f'{L:g}':SHEAR*b/L/1e6 for L in (1e-7,1e-6,1e-5)},
            tau_over_mu_equals_b_over_L=True,stacking_fault_J_m2=list(SFE),dissociation_width_nm=[SHEAR*b**2/(24*np.pi*s)*(2+POISSON)/(1-POISSON)*1e9 for s in SFE],
            drag_Pa_s=list(DRAG),velocity_at_10MPa_m_s=[1e7*b/B for B in DRAG[::-1]]),
        void_nucleation={f'{S:g}':dict(r_star_nm=float(2*SIGMA_S*omega/(K_B*800*EV*np.log(S))*1e9),barrier_over_kT=float(16*np.pi*SIGMA_S**3*omega**2/(3*(K_B*800*EV*np.log(S))**2)/(K_B*800*EV)))
            for S in (10.,100.,1e3,1e4)})
    report=dict(scope='Section 69 copper calibration; declared gates V1, V2 (validation against an independent tracer measurement).',
        inputs=dict(E_f=[ef,def_],E_m=[E_M,DE_M],S_f_band=S_F,S_m_band=S_M,nu_band=NU,correlation=CORRELATION,lattice=LATTICE,shear=SHEAR,samples=SAMPLES,seed=20260927),
        validation=dict(target=MAIER,T_gated=T_VALIDATION,rows=rows,model_Q=[q_model,dq_model],z=float(z)),gates=gates,mapped=mapped)
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item() if hasattr(o,'item') else str(o))+'\n')
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,ax=plt.subplots(figsize=(6.4,4.4),layout='constrained')
    inv=np.array([1e4/r['T'] for r in rows]);band=np.array([r['band'] for r in rows])
    ax.fill_between(inv,band[:,0],band[:,2],color='C0',alpha=.25,label='calibrated closure, 95% band');ax.plot(inv,band[:,1],'C0-',lw=1,label='median')
    ax.plot(inv,[r['measured'] for r in rows],'k--',label='Maier 1977 tracer fit');ax.set_yscale('log')
    for T in T_VALIDATION:ax.axvline(1e4/T,color='0.8',lw=.6,zorder=0)
    ax.set(xlabel='10⁴ / T (K⁻¹)',ylabel='D_self (m²/s)',title='Copper self-diffusion: calibrated prediction vs measurement');ax.legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'copper-self-diffusion.{ext}',dpi=150)
    plt.close(fig);print(gates,'z',round(z,2),[(r['T'],r['inside']) for r in rows if r['T'] in T_VALIDATION])


if __name__=='__main__':main()
