"""Section 76: steady-state void nucleation rates for copper, exact Becker-Doring flux vs classical theory; onset criterion."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

K_B=8.617333262e-5;EV=1.602176634e-19;A0=3.615e-10;OMEGA=A0**3/4;E_M=.74;NU=1e13;E_F=1.287;S_F=1.5


def energies(path='results/embryo-atomistics-2026-09-27/report.json'):
    r=json.loads(Path(path).read_text());E={int(k):v for k,v in r['E_f_eV'].items()};f=r['fit']
    N=np.arange(1,20001);Ef=np.array([E[n] if n<=24 else f['A']*n**(2/3)+f['B']*n**(1/3)+f['C'] for n in N]);return N,Ef,f,r['omega_A3']


def rates(T,S,N,Ef,f,omega_a3):
    kT=K_B*T;dG=Ef-N*kT*np.log(S);Dv=A0**2*NU*np.exp(-E_M/kT);xeq=np.exp(S_F-E_F/kT);C1=S*xeq/OMEGA
    r=(3*N*OMEGA/(4*np.pi))**(1/3);beta=4*np.pi*r*Dv*C1
    logCeq=-np.log(OMEGA)-dG/kT;terms=-(np.log(beta)+logCeq);m=terms.max();logJ=-(m+np.log(np.sum(np.exp(terms-m))));J_bd=float(np.exp(logJ))
    g=f['gamma_eff_J_m2']/(EV*1e20);Ae=(36*np.pi)**(1/3)*(omega_a3)**(2/3)*g;Dm=kT*np.log(S)
    nstar=(2*Ae/(3*Dm))**3;dGs=Ae*nstar**(2/3)-nstar*Dm;d2=abs(-(2/9)*Ae*nstar**(-4/3));Z=np.sqrt(d2/(2*np.pi*kT))
    bstar=4*np.pi*(3*nstar*OMEGA/(4*np.pi))**(1/3)*Dv*C1;J_cnt=float(Z*bstar/OMEGA*np.exp(-dGs/kT))
    return dict(T=T,S=S,J_BD=J_bd,log10_J_BD=float(logJ/np.log(10)),J_CNT=J_cnt,ratio=J_cnt/J_bd if J_bd>0 else None,n_star=float(nstar),dG_star_eV=float(dGs),Z=float(Z),beta_star=float(bstar),nu_58=float(Z*bstar/OMEGA))


def full_law(T,S,N,Ef,f,om,J_bd,log10_bd):
    """Post hoc (76.1): CNT with Zeldovich factor on the full capillary law A n^{2/3} + B n^{1/3} + C."""
    kT=K_B*T;Dm=kT*np.log(S);A,B,C=f['A'],f['B'],f['C']
    dG=lambda n:A*n**(2/3)+B*n**(1/3)+C-n*Dm;d1=lambda n:(2/3)*A*n**(-1/3)+(1/3)*B*n**(-2/3)-Dm
    d2=lambda n:-(2/9)*A*n**(-4/3)-(2/9)*B*n**(-5/3);nstar=brentq(d1,.01,1e7,xtol=1e-12)
    Dv=A0**2*NU*np.exp(-E_M/kT);C1=S*np.exp(S_F-E_F/kT)/OMEGA;bstar=4*np.pi*(3*nstar*OMEGA/(4*np.pi))**(1/3)*Dv*C1
    logJ=np.log(np.sqrt(abs(d2(nstar))/(2*np.pi*kT))*bstar/OMEGA)-dG(nstar)/kT
    return dict(T=T,S=S,J_BD=J_bd,J_CNT_full=float(np.exp(logJ)),ratio=float(np.exp(logJ-log10_bd*np.log(10))),n_star=float(nstar),log10_J_CNT_full=float(logJ/np.log(10)))


def onset(T,N,Ef,f,om):
    lo,hi=np.log(1.01),np.log(1e12)
    for _ in range(100):
        mid=.5*(lo+hi);J=rates(T,np.exp(mid),N,Ef,f,om)['J_BD']
        if J<1:lo=mid
        else:hi=mid
    return float(np.exp(.5*(lo+hi)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--embryo',default='results/embryo-atomistics-2026-09-27/report.json')
    parser.add_argument('--output',default='results/nucleation-rates-2026-09-28');args=parser.parse_args()
    N,Ef,f,om=energies(args.embryo);table=[rates(T,S,N,Ef,f,om) for T in (700.,800.,900.) for S in (10.,1e2,1e3,1e4)]
    for row in table:print({k:(f'{v:.3g}' if isinstance(v,float) else v) for k,v in row.items() if k in ('T','S','J_BD','J_CNT','ratio','n_star','dG_star_eV')})
    gate=all(1/3<=r['ratio']<=3 for r in table if r['T']==800. and r['S'] in (1e3,1e4))
    on={f'{T:g}':onset(T,N,Ef,f,om) for T in (600.,700.,800.,900.,1000.)};print('onset S (J=1 /m3/s):',on)
    full=[full_law(r['T'],r['S'],N,Ef,f,om,r['J_BD'],r['log10_J_BD']) for r in table]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(dict(scope='Sections 76 and 85: Becker-Doring vs CNT rates with atomistic embryo energies and Section 69 diffusion data.',table=table,onset_S=on,gates=dict(N1_cnt_vs_bd=gate),post_hoc_full_capillary_cnt=full,embryo_source=args.embryo,
        assumptions=dict(E_f=E_F,S_f=S_F,E_m=E_M,nu=NU,capture='diffusion-limited 4 pi r D C1',entropy='cluster vibrational/configurational entropy differences neglected')),indent=1)+'\n');print('N1',gate)
