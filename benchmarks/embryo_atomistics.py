"""Sections 72 and 85: molecular statics for copper vacancy-cluster (void embryo) energetics; declared gates A1-A3 (EMT) and M1-M2 (Mishin)."""
import argparse,json,time
from pathlib import Path
import numpy as np
from ase.build import bulk,fcc111,fcc100,fcc110
from functools import partial
from ase.calculators.emt import EMT
from bmvr.eam_setfl import SetflEAM
from ase.optimize import FIRE

K_B=8.617333262e-5;EV=1.602176634e-19
SIZES=(1,2,4,6,13,19,43,55,79,87,135);POST_HOC_SIZES=tuple(n for n in range(1,25) if n not in (1,2,4,6,13,19))  # 72.3, labelled
FMAX=.01;T=800.;SUPERSAT=(10.,1e2,1e3,1e4)
ROOT=Path(__file__).resolve().parents[1]
CALCS=dict(emt=EMT,mishin=partial(SetflEAM,ROOT/'data'/'potentials'/'Cu01.eam.alloy'))
CALC=EMT


def energy(atoms,relax=True):
    atoms.calc=CALC()
    if relax:FIRE(atoms,logfile=None).run(fmax=FMAX,steps=3000)
    return float(atoms.get_potential_energy())


def lattice():
    a_s=np.linspace(3.56,3.64,9);e=[]
    for a in a_s:
        at=bulk('Cu','fcc',a=a,cubic=True).repeat((3,3,3));e.append(energy(at,False)/len(at))
    c=np.polyfit(a_s,e,2);a0=-c[1]/(2*c[0]);at=bulk('Cu','fcc',a=a0,cubic=True).repeat((3,3,3))
    return float(a0),energy(at,False)/len(at)


def surfaces(a0,e_bulk):
    out={}
    for name,build in (('111',fcc111),('100',fcc100),('110',fcc110)):
        slab=build('Cu',size=(4,4,12),a=a0,vacuum=10.,periodic=True);E=energy(slab)
        area=np.linalg.norm(np.cross(slab.cell[0],slab.cell[1]));out[name]=(E-len(slab)*e_bulk)/(2*area)*EV*1e20
    return out


def cluster(a0,e_bulk,n):
    at=bulk('Cu','fcc',a=a0,cubic=True).repeat((8,8,8));centre=at.positions[np.argmin(np.linalg.norm(at.positions-at.cell.sum(0)/2,axis=1))]
    d=np.linalg.norm(at.get_distances(np.argmin(np.linalg.norm(at.positions-centre,axis=1)),range(len(at)),mic=True)[None,:],axis=0)
    order=np.lexsort((np.arange(len(at)),np.round(d,6)));del at[sorted(order[:n].tolist())]
    start=time.time();E=energy(at);return float(E-len(at)*e_bulk),time.time()-start


def main():
    global CALC
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--potential',choices=CALCS,default='emt');args=parser.parse_args()
    CALC=CALCS[args.potential]
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    cache=out/'partial.json';cached=json.loads(cache.read_text()) if cache.exists() else None
    if cached:a0,e_bulk,gam=cached['a0'],cached['e_bulk'],cached['surfaces'];Ef={int(k):v for k,v in cached['E_f'].items()}  # reuse computed energies
    else:
        a0,e_bulk=lattice();print('a0',a0,'e_bulk',e_bulk,flush=True)
        gam=surfaces(a0,e_bulk);print('surfaces J/m2',gam,flush=True);Ef={}
    for n in SIZES+POST_HOC_SIZES:
        if n in Ef:continue
        Ef[n],sec=cluster(a0,e_bulk,n);print('n',n,'E_f',Ef[n],'eV',round(sec,1),'s',flush=True)
        (out/'partial.json').write_text(json.dumps(dict(a0=a0,e_bulk=e_bulk,surfaces=gam,E_f={str(k):v for k,v in Ef.items()}),indent=1))
    # the declared fit set; post hoc sizes are not fitted
    omega=a0**3/4;ns=np.array([n for n in SIZES if n>=13],float);E=np.array([Ef[int(n)] for n in ns])
    X=np.stack([ns**(2/3),ns**(1/3),np.ones_like(ns)],1);coef,*_=np.linalg.lstsq(X,E,rcond=None);pred=X@coef
    r2=1-np.sum((E-pred)**2)/np.sum((E-E.mean())**2);A,B,C=coef
    gamma_eff=A/((36*np.pi)**(1/3)*omega**(2/3))*EV*1e20;gamma_flat=float(np.mean(list(gam.values())))
    kT=K_B*T;E1=Ef[1];nn=np.linspace(1,20000,200000);barriers={}
    for S in SUPERSAT:
        # mu_v = E_f(1) + kT ln x_v with x_v = S x_eq: the n E_f(1) terms cancel, leaving dG(n) = E_f(n) - n kT ln S.
        dG=A*nn**(2/3)+B*nn**(1/3)+C-nn*kT*np.log(S);i=int(np.argmax(dG))
        small=np.array(sorted(k for k in Ef if k<=24),float);dGd=np.array([Ef[int(k)] for k in small])-small*kT*np.log(S)
        big=nn>24;j=int(np.argmax(dG*big-1e9*(~big)));combined=max(dGd.max(),dG[j]);nstar=small[int(np.argmax(dGd))] if dGd.max()>=dG[j] else nn[j]
        cnt=16*np.pi*(gamma_eff/(EV*1e20))**3*omega**2/(3*(kT*np.log(S))**2)
        barriers[f'{S:g}']=dict(n_star=float(nn[i]),R_star_nm=float((3*nn[i]*omega/(4*np.pi))**(1/3)/10),barrier_eV=float(dG[i]),barrier_over_kT=float(dG[i]/kT),
            cnt_gamma_eff_eV=float(cnt),cnt_ratio=float(cnt/dG[i]),
            post_hoc_discrete=dict(n_star=float(nstar),barrier_eV=float(combined),barrier_over_kT=float(combined/kT),cnt_ratio=float(cnt/combined)))
    gates=dict(A1_fidelity=abs(a0-3.615)/3.615<=.01 and abs(E1-1.287)/1.287<=.15,
        A2_capillarity=bool(r2>=.999 and abs(gamma_eff-gamma_flat)/gamma_flat<=.30),
        A3_cnt_adequate=all(abs(barriers[k]['cnt_ratio']-1)<=.25 for k in ('1000','10000')))
    if args.potential=='mishin':gates.update(M1_fidelity=abs(a0/3.615-1)<=1e-3 and abs(E1/1.287-1)<=.05,M2_capillarity=bool(r2>=.999))
    report=dict(scope=f'Sections 72 and 85: {args.potential} molecular statics for Cu vacancy clusters; declared gates.',potential=args.potential,a0_angstrom=a0,e_bulk_eV=e_bulk,omega_A3=omega,
        surfaces_J_m2=gam,surface_mean_J_m2=gamma_flat,measured_surface_J_m2=1.75,E_f_eV={str(k):v for k,v in Ef.items()},
        fit=dict(A=A,B=B,C=C,r2=r2,gamma_eff_J_m2=gamma_eff),T=T,barriers=barriers,gates=gates)
    (out/'report.json').write_text(json.dumps(report,indent=1)+'\n');print(gates,json.dumps(barriers,indent=0)[:800],flush=True)


if __name__=='__main__':main()
