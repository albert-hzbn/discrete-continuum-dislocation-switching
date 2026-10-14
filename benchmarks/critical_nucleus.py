"""Section 84: finite-width critical nucleus, its barrier against 2D classical nucleation theory, and phase appearance and
disappearance from the saddle; declared gates N1-N3."""
import json,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.critical_nucleus import NucleationEnergy,planar_interface_energy,p

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'critical-nucleus-2026-09-28'
CASES=((.08,64),(.04,96),(.02,192),(.01,384))  # box side n h >= 8 R*


def main():
    OUT.mkdir(parents=True,exist_ok=True);gamma=planar_interface_energy();rows=[];saddles={}
    for dg,n in CASES:
        e=NucleationEnergy(n,dg=dg);R=gamma/dg;start=time.time();phi,iters=e.critical_nucleus(R)
        low,near,vecs=e.spectrum(phi,R);barrier=e.energy(phi)-e.energy(np.zeros_like(phi));cnt=np.pi*gamma**2/dg
        rows.append(dict(dg=dg,n=n,L=n*e.h,R_cnt=R,R_eff=float(np.sqrt(e.h**2*p(phi).sum()/np.pi)),newton_iterations=iters,
            grad_inf=float(np.abs(e.gradient(phi)).max()/e.h**2),barrier=barrier,barrier_cnt=cnt,deviation=(barrier-cnt)/cnt,
            smallest_two=low.tolist(),near_unstable=near.tolist(),phi_max=float(phi.max()),seconds=time.time()-start))
        saddles[dg]=(e,phi,vecs[:,0].reshape(phi.shape));print(rows[-1],flush=True)
    dev=[abs(r['deviation']) for r in rows]
    N1=all(r['grad_inf']<=1e-8 and r['smallest_two'][0]<-1e-6 and r['smallest_two'][1]>=-1e-6 and all(abs(v)<=1e-6 for v in r['near_unstable'][1:]) for r in rows)
    N2=all(a>b for a,b in zip(dev,dev[1:])) and dev[-1]<=.05
    e,phi,v=saddles[.04];R=gamma/.04;v=v if v.sum()>0 else -v;eps=1e-3/np.abs(v).max();flows={}
    for side,sign in (('minus',-1),('plus',1)):
        stop=(lambda q:q.max()<1e-6) if side=='minus' else (lambda q:np.sqrt(e.h**2*p(q).sum()/np.pi)>=2*R)
        q,t,E,dec=e.flow(phi+sign*eps*v,.02,20000.,stop)
        flows[side]=dict(time=t,final_max=float(q.max()),final_R=float(np.sqrt(e.h**2*p(q).sum()/np.pi)),energy_decreasing=dec,E_start=E[0],E_end=E[-1],reached=bool(stop(q)),
            energy_samples=E[::max(len(E)//400,1)]);print(side,{k:x for k,x in flows[side].items() if k!='energy_samples'},flush=True)
    N3=all(f['reached'] and f['energy_decreasing'] for f in flows.values())
    gates=dict(N1_saddle_index=bool(N1),N2_sharp_interface_limit=bool(N2),N3_appearance_disappearance=bool(N3))
    report=dict(scope='Section 84 finite-width critical nucleus; declared gates N1-N3. kappa = W = 1, h = 0.5.',gamma_h=gamma,gamma_continuum=float(np.sqrt(2)/6),
        rows=rows,flows=flows,gates=gates)
    (OUT/'report.json').write_text(json.dumps(report,indent=1,default=float)+'\n');print(gates,flush=True)
    figures=OUT/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(16,4.2),layout='constrained')
    for dg,(e,phi,_) in saddles.items():
        c=(np.arange(e.n)+.5)*e.h-e.L/2;axes[0].plot(c/(gamma/dg),phi[:,e.n//2],label=f'Δg = {dg:g}')
    axes[0].axvline(1,color='0.6',lw=.8,ls=':');axes[0].set(xlim=(-2.5,2.5),xlabel='x / R*_CNT',ylabel='φ* along a diameter',title='Critical nuclei');axes[0].legend(fontsize=8)
    x=np.array([1/r['dg'] for r in rows]);axes[1].plot(x,[r['barrier'] for r in rows],'o',label='finite-width saddle');xx=np.linspace(0,110);axes[1].plot(xx,np.pi*gamma**2*xx,'-',lw=.9,label='π γ_h² / Δg')
    axes[1].set(xlabel='1 / Δg',ylabel='ΔF* (per unit thickness)',title='Barrier against classical theory');axes[1].legend(fontsize=8)
    for side,f in flows.items():axes[2].plot(np.linspace(0,f['time'],len(f['energy_samples'])),np.array(f['energy_samples'])-f['E_start'],label=f"{side}: {'dissolves' if side=='minus' else 'grows'}")
    axes[2].set(xlabel='Time',ylabel='F - F[φ*]',title='Leaving the saddle (Δg = 0.04)');axes[2].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'critical-nucleus.{ext}',dpi=150)
    plt.close(fig)


if __name__=='__main__':main()
