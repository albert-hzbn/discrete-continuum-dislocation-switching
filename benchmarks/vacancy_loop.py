"""Section 86: a vacancy platelet on (111) collapsing into a Frank loop, against a compact void, with the Mishin potential; gates L1-L2."""
import json,time
from pathlib import Path
import numpy as np
from ase.build import bulk
from ase.optimize import FIRE
from bmvr.eam_setfl import SetflEAM

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'vacancy-loop-2026-09-28';POT=ROOT/'data'/'potentials'/'Cu01.eam.alloy'
A0=3.615;SIZES=(19,37,61,91,127);FMAX=.01


def cell():
    at=bulk('Cu','fcc',a=A0,cubic=True).repeat((12,12,12));c=at.positions[np.argmin(np.linalg.norm(at.positions-at.cell.sum(0)/2,axis=1))].copy();return at,c


def relax(at):
    at.calc=SetflEAM(POT);start=time.time();opt=FIRE(at,logfile=None);opt.run(fmax=FMAX,steps=20000);return float(at.get_potential_energy()),opt.nsteps,time.time()-start


def main():
    OUT.mkdir(parents=True,exist_ok=True);perfect,c=cell();perfect.calc=SetflEAM(POT);e_coh=perfect.get_potential_energy()/len(perfect)
    nhat=np.ones(3)/np.sqrt(3);d=A0/np.sqrt(3);rows=[]
    for n in SIZES:
        at,_=cell();D=at.positions-c;D-=np.round(D/(12*A0))*12*A0;h=D@nhat;rin=np.linalg.norm(D-np.outer(h,nhat),axis=1)
        void=at.copy();del void[np.argsort(np.linalg.norm(D,axis=1),kind='stable')[:n].tolist()];Ev,sv,tv=relax(void)
        plane=np.where(abs(h)<.1)[0];gone=plane[np.argsort(rin[plane],kind='stable')[:n]];Rdisc=float(rin[gone].max())
        loop=at.copy();keep=np.setdiff1d(np.arange(len(at)),gone)
        above=keep[(abs(h[keep]-d)<.1)&(rin[keep]<.5*Rdisc+1)];below=keep[(abs(h[keep]+d)<.1)&(rin[keep]<.5*Rdisc+1)]
        del loop[gone.tolist()];idx=np.searchsorted(keep,above),np.searchsorted(keep,below);El,sl,tl=relax(loop)
        P=loop.positions-c;gap=float(np.mean(P[idx[0]]@nhat)-np.mean(P[idx[1]]@nhat))
        rows.append(dict(n=n,E_void=Ev-(len(at)-n)*e_coh,E_loop=El-(len(at)-n)*e_coh,gap_over_d=gap/d,disc_radius_A=Rdisc,steps=[sv,sl],seconds=[tv,tl]));print(rows[-1],flush=True)
        (OUT/'rows.json').write_text(json.dumps(rows,indent=1)+'\n')
    mu,nu,gam=55.156e9,.3241,44.43e-3;EV=1.602176634e-19;bF=A0/np.sqrt(3)*1e-10;Aat=np.sqrt(3)/4*(A0*1e-10)**2
    for r in rows:
        R=np.sqrt(r['n']*Aat/np.pi);r['E_loop_continuum']=float((mu*bF**2*R/(2*(1-nu))*(np.log(4*R/bF)-2)+np.pi*R**2*gam)/EV)
    gates=dict(L1_collapse=all(abs(r['gap_over_d']-1)<=.1 for r in rows if r['n']>=37),L2_loop_lower=all(r['E_loop']<r['E_void'] for r in rows if r['n']>=37))
    (OUT/'report.json').write_text(json.dumps(dict(scope='Section 86: vacancy platelet to Frank loop, Mishin Cu; declared gates L1-L2.',cells=6912,e_coh=e_coh,rows=rows,gates=gates),indent=1)+'\n');print(gates,flush=True)


if __name__=='__main__':main()
