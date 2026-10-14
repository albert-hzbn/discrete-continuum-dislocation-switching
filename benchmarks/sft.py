"""Section 87: stacking-fault tetrahedra from triangular Frank loops by annealing (Silcox-Hirsch), Mishin copper; gates T1-T2."""
import argparse,json,time
from pathlib import Path
import numpy as np
from ase import units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE
from bmvr.eam_setfl import SetflEAM

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'sft-2026-09-28';POT=ROOT/'data'/'potentials'/'Cu01.eam.alloy'
A0=3.615;NORMALS=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)


def cell(m):
    at=bulk('Cu','fcc',a=A0,cubic=True).repeat((m,m,m));c=at.positions[np.argmin(np.linalg.norm(at.positions-at.cell.sum(0)/2,axis=1))].copy();return at,c


def quench(at):
    at.calc=SetflEAM(POT);opt=FIRE(at,logfile=None);opt.run(fmax=.01,steps=30000);return float(at.get_potential_energy()),opt.nsteps


def triangle(at,c,N):
    """Sites c + i e1 + j e2 (i, j >= 0, i + j < N) of the (111) plane through c; e1, e2 are a/2<110> at 60 degrees."""
    e1,e2=A0/2*np.array([1.,-1,0]),A0/2*np.array([1.,0,-1]);L=at.cell.lengths()[0];out=[]
    for i in range(N):
        for j in range(N-i):
            p=c+i*e1+j*e2;D=at.positions-p;D-=np.round(D/L)*L;out.append(int(np.argmin(np.linalg.norm(D,axis=1))))
    assert len(set(out))==len(out);return sorted(out)


def faults(at,c,radius=30.):
    """Centrosymmetry parameter for atoms near c with 12 first neighbours. An intrinsic fault in this potential gives
    11.7 A^2 (calibrated on the 82 slab), so fault atoms are 6 < CSP < 16 A^2; partial cores lie below, surfaces above."""
    L=at.cell.lengths()[0];i,j,D=neighbor_list('ijD',at,3.0);sel=[];csp={}
    near=np.where(np.linalg.norm((at.positions-c)-np.round((at.positions-c)/L)*L,axis=1)<radius)[0];order=np.argsort(i,kind='stable');i,D=i[order],D[order];starts=np.searchsorted(i,np.arange(len(at)+1))
    for a in near:
        R=D[starts[a]:starts[a+1]]
        if len(R)!=12:continue
        used=np.zeros(12,bool);s=0.
        S=np.linalg.norm(R[:,None,:]+R[None,:,:],axis=2)**2;np.fill_diagonal(S,np.inf)
        for _ in range(6):
            k=int(np.argmin(np.where(used[:,None]|used[None,:],np.inf,S)));p,q=divmod(k,12);s+=S[p,q];used[p]=used[q]=True
        csp[int(a)]=s
    f=[a for a,v in csp.items() if 6.<v<16.];return f


def families(at,c,f):
    """Assign each fault atom to the {111} normal along which most fault atoms share its plane (within 0.4 A)."""
    if not f:return [0,0,0,0]
    L=at.cell.lengths()[0];P=(at.positions[f]-c)-np.round((at.positions[f]-c)/L)*L;counts=np.zeros(4,int)
    h=P@NORMALS.T  # (atoms, 4)
    share=np.array([[np.sum(abs(h[:,k]-h[a,k])<.4) for k in range(4)] for a in range(len(f))])
    for a in range(len(f)):counts[int(np.argmax(share[a]))]+=1
    return counts.tolist()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--N',type=int,required=True);parser.add_argument('--ps',type=float,default=20.);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);N=args.N;n=N*(N+1)//2;m=12 if N<=9 else 16
    perfect,c=cell(m);perfect.calc=SetflEAM(POT);e_coh=perfect.get_potential_energy()/len(perfect);res=dict(N=N,n=n,cell=m**3*4)
    at,_=cell(m);D=at.positions-c;L=at.cell.lengths()[0];D-=np.round(D/L)*L
    void=at.copy();del void[np.argsort(np.linalg.norm(D,axis=1),kind='stable')[:n].tolist()];E,_=quench(void);res['E_void']=E-(len(at)-n)*e_coh
    loop=at.copy();del loop[triangle(at,c,N)];E,_=quench(loop);res['E_frank']=E-(len(at)-n)*e_coh;np.save(OUT/f'frank-N{N}.npy',loop.positions)
    f=faults(loop,c);res['frank_fault_atoms']=len(f);res['frank_families']=families(loop,c,f);print(N,res,flush=True)
    hot=loop.copy();hot.calc=SetflEAM(POT);MaxwellBoltzmannDistribution(hot,temperature_K=800.,rng=np.random.default_rng(87));dyn=Langevin(hot,2*units.fs,temperature_K=800.,friction=.01/units.fs,rng=np.random.default_rng(870))
    start=time.time();steps=int(args.ps*500);log=[]
    for k in range(0,steps,500):
        dyn.run(500);log.append(dict(ps=(k+500)/500,T=float(hot.get_temperature()),E=float(hot.get_potential_energy())));print(N,log[-1],round(time.time()-start),flush=True)
    E,st=quench(hot);res['E_annealed']=E-(len(at)-n)*e_coh;f=faults(hot,c);res['annealed_fault_atoms']=len(f);res['annealed_families']=families(hot,c,f)
    res.update(anneal_log=log,quench_steps=st,seconds=time.time()-start);np.save(OUT/f'annealed-N{N}.npy',hot.positions)
    (OUT/f'N{N}.json').write_text(json.dumps(res,indent=1)+'\n');print(N,{k:v for k,v in res.items() if k!='anneal_log'},flush=True)


if __name__=='__main__':main()
