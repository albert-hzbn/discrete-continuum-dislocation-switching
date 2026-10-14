"""Section 88: wedge-disclination dynamics; declared gates Q1-Q4."""
import json,time
from pathlib import Path
import numpy as np
from bmvr.disclinations import Disclinations,edge_force

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'disclination-dynamics-2026-09-28';N=512;MU,NU=1.,.3


def main():
    OUT.mkdir(parents=True,exist_ok=True);d=Disclinations(N,float(N),MU,NU,1.);c=N/2;rep={}
    # Q1: forces are the exact gradient of the discrete energy
    pos=np.array([[c-40,c-10],[c-10,c],[c+40,c+30],[c+20,c-40]]);om=[.2,-.2,.1,-.1];F=d.forces(pos,om);h=1e-4;err=0.
    for i in range(4):
        for k in range(2):
            p=pos.copy();p[i,k]+=h;ep=d.energy(p,om);p[i,k]-=2*h;err=max(err,abs(-(ep-d.energy(p,om))/(2*h)-F[i,k]))
    rep['Q1']=dict(rel_error=err/abs(F).max());print('Q1',rep['Q1'],flush=True)
    # Q2: glide-constrained dipoles interact like edge dislocations
    a=2.;w=1/(2*a);rows=[]
    for (x,y) in ((16,0),(24,8),(32,16),(12,30),(40,20),(50,10),(30,-30),(20,45)):
        pos=np.array([[c,c+a],[c,c-a],[c+x,c+y+a],[c+x,c+y-a]]);F=d.forces(pos,[w,-w,w,-w]);got=F[2,0]+F[3,0];ref=edge_force(MU,NU,1.,1.,x,y)
        rows.append(dict(x=x,y=y,r=float(np.hypot(x,y)),force=float(got),edge=float(ref),rel=float(abs(got-ref)/abs(ref))))
    rep['Q2']=rows;print('Q2',[(r['r'],round(r['rel'],4)) for r in rows],flush=True)
    # Q3: an opposite-sign pair with isotropic mobility annihilates
    pos=np.array([[c-10,c],[c+10,c]]);om=[.25,-.25];W=[d.energy(pos,om)];worst=0.;steps=0;start=time.time()
    while np.linalg.norm(pos[0]-pos[1])>=1. and steps<200000:
        F=d.forces(pos,om);dt=.01/np.abs(F).max();pos=pos+dt*F;W.append(d.energy(pos,om));steps+=1
        dW=W[-1]-W[-2];pred=-dt*np.sum(F*F);worst=max(worst,abs(dW-pred)/abs(dW));
    rep['Q3']=dict(steps=steps,final_separation=float(np.linalg.norm(pos[0]-pos[1])),W_start=W[0],W_end=W[-1],monotone=bool(np.all(np.diff(W)<=0)),
        worst_identity_error=worst,seconds=time.time()-start,W_samples=W[::max(len(W)//300,1)]);print('Q3',{k:v for k,v in rep['Q3'].items() if k!='W_samples'},flush=True)
    # Q4: an edge-dislocation dipole of two glide-constrained disclination dipoles relaxes to 45 degrees
    hgt=20.;xa,xb=c,c+5.;traj=[];start=time.time()
    for k in range(40000):
        pos=np.array([[xa,c+a],[xa,c-a],[xb,c+hgt-a],[xb,c+hgt+a]]);F=d.forces(pos,[w,-w,w,-w])  # second dipole inverted: -b
        fa,fb=(F[0,0]+F[1,0])/2,(F[2,0]+F[3,0])/2;dt=.05/max(abs(fa),abs(fb),1e-12);dt=min(dt,2e4)
        xa+=dt*fa;xb+=dt*fb
        if k%50==0:traj.append(dict(step=k,dx=float(xb-xa),fb=float(fb)))
        if abs(fb)<1e-9*abs(edge_force(MU,NU,1.,1.,hgt,0.))+1e-12:break
    rep['Q4']=dict(steps=k,dx=float(xb-xa),h=hgt,rel=float(abs(abs(xb-xa)-hgt)/hgt),seconds=time.time()-start,trajectory=traj);print('Q4',{k2:v for k2,v in rep['Q4'].items() if k2!='trajectory'},flush=True)
    sel=[r for r in rows if 8*a<=r['r']<=N/10]
    gates=dict(Q1_variational=rep['Q1']['rel_error']<=1e-6,Q2_edge_far_field=all(r['rel']<=.05 for r in sel),
        Q3_annihilation=rep['Q3']['final_separation']<1. and rep['Q3']['monotone'] and rep['Q3']['worst_identity_error']<=.01,Q4_dipole_45=rep['Q4']['rel']<=.05)
    rep['gates']=gates;(OUT/'report.json').write_text(json.dumps(rep,indent=1,default=float)+'\n');print(gates,flush=True)


if __name__=='__main__':main()
