"""Sections 91 and 91.2: discrete/continuum allocation for a gliding dislocation ensemble against both specialized methods; gates A1-A4."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json,time
from pathlib import Path
import numpy as np
from bmvr.representation_allocator import GlideEnsemble,Run

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'representation-allocator-2026-09-28'
G=256;N_DENSE=500000;N_SPARSE=200;STEPS=int(os.environ.get('STEPS','200'));TAU=.5;POST_HOC=os.environ.get('POST_HOC')=='1'


def ensemble(seed=91):
    rng=np.random.default_rng(seed);xd=(.5+.08*rng.standard_normal(N_DENSE))%1;yd=(.5+.08*rng.standard_normal(N_DENSE))%1;sd=np.where(rng.random(N_DENSE)<.6,1,-1)
    ys=np.where(rng.random(N_SPARSE)<.5,rng.uniform(0,.2,N_SPARSE),rng.uniform(.8,1.,N_SPARSE));xs=rng.random(N_SPARSE);ss=np.where(rng.random(N_SPARSE)<.5,1,-1)
    return np.r_[xd,xs],np.r_[yd,ys],np.r_[sd,ss],np.r_[-np.ones(N_DENSE,int),np.arange(N_SPARSE)]


def main():
    global OUT
    if POST_HOC:OUT=OUT/'post-hoc-1000'
    OUT.mkdir(parents=True,exist_ok=True);x,y,s,ids=ensemble()
    # setup (before any comparison): b0 so that the initial internal stress is at most half the applied stress
    probe=GlideEnsemble(G,b=1.,tau=TAU);smax=float(np.abs(probe.stress(probe.deposit(x,y,s.astype(float)))).max());b0=.5*TAU/smax
    M=1/(b0*TAU);e=GlideEnsemble(G,b=b0,M=M,tau=TAU);dt=.25*e.h/(M*b0*TAU);runs={}
    for method in ('D','C','H'):
        r=Run(e,x,y,s,ids,method);start=time.time()
        for k in range(STEPS):r.step(dt)
        runs[method]=dict(run=r,seconds=time.time()-start);print(method,'s/step',runs[method]['seconds']/STEPS,'particles',len(r.x),'continuum blocks',int(r.cont.sum()),flush=True)
    if POST_HOC:  # Section 91.2: C at 512^2 with the same physical core width (3 cells)
        e2=GlideEnsemble(2*G,b=b0,M=M,tau=TAU,width=3.,block=32);r2=Run(e2,x,y,s,ids,'C');start=time.time()
        for k in range(2*STEPS):r2.step(dt/2)  # half the step: the same CFL number on the finer grid
        runs['C512']=dict(run=r2,seconds=(time.time()-start)/2)  # per 256-grid step of time dt;print('C512 s/step',runs['C512']['seconds']/STEPS,flush=True)
    D,C,H=(runs[k]['run'] for k in 'DCH');B=e.B;nb=G//B
    blk=lambda g:g.reshape(nb,B,nb,B).mean(axis=(1,3))
    rel=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
    sparse_rows=np.zeros(G,bool);yc=(np.arange(G)+.5)*e.h;sparse_rows[(yc<.2)|(yc>.8)]=True
    pos=lambda r:{int(i):(float(xx)) for i,xx in zip(r.ids,r.x) if i>=0}
    pd,ph=pos(D),pos(H);common=sorted(set(pd)&set(ph));dx=np.array([(ph[i]-pd[i]+.5)%1-.5 for i in common])
    ledger=H.ledger;A1=max(max(l['count_error'] for l in ledger)/(N_DENSE+N_SPARSE),max(l['burgers_error'] for l in ledger)/(N_DENSE+N_SPARSE))
    rep=dict(G=G,N_dense=N_DENSE,N_sparse=N_SPARSE,steps=STEPS,dt=dt,b0=b0,M=M,tau=TAU,
        seconds_per_step={k:v['seconds']/STEPS for k,v in runs.items()},
        block_strain_error=dict(H=rel(blk(H.gamma),blk(D.gamma)),C=rel(blk(C.gamma),blk(D.gamma))),
        sparse_cell_strain_error=dict(H=rel(H.gamma[:,sparse_rows],D.gamma[:,sparse_rows]),C=rel(C.gamma[:,sparse_rows],D.gamma[:,sparse_rows])),
        sparse_tracked=len(common),sparse_rms_over_h=float(np.sqrt(np.mean(dx**2))/e.h) if len(dx) else None,
        conversions=dict(max_relative_error=A1,events=len(ledger),final_continuum_blocks=int(H.cont.sum()),final_particles=len(H.x)))
    if POST_HOC:
        g512=runs['C512']['run'].gamma.reshape(G,2,G,2).mean(axis=(1,3))  # restrict to 256^2 cells
        rep['sparse_cell_strain_error']['C512']=rel(g512[:,sparse_rows],D.gamma[:,sparse_rows]);rep['block_strain_error']['C512']=rel(blk(g512),blk(D.gamma))
    t=rep['seconds_per_step']
    rep['gates']=dict(A1_conservation=A1<=1e-12,A2_accuracy=rep['block_strain_error']['H']<=.05 and len(common)==N_SPARSE and rep['sparse_rms_over_h']<=.5,
        A3_cost=t['H']<=t['D']/3,A4_continuum_fails_sparse=rep['sparse_cell_strain_error']['C']>.2)
    np.savez_compressed(OUT/'fields.npz',gamma_D=D.gamma,gamma_C=C.gamma,gamma_H=H.gamma,cont=H.cont)
    (OUT/'report.json').write_text(json.dumps(rep,indent=1,default=float)+'\n');print(json.dumps({k:v for k,v in rep.items()},indent=1,default=float),flush=True)


if __name__=='__main__':main()
