"""Section 92: error control by the allocation threshold n_c, calibrated on one seed and tested on another; gates E1-E3."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json,time
from pathlib import Path
import numpy as np
from bmvr.representation_allocator import GlideEnsemble,Run
from representation_allocator import ensemble,G,TAU

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'allocator-control-2026-09-28';STEPS=1000;NC=(4,8,16,32,64,128,256,512);TOLS=(.02,.01,.005)


def seed_study(seed):
    x,y,s,ids=ensemble(seed);probe=GlideEnsemble(G,b=1.,tau=TAU);b0=.5*TAU/float(np.abs(probe.stress(probe.deposit(x,y,s.astype(float)))).max())
    M=1/(b0*TAU);dt=None;res={};gam={}
    def run(method,nc=64):
        e=GlideEnsemble(G,b=b0,M=M,tau=TAU,n_c=nc);r=Run(e,x,y,s,ids,method);step=.25*e.h/(M*b0*TAU);start=time.time()
        for _ in range(STEPS):r.step(step)
        return r,(time.time()-start)/STEPS,e
    D,tD,e=run('D');C,tC,_=run('C');yc=(np.arange(G)+.5)*e.h;sp=(yc<.2)|(yc>.8)
    err=lambda g:float(np.linalg.norm(g[:,sp]-D.gamma[:,sp])/np.linalg.norm(D.gamma[:,sp]))
    res['D']=dict(cost=tD);res['C']=dict(cost=tC,error=err(C.gamma))
    for nc in NC:
        H,tH,_=run('H',nc);res[str(nc)]=dict(cost=tH,error=err(H.gamma),continuum_blocks=int(H.cont.sum()),particles=len(H.x));print(seed,nc,res[str(nc)],flush=True)
    return res


def main():
    OUT.mkdir(parents=True,exist_ok=True);study={seed:seed_study(seed) for seed in (91,92)}
    def monotone(r,key,sign):
        v=[r[str(n)][key] for n in NC];return all(sign*(b-a)<=.1*abs(a) for a,b in zip(v,v[1:]))
    E1=all(monotone(study[s],'error',1) and monotone(study[s],'cost',-1) for s in study)
    choice={};E2=True
    for tol in TOLS:
        ok=[n for n in NC if study[91][str(n)]['error']<=tol];n=min(ok) if ok else None;choice[str(tol)]=n
        E2&=n is not None and study[92][str(n)]['error']<=tol and study[92][str(n)]['cost']<=study[92]['D']['cost']/3
    t=study[92];E3=all(t[str(n)]['cost']<t['D']['cost'] and t[str(n)]['error']<t['C']['error'] for n in NC if n>=16)
    rep=dict(steps=STEPS,thresholds=NC,tolerances=TOLS,study=study,chosen_n_c=choice,gates=dict(E1_monotone=E1,E2_transfer=bool(E2),E3_between=E3))
    (OUT/'report.json').write_text(json.dumps(rep,indent=1)+'\n');print(rep['gates'],choice,flush=True)


if __name__=='__main__':main()
