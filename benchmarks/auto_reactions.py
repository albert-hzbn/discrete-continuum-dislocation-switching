"""Section 78: automatic line-line reaction detection and execution; declared gates D1, D2."""
import json
from pathlib import Path
import numpy as np
from bmvr import graphs3d as G
from bmvr import reactions_auto as R

if __name__=='__main__':
    rng=np.random.default_rng(7801);errs=[];never_larger=True
    u=np.linspace(0,1,400)
    for _ in range(500):
        p0,p1,q0,q1=rng.normal(size=(4,3));s,t,d=R.segment_closest(p0,p1,q0,q1)
        P=p0+u[:,None]*(p1-p0);Q=q0+u[:,None]*(q1-q0);brute=float(np.min(np.linalg.norm(P[:,None]-Q[None],axis=-1)))
        errs.append(abs(d-brute));never_larger&=d<=brute+1e-12
    theta=np.arccos(1/3);nA=np.array([0,0,1.]);nB=np.array([np.sin(theta),0,np.cos(theta)]);cases={}
    for case in ('favorable','unfavorable','neutral'):
        pair=G.junction_pair(case);c=pair['crossed'];a,b=R.crossing_nodes(c,R.detect(c,.05));info=R.classify(c,a,b,nA,nB)
        jump=pair['junction'].core_energy-c.core_energy;cases[case]=dict(frank_gain=info['frank_gain'],favorable=info['favorable'],hand_built_core_jump=float(jump))
        if case=='favorable':
            auto=R.form_junction(c,a,b,info,.4);ref=pair['junction'];cases[case]['core_energy_difference']=abs(auto.core_energy-ref.core_energy)
            ga,gr=auto,ref;sa=sr=None
            for _ in range(400):
                ga,sa,_=G.line_step(ga,.01,None,sa);gr,sr,_=G.line_step(gr,.01,None,sr)
            la=float(np.linalg.norm(ga.nodes[-1]-ga.nodes[-2]));lr=G.junction_length(gr)
            cases[case].update(auto_length=la,hand_length=lr,length_difference=abs(la-lr))
        print(case,cases[case],flush=True)
    consistent=all((v['favorable'] and v['hand_built_core_jump']<0) or (not v['favorable'] and v['hand_built_core_jump']>=-1e-15) for v in cases.values())
    gates=dict(D1_geometry=bool(max(errs)<=1e-3 and never_larger),D2_classification_and_execution=bool(consistent and cases['favorable']['core_energy_difference']<=1e-12 and cases['favorable']['length_difference']<=1e-10))
    out=Path('results/auto-reactions-2026-09-28');out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(dict(D1=dict(max_error=max(errs),never_larger=bool(never_larger)),cases=cases,gates=gates),indent=1)+'\n');print(gates,max(errs))
