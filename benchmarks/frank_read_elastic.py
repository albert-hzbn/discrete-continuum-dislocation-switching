"""Section 77: Frank-Read critical stress with non-singular elastic self-interaction; F1 (declared + exact check) and F2."""
import json
from pathlib import Path
import numpy as np
from bmvr.elastic_line import parallel_interaction,critical_stress

if __name__=='__main__':
    out=Path('results/frank-read-elastic-2026-09-28');out.mkdir(parents=True,exist_ok=True);mu,nu,L0,a=1.,.324,400.,1.
    I=lambda D:2*(L0*np.arcsinh(L0/D)-np.sqrt(L0**2+D**2)+D);F1=[]
    for deg in (0,45,90):
        ang=np.radians(deg);K=mu/(2*np.pi)*(np.cos(ang)**2+np.sin(ang)**2/(1-nu))
        for d in (5.,10.,20.):
            h=.01*d;f=-(parallel_interaction(d+h,ang)-parallel_interaction(d-h,ang))/(2*h)
            exact=-K/2*(I(np.hypot(d+h,a))-I(np.hypot(d-h,a)))/(2*h)
            F1.append(dict(angle=deg,d=d,declared_rel=abs(f/L0-K/d)/(K/d),exact_rel=abs(f-exact)/exact))
    rows=[]
    for L in (50.,100.,200.,400.):
        tc,lt=critical_stress(L);rows.append(dict(L=L,tau_c=tc,tau_line_tension=lt,ratio=tc/lt,scaled=tc*L/mu));print(rows[-1],flush=True)
    x=np.log([r['L'] for r in rows]);y=np.array([r['scaled'] for r in rows]);c=np.polyfit(x,y,1);r2=1-np.sum((y-np.polyval(c,x))**2)/np.sum((y-y.mean())**2)
    gates=dict(F1_declared=all(r['declared_rel']<=.02 for r in F1),F1_exact_post_hoc=all(r['exact_rel']<=1e-10 for r in F1),F2_foreman_log=bool(r2>=.99))
    (out/'report.json').write_text(json.dumps(dict(F1=F1,critical=rows,fit=dict(slope=c[0],intercept=c[1],r2=r2),gates=gates),indent=1)+'\n');print(gates,c,r2,flush=True)
