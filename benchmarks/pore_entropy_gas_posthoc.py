"""Section 75.3 post hoc: a stress test of the entropy barrier and equilibrated gas bubbles (labelled post hoc)."""
import json
from pathlib import Path
import numpy as np
from bmvr.pore_topology import PoreModel,EntropyPoreModel,GasPoreModel,GAMMA
from pore_entropy_gas import run

if __name__=='__main__':
    out=Path('results/pore-entropy-gas-2026-09-28');res={}
    for name,model,n_init in (('entropy_deep_undersaturation',EntropyPoreModel(n_eq=.005,length=32.,h=.5),1e-6),('quadratic_same_start',PoreModel(length=32.,h=.5),1e-6)):
        p=model.pore((0.,0.),6.);n=np.full(p.shape,n_init)
        try:_,_,info=run(model,n,p,50.,limiter=False);info.pop('count')
        except RuntimeError as e:info=dict(error=str(e))
        res[name]=info;print(name,info,flush=True)
    rows=[]
    for g in (4.,7.,10.):
        mg=EntropyPoreModel.__new__(GasPoreModel);GasPoreModel.__init__(mg,gas=g,theta_g=1.,length=64.,h=.5)
        p=mg.pore((0.,0.),8.);n=np.full(p.shape,.1);n,p,info=run(mg,n,p,4000.,limiter=True)
        A=mg.grid.integrate(p);R=np.sqrt(A/np.pi);P=mg.pressure(p);mu=float(np.mean(n))
        rows.append(dict(gas=g,R=float(R),P=float(P),mu=mu,rel_error=float(abs(mu+P-GAMMA/R)/(GAMMA/R)),n_spread=float(n.max()-n.min()),min_n=info['min_n']));print(rows[-1],flush=True)
    res['gas_equilibrated']=rows;(out/'post-hoc.json').write_text(json.dumps(res,indent=1,default=float)+'\n')
