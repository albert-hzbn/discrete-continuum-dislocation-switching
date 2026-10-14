"""Section 75.3 post hoc: does the gas-bubble Laplace residual scale like a diffuse-interface correction O(w/R)?
Each bubble starts near its predicted equilibrium radius so only a small adjustment is left; uniformity of n is checked."""
import json
from pathlib import Path
import numpy as np
from bmvr.pore_topology import GasPoreModel,GAMMA
from pore_entropy_gas import run

if __name__=='__main__':
    rows=[]
    for R_target,L,T in ((7.,64.,8000.),(14.,64.,8000.),(28.,128.,40000.)):
        mu0=.004;g=(GAMMA/R_target-mu0)*np.pi*R_target**2  # gas content for equilibrium at R_target with matrix potential mu0
        mg=GasPoreModel(gas=g,theta_g=1.,length=L,h=.5);p=mg.pore((0.,0.),R_target);n=np.full(p.shape,mu0)
        n,p,info=run(mg,n,p,T,limiter=True);A=mg.grid.integrate(p);R=np.sqrt(A/np.pi);P=mg.pressure(p);mu=float(np.mean(n))
        rows.append(dict(R_target=R_target,gas=g,L=L,R=float(R),residual=float((mu+P-GAMMA/R)/(GAMMA/R)),n_spread=float(n.max()-n.min())));print(rows[-1],flush=True)
    Path('results/pore-entropy-gas-2026-09-28/gas-size-post-hoc.json').write_text(json.dumps(rows,indent=1)+'\n')
