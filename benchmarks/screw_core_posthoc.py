"""Section 82.4 post hoc (declared before running): separation of the ZJW copper screw partials against the force tolerance and the cylinder radius."""
import json
from pathlib import Path
import numpy as np
from ase.optimize import FIRE
from bmvr.eam_zjw import ZhouEAM
from screw_core import lattice,elastic,stacking_fault,screw,disregistry,crossing


def separation(at,ref,x0,yc,b,a0):
    xs,dis=disregistry(at,ref,x0,yc,b,a0);return crossing(xs,dis,.25)-crossing(xs,dis,.75)


if __name__=='__main__':
    out=Path('results/screw-core-2026-09-28');a0,_=lattice(ZhouEAM);rows=[]
    for R in (60.,120.):
        at,ref,x0,yc,b,sec,steps=screw(ZhouEAM,a0,R=R,fmax=.01);total=steps
        rows.append(dict(R=R,fmax=.01,steps=total,separation=separation(at,ref,x0,yc,b,a0)));print(rows[-1],flush=True)
        for fmax in ((1e-3,1e-4) if R==60. else (1e-4,)):
            opt=FIRE(at,logfile=None)
            for k,_ in enumerate(opt.irun(fmax=fmax,steps=100000)):
                if k and k%2000==0:print(dict(R=R,target=fmax,steps=total+k,separation=separation(at,ref,x0,yc,b,a0),fmax_now=float(np.abs(at.get_forces()).max())),flush=True)
            total+=opt.nsteps;rows.append(dict(R=R,fmax=fmax,steps=total,separation=separation(at,ref,x0,yc,b,a0)));print(rows[-1],flush=True)
            (out/'post-hoc-tolerance.json').write_text(json.dumps(rows,indent=1)+'\n')
        np.savez_compressed(out/f'screw-zjw-R{int(R)}-tight.npz',relaxed=at.positions,reference=ref.positions)
