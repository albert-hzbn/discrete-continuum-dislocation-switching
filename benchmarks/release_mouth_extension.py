"""Section 65.4 follow-up: extend a node-release checkpoint at a deeper certificate and track the mouth-cell Jacobian.

Steps the trajectory with the certificate depth given (still rigorous), node release after every step, and after
each accepted step records the dense minimum of the exact determinant. Stops at the first refused step and reports
whether the refusal is a genuine fold (dense J <= 0 in the certificate-bypassed step).
"""
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from bmvr.nodal_damage import NodalThermalFracture2D
from bmvr import quadratic_mesh as qm

DT=.00025
s=np.linspace(0,1,81);BERNSTEIN=np.stack([(1-s)**3,3*s*(1-s)**2,3*s*s*(1-s),s**3],1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path);parser.add_argument('--depth',type=int,default=12)
    parser.add_argument('--steps',type=int,default=120);args=parser.parse_args()
    model,state,clock=NodalThermalFracture2D.load(args.checkpoint);mech=model.mechanics;M=mech.mesh
    mech.p=replace(mech.p,split_smoothing=.001);mech.subdivide=args.depth;t=clock['time'];rows=[];end=None
    dense=lambda u:np.einsum('ia,eab,jb->eij',BERNSTEIN,M.determinant_coefficients(u),BERNSTEIN).reshape(len(M.cells),-1).min(1)
    for k in range(args.steps):
        try:
            state,d=model.step(state,DT);state,_=model.split_broken_line(state);t+=DT
        except (ValueError,RuntimeError) as error:
            original=qm.QuadraticMesh.orientation_lower_bound;qm.QuadraticMesh.orientation_lower_bound=lambda self,u,cells=None,subdivide=0,per_cell=False:(np.ones(self.n*self.n if cells is None else int(np.count_nonzero(cells))) if per_cell else 1.)
            try:
                s2,_=model.step(state,DT);m=dense(s2.u);bypass=dict(dense_min_J=float(m.min()),centre=M.coordinates[M.cells[m.argmin()]].mean(0).tolist())
            except (ValueError,RuntimeError) as e2:bypass=dict(error=str(e2))
            qm.QuadraticMesh.orientation_lower_bound=original
            end=dict(time=t,error=str(error),bypassed_step=bypass,genuine_fold=bool(bypass.get('dense_min_J',1.)<=0));break
        m=dense(state.u);rows.append(dict(time=t,min_J=float(m.min()),cell=int(m.argmin()),centre=M.coordinates[M.cells[m.argmin()]].mean(0).tolist(),iterations=d['midpoint_iterations']))
        if k%8==0:print(f"t={t:.5f} min J {m.min():.5f} at {np.round(rows[-1]['centre'],4)}",flush=True)
    result=dict(checkpoint=str(args.checkpoint),depth=args.depth,start=clock['time'],rows=rows,end=end)
    out=args.checkpoint.with_name(args.checkpoint.stem+f'-extension-depth{args.depth}.json');out.write_text(json.dumps(result,indent=1)+'\n');print(json.dumps(end))


if __name__=='__main__':main()
