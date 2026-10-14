"""Section 65 diagnostic: is a refused step from a node-release checkpoint a genuine fold or a conservative refusal?

Retries the step from the last accepted state with a deeper subdivided certificate, then samples the exact bicubic
determinant densely in every flagged cell, and finally replays the step with the certificate bypassed to see whether
the converged solution itself inverts.
"""
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from bmvr.nodal_damage import NodalThermalFracture2D
from bmvr import quadratic_mesh as qm

DT=.00025
def BYPASS(self,u,cells=None,subdivide=0,per_cell=False):
    n=self.n*self.n if cells is None else int(np.count_nonzero(cells)) if np.asarray(cells).dtype==bool else len(cells)
    return np.ones(n) if per_cell else 1.


s=np.linspace(0,1,81);BERNSTEIN=np.stack([(1-s)**3,3*s*(1-s)**2,3*s*s*(1-s),s**3],1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path);parser.add_argument('--smoothing',type=float,default=.001);args=parser.parse_args()
    model,state,clock=NodalThermalFracture2D.load(args.checkpoint);mech=model.mechanics;M=mech.mesh
    mech.p=replace(mech.p,split_smoothing=args.smoothing)
    dense=lambda u,cells:np.einsum('ia,eab,jb->eij',BERNSTEIN,M.determinant_coefficients(u)[cells],BERNSTEIN).reshape(len(cells),-1).min(1)
    seen=[];original=qm.QuadraticMesh.orientation_lower_bound
    def spy(self,u,cells=None,subdivide=0,per_cell=False):
        b=original(self,u,cells,subdivide,per_cell)
        if np.min(b)<=0:seen.append(u.copy())
        return b
    qm.QuadraticMesh.orientation_lower_bound=spy;result=dict(checkpoint=str(args.checkpoint),time=clock['time'],released=len(getattr(model,'split',{})),retries={})
    for depth in (6,12):
        mech.subdivide=depth;seen.clear()
        try:model.step(state,DT);result['retries'][depth]='accepted';break
        except (ValueError,RuntimeError) as error:result['retries'][depth]=str(error)
    active=np.flatnonzero(getattr(mech,'active',None) if getattr(mech,'active',None) is not None else np.ones(len(M.cells),bool))
    m0=dense(state.u,active);result['last_accepted_dense_min_J']=dict(value=float(m0.min()),cell=int(active[m0.argmin()]),centre=M.coordinates[M.cells[active[m0.argmin()]]].mean(0).tolist())
    if not seen and 'accepted' not in result['retries'].values():
        # A solver failure without a certificate refusal: replay with the certificate bypassed and report the outcome.
        qm.QuadraticMesh.orientation_lower_bound=BYPASS
        try:
            s2,d=model.step(state,DT);m=dense(s2.u,active)
            result['bypassed_step']=dict(iterations=d['midpoint_iterations'],dense_min_J=float(m.min()),cell=int(active[m.argmin()]),centre=M.coordinates[M.cells[active[m.argmin()]]].mean(0).tolist())
        except (ValueError,RuntimeError) as error:result['bypassed_step']=dict(error=str(error))
        for fraction in (.5,.25):
            try:model.step(state,DT*fraction);result.setdefault('smaller_steps',{})[str(fraction)]='accepted'
            except (ValueError,RuntimeError) as error:result.setdefault('smaller_steps',{})[str(fraction)]=str(error)
    if seen:
        u=seen[-1];active=np.flatnonzero(getattr(mech,'active',None) if getattr(mech,'active',None) is not None else np.ones(len(M.cells),bool))
        C=M.determinant_coefficients(u);bad=active[[qm.subdivided_bound(C[k],12)<=0 for k in active]]
        result['flagged']=[dict(cell=int(k),centre=M.coordinates[M.cells[k]].mean(0).tolist(),dense_min_J=float(v),corner_J=C[k][[0,0,-1,-1],[0,-1,0,-1]].tolist())
            for k,v in zip(bad,dense(u,bad))]
        qm.QuadraticMesh.orientation_lower_bound=BYPASS
        s2,d=model.step(state,DT);m=dense(s2.u,active)
        result['bypassed_step']=dict(iterations=d['midpoint_iterations'],dense_min_J=float(m.min()),cell=int(active[m.argmin()]),centre=M.coordinates[M.cells[active[m.argmin()]]].mean(0).tolist())
        result['genuine_fold']=bool(min(f['dense_min_J'] for f in result['flagged'])<=0 and m.min()<=0)
    out=args.checkpoint.with_name(args.checkpoint.stem+'-fold-diagnostic.json');out.write_text(json.dumps(result,indent=1)+'\n');print(json.dumps(result,indent=1))


if __name__=='__main__':main()
