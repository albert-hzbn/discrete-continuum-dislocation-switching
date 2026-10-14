"""Section 80: fcc slip family verification and copper interaction matrix; declared gates C1-C3."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from bmvr.continuum3d import fcc_slip_systems

A_IJ=dict(self=.122,coplanar=.122,Hirth=.07,Lomer=.122,glissile=.137,collinear=.625)


def classify(d,n):
    D=np.rint(d*np.sqrt(2)).astype(int);N=np.rint(n*np.sqrt(3)).astype(int);m=len(d);C=np.empty((m,m),dtype=object)
    for i in range(m):
        for j in range(m):
            if i==j:C[i,j]='self'
            elif (N[i]==N[j]).all() or (N[i]==-N[j]).all():C[i,j]='coplanar'
            elif abs(D[i]@D[j])==2:C[i,j]='collinear'
            elif D[i]@D[j]==0:C[i,j]='Hirth'
            else:
                d3=D[i]-np.sign(D[i]@D[j])*D[j];C[i,j]='glissile' if (N[i]@d3==0 or N[j]@d3==0) else 'Lomer'
    return C


def schmid(axis,d,n):
    e=np.asarray(axis,float);e/=np.linalg.norm(e);return np.abs((d@e)*(n@e))


def taylor(R,d,n,sym):
    e=R.apply([0,0,1.]);D=1.5*np.outer(e,e)-.5*np.eye(3)  # uniaxial tension, unit axial strain rate, isochoric
    idx=[(0,0),(1,1),(0,1),(0,2),(1,2)];Aeq=np.array([[sym[s][i,j] for s in range(12)] for i,j in idx]);beq=np.array([D[i,j] for i,j in idx])
    res=linprog(np.ones(24),A_eq=np.hstack([Aeq,-Aeq]),b_eq=beq,bounds=(0,None),method='highs');return res.fun


if __name__=='__main__':
    d,n,sym=fcc_slip_systems();C=classify(d,n)
    s001,s111,s123=(np.sort(schmid(a,d,n))[::-1] for a in ([0,0,1],[1,1,1],[1,2,3]))
    C1=bool(np.allclose(s001[:8],1/np.sqrt(6),atol=1e-12) and np.allclose(s001[8:],0,atol=1e-12) and np.allclose(s111[:6],np.sqrt(6)/9,atol=1e-12) and s123[0]-s123[1]>1e-3 and abs(s123[0]-.46656)<1e-4)
    counts=[{k:int(np.sum(C[i]==k)) for k in A_IJ} for i in range(12)];want=dict(self=1,coplanar=2,collinear=1,Hirth=2,Lomer=2,glissile=4);C2=all(c==want for c in counts)
    M=np.vectorize(lambda k:A_IJ[k])(C).astype(float)
    Rs=Rotation.random(2000,random_state=8001);T=np.array([taylor(R,d,n,sym) for R in Rs]);C3=bool(abs(T.mean()-3.06)/3.06<=.01)
    out=Path('results/fcc-family-2026-09-28');out.mkdir(parents=True,exist_ok=True)
    rep=dict(schmid=dict(s001=s001.tolist(),s111=s111.tolist(),s123=s123.tolist()),class_counts=counts[0],interaction_matrix=M.tolist(),classes=C.tolist(),
        taylor=dict(mean=float(T.mean()),std_error=float(T.std(ddof=1)/np.sqrt(len(T))),min=float(T.min()),max=float(T.max())),gates=dict(C1_schmid=C1,C2_classes=C2,C3_taylor=C3),
        copper_taylor_stress_MPa={f'{rho:g}':float(54.6e9*.255e-9*np.sqrt(M.sum(axis=1).mean()/12*rho)/1e6) for rho in (1e10,1e12,1e14)})
    (out/'report.json').write_text(json.dumps(rep,indent=1)+'\n');print(rep['gates'],rep['taylor'],rep['class_counts'],s123[:3])
