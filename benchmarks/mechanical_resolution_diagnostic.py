"""Independent Q1 gradient evaluation without interpolating saved stresses."""
import argparse
from pathlib import Path
import numpy as np
from bmvr.thermochemical import ThermalChemical2D
from bmvr.finite_elasticity import constitutive
from thermochemical import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--fine',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    model,_,_=ThermalChemical2D.load(args.fine/'coupled-48-0.00025-initial.npz')
    points=model.mechanics.mesh.quadrature_points.reshape(-1,2);x,y=points.T
    c=.3+.08*np.cos(np.pi*x)*np.cos(np.pi*y);T=1+.1*np.cos(np.pi*x)*np.cos(np.pi*y)
    eta=.5+.3*np.tanh((.22-np.hypot(x-.5,y-.5))/.08);results={};stress=[]
    for n in (16,24,32,48):
        dt='0.0005' if n<32 else '0.00025'
        path=(args.base if n<32 else args.fine)/f'coupled-{n}-{dt}-initial.npz'
        m,s,_=ThermalChemical2D.load(path);mesh=m.mechanics.mesh
        indices=np.minimum((points/mesh.h).astype(int),n-1);cells=indices[:,0]*n+indices[:,1]
        xi=2*(points[:,0]/mesh.h-indices[:,0])-1;yy=2*(points[:,1]/mesh.h-indices[:,1])-1
        sx=np.array([-1,1,1,-1]);sy=np.array([-1,-1,1,1])
        grad=np.stack((sx*(1+yy[:,None]*sy),sy*(1+xi[:,None]*sx)),axis=-1)/(2*mesh.h)
        F=np.broadcast_to(np.eye(3),(len(points),3,3)).copy()
        F[:,:2,:2]+=np.einsum('paJ,pai->piJ',grad,s.u.reshape(-1,2)[mesh.cells[cells]])
        stress.append(constitutive(F,c=c,eta=eta,T=T,tangent=False)['P'])
        if n==48:results['native_gradient_reconstruction_error']=float(np.max(abs(F-m.mechanics.evaluate(s.u)['F'].reshape(-1,3,3))))
    for n,P in zip((16,24,32),stress[:-1]):
        diff=P-stress[-1];norms=np.sum(diff**2,axis=(1,2))
        results[str(n)]=dict(relative_Piola_error_against_48=float(np.linalg.norm(diff)/np.linalg.norm(stress[-1])),
            top_fifth_squared_error_fraction=float(norms[y>.8].sum()/norms.sum()))
    args.output.mkdir(parents=True,exist_ok=True);write_json(args.output/'independent-initial-gradient.json',results)


if __name__=='__main__':main()
