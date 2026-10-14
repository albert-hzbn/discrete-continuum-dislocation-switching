"""Plane-strain Q1 spectral AT2 fracture with explicit irreversible bounds.

Quasistatic staggered local stationarity; no claim of global crack-path optimality.
Energies and reactions are per unit out-of-plane thickness.
"""
from dataclasses import dataclass,asdict
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def spectral_elasticity(strain,damage,lam=2.,shear=1.,residual=1e-4,tangent=True):
    e=np.asarray(strain);tensor=np.zeros(e.shape[:-1]+(2,2))
    tensor[...,0,0]=e[...,0];tensor[...,1,1]=e[...,1];tensor[...,0,1]=tensor[...,1,0]=e[...,2]/2
    values,Q=np.linalg.eigh(tensor);positive=np.maximum(values,0);trace=e[...,0]+e[...,1]
    ep=np.einsum('...ia,...a,...ja->...ij',Q,positive,Q)
    sp=lam*np.maximum(trace,0)[...,None,None]*np.eye(2)+2*shear*ep
    sigma=lam*trace[...,None,None]*np.eye(2)+2*shear*tensor
    g=(1-damage)**2+residual;sigma+=(g-1)[...,None,None]*sp
    psi_plus=lam*np.maximum(trace,0)**2/2+shear*np.sum(positive**2,axis=-1)
    psi=lam*trace**2/2+shear*np.sum(tensor*tensor,axis=(-2,-1))+(g-1)*psi_plus
    result={'density':psi,'positive_energy':psi_plus,'stress':np.stack((sigma[...,0,0],sigma[...,1,1],sigma[...,0,1]),axis=-1)}
    if tangent:
        difference=values[..., :,None]-values[...,None,:]
        numerator=positive[..., :,None]-positive[...,None,:]
        heaviside=np.where(values>0,1.,np.where(values<0,0.,.5))
        repeated=(heaviside[..., :,None]+heaviside[...,None,:])/2
        threshold=1e-12*np.maximum(np.max(abs(values),axis=-1),1e-15)[...,None,None]
        divided=np.divide(numerator,difference,out=repeated.copy(),where=abs(difference)>threshold)
        directions=np.array([[[1.,0],[0,0]],[[0.,0],[0,1]],[[0.,.5],[.5,0]]])
        rotated=np.einsum('...ia,mij,...jb->...mab',Q,directions,Q)
        dep=np.einsum('...ia,...ab,...mab,...jb->...mij',Q,divided,rotated,Q)
        htrace=np.where(trace>0,1.,np.where(trace<0,0.,.5))
        dsp=2*shear*dep+lam*htrace[...,None,None,None]*np.array([1.,1.,0.])[:,None,None]*np.eye(2)
        Cp=np.stack((dsp[...,0,0],dsp[...,1,1],dsp[...,0,1]),axis=-1).swapaxes(-1,-2)
        C=np.array([[lam+2*shear,lam,0.],[lam,lam+2*shear,0.],[0.,0.,shear]])
        result['tangent']=C+(g-1)[...,None,None]*Cp
    return result


@dataclass(frozen=True)
class FractureParameters:
    lam:float=2.
    shear:float=1.
    toughness:float=.001
    ell:float=.06
    residual:float=1e-4


def bound_quadratic(A,rhs,lower,initial=None,tolerance=1e-9,max_iterations=300):
    q=np.clip(lower if initial is None else initial,lower,1.).copy();diagonal=A.diagonal()
    for iteration in range(max_iterations+1):
        residual=A@q-rhs;projected=q-np.clip(q-residual/diagonal,lower,1.)
        error=float(np.max(abs(projected)))
        if error<=tolerance:return q,{'iterations':iteration,'projected_residual':error}
        prediction=q-residual/diagonal;lo=prediction<=lower;hi=prediction>=1.;active=lo|hi|(lower>=1.)
        free=~active;target=q.copy();target[lo]=lower[lo];target[hi]=1.;target[lower>=1.]=1.
        if np.any(free):target[free]=spsolve(A[free][:,free],rhs[free]-A[free][:,active]@target[active])
        target=np.clip(target,lower,1.);direction=target-q
        slope=float(residual@direction)
        if slope>=0:direction=-projected;slope=float(residual@direction)
        curvature=float(direction@(A@direction));step=1.
        for _ in range(40):
            if step*slope+step*step*curvature/2<=1e-4*step*slope+1e-16:break
            step*=.5
        else:raise RuntimeError('Bound quadratic line search failed.')
        q=np.clip(q+step*direction,lower,1.)
    raise RuntimeError(f'Damage quadratic did not converge: projected residual {error:.3e}.')


class Fracture2D:
    def __init__(self,n,length=1.,parameters=None,toughness=None):
        self.n=int(n);self.length=float(length);self.p=parameters or FractureParameters();p=self.p
        if n<2 or length<=0 or p.lam<0 or min(p.shear,p.toughness,p.ell,p.residual)<=0:raise ValueError('Invalid grid or physical coefficients.')
        x=np.linspace(0,length,n+1);self.coordinates=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1).reshape(-1,2)
        nodes=np.arange((n+1)**2).reshape(n+1,n+1)
        self.cells=np.stack((nodes[:-1,:-1],nodes[1:,:-1],nodes[1:,1:],nodes[:-1,1:]),axis=-1).reshape(-1,4)
        self.dofs=(2*self.cells[...,None]+np.arange(2)).reshape(-1,8);self.h=length/n;self.weight=self.h**2/4
        gauss=np.array(np.meshgrid([-1/np.sqrt(3),1/np.sqrt(3)],[-1/np.sqrt(3),1/np.sqrt(3)],indexing='ij')).reshape(2,-1).T
        sx=np.array([-1,1,1,-1]);sy=np.array([-1,-1,1,1]);xi=gauss[:,0,None];eta=gauss[:,1,None]
        self.N=(1+xi*sx)*(1+eta*sy)/4
        self.gradient=np.stack((sx*(1+eta*sy),sy*(1+xi*sx)),axis=-1)/(2*self.h)
        self.B=np.zeros((4,3,8));self.B[:,0,0::2]=self.gradient[:,:,0];self.B[:,1,1::2]=self.gradient[:,:,1]
        self.B[:,2,0::2]=self.gradient[:,:,1];self.B[:,2,1::2]=self.gradient[:,:,0]
        self.quadrature_points=np.einsum('ga,eai->egi',self.N,self.coordinates[self.cells])
        value=p.toughness if toughness is None else toughness
        if callable(value):value=value(self.quadrature_points[...,0],self.quadrature_points[...,1])
        self.toughness=np.broadcast_to(value,(n*n,4)).copy()
        if not np.isfinite(self.toughness).all() or self.toughness.min()<=0:raise ValueError('Positive finite toughness required.')

    @property
    def node_count(self):return len(self.coordinates)

    def matrix(self,element,indices,size):
        rows=np.broadcast_to(indices[:,:,None],element.shape).ravel();cols=np.broadcast_to(indices[:,None,:],element.shape).ravel()
        return coo_matrix((element.ravel(),(rows,cols)),shape=(size,size)).tocsr()

    def evaluate(self,u,q,tangent=False):
        strain=np.einsum('gai,ei->ega',self.B,u[self.dofs]);damage=q[self.cells]@self.N.T
        elastic=spectral_elasticity(strain,damage,self.p.lam,self.p.shear,self.p.residual,tangent)
        gradient=np.einsum('gai,ea->egi',self.gradient,q[self.cells])
        fracture=float(self.weight*np.sum(self.toughness*(damage**2/self.p.ell+self.p.ell*np.sum(gradient**2,axis=-1))/2))
        energy=float(self.weight*np.sum(elastic['density']));force_element=self.weight*np.einsum('gai,ega->ei',self.B,elastic['stress'])
        force=np.bincount(self.dofs.ravel(),weights=force_element.ravel(),minlength=2*self.node_count)
        result={'energy':energy+fracture,'elastic_energy':energy,'fracture_energy':fracture,'force':force,
            'positive_energy':elastic['positive_energy'],'damage_gauss':damage,'strain':strain,'stress':elastic['stress']}
        if tangent:
            element=self.weight*np.einsum('gai,egab,gbj->eij',self.B,elastic['tangent'],self.B)
            result['stiffness']=self.matrix(element,self.dofs,2*self.node_count)
        return result

    def damage_system(self,u):
        strain=np.einsum('gai,ei->ega',self.B,u[self.dofs])
        positive=spectral_elasticity(strain,np.zeros(strain.shape[:-1]),self.p.lam,self.p.shear,self.p.residual,False)['positive_energy']
        coefficient=2*positive+self.toughness/self.p.ell
        element=self.weight*(np.einsum('eg,ga,gb->eab',coefficient,self.N,self.N)
            +self.p.ell*np.einsum('eg,gai,gbi->eab',self.toughness,self.gradient,self.gradient))
        rhs_element=2*self.weight*np.einsum('eg,ga->ea',positive,self.N)
        rhs=np.bincount(self.cells.ravel(),weights=rhs_element.ravel(),minlength=self.node_count)
        return self.matrix(element,self.cells,self.node_count),rhs

    def boundary(self,load,compression=False):
        x,y=self.coordinates.T
        if compression:
            nodes=np.flatnonzero((x==0)|(x==self.length)|(y==0)|(y==self.length))
            fixed=(2*nodes[:,None]+np.arange(2)).ravel();values=(-load*self.coordinates[nodes]).ravel()
        else:
            bottom=np.flatnonzero(y==0);top=np.flatnonzero(y==self.length)
            fixed=np.r_[2*bottom+1,2*top+1,0];values=np.r_[np.zeros(len(bottom)),np.full(len(top),load),0.]
        return fixed,values

    def mechanical_error(self,state,fixed):
        free=np.ones(2*self.node_count,dtype=bool);free[fixed]=False
        return float(np.linalg.norm(state['force'][free])/max(np.linalg.norm(state['force'][fixed]),1e-12))

    def equilibrate(self,u,q,fixed,values,tolerance=1e-9,max_iterations=60):
        u=u.copy();u[fixed]=values;free=np.ones(len(u),dtype=bool);free[fixed]=False
        energies=[]
        for iteration in range(max_iterations+1):
            state=self.evaluate(u,q,tangent=True);energies.append(state['energy']);error=self.mechanical_error(state,fixed)
            if error<=tolerance:return u,state,{'iterations':iteration,'relative_residual':error,'energy':energies}
            direction=np.zeros_like(u);direction[free]=spsolve(state['stiffness'][free][:,free],-state['force'][free])
            slope=float(state['force']@direction);step=1.
            if slope>=0:raise RuntimeError('Mechanical Newton direction is not decreasing.')
            for _ in range(40):
                candidate=u+step*direction;following=self.evaluate(candidate,q)
                if following['energy']<=state['energy']+1e-4*step*slope+1e-14:break
                step*=.5
            else:raise RuntimeError('Mechanical line search failed.')
            u=candidate
        raise RuntimeError(f'Mechanics did not converge: relative residual {error:.3e}.')

    def solve_load(self,u,old_q,load,compression=False,tolerance=1e-6,max_iterations=2000,relaxation=1.5):
        fixed,values=self.boundary(load,compression);u=u.copy();u[fixed]=values;q=old_q.copy()
        energies=[self.evaluate(u,q)['energy']];mechanical_iterations=damage_iterations=0
        for iteration in range(1,max_iterations+1):
            u,_,mechanics=self.equilibrate(u,q,fixed,values);mechanical_iterations+=mechanics['iterations']
            energies.append(self.evaluate(u,q)['energy']);A,rhs=self.damage_system(u)
            minimum,damage=bound_quadratic(A,rhs,old_q,q);damage_iterations+=damage['iterations']
            omega=relaxation;residual=A@q-rhs
            for _ in range(40):
                new_q=np.clip(q+omega*(minimum-q),old_q,1.);direction=new_q-q
                if float(residual@direction+direction@(A@direction)/2)<=1e-16:break
                omega*=.5
            else:raise RuntimeError('Damage overrelaxation failed its energy audit.')
            damage_error=float(np.max(abs(new_q-np.clip(new_q-(A@new_q-rhs)/A.diagonal(),old_q,1.))))
            change=float(np.max(abs(new_q-q)));q=new_q;state=self.evaluate(u,q);energies.append(state['energy'])
            mechanical_error=self.mechanical_error(state,fixed)
            if change<=tolerance and mechanical_error<=tolerance and damage_error<=tolerance:
                top=2*np.flatnonzero(self.coordinates[:,1]==self.length)+1
                return u,q,{**state,'reaction':float(np.sum(state['force'][top])),
                    'alternate_iterations':iteration,'mechanical_iterations':mechanical_iterations,'damage_iterations':damage_iterations,
                    'mechanical_residual':mechanical_error,'damage_residual':damage_error,
                    'damage_change':change,'fixed_load_energy':energies}
        raise RuntimeError(f'Alternation failed at load {load}: displacement {mechanical_error:.3e}, damage change {change:.3e}.')

    def seed(self,length=.3):
        x,y=self.coordinates.T;distance=np.sqrt(np.maximum(x-length,0)**2+(y-self.length/2)**2)
        return np.exp(-distance/self.p.ell)

    def save(self,path,u,q,load,step):
        metadata={'n':self.n,'length':self.length,'parameters':asdict(self.p),'load':load,'step':step}
        np.savez_compressed(path,u=u,q=q,toughness=self.toughness,metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            p=json.loads(str(data['metadata']));clock={key:p.pop(key) for key in ('load','step')}
            model=cls(parameters=FractureParameters(**p.pop('parameters')),toughness=data['toughness'],**p)
            return model,data['u'].copy(),data['q'].copy(),clock
