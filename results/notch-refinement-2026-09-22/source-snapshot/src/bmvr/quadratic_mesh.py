"""Regular Q2 geometry and sufficient Bernstein orientation bounds."""
import numpy as np
from scipy.sparse import coo_matrix
from .small_matrix import determinant


def quadratic_shape(points,h):
    points=np.asarray(points,dtype=float)
    def basis(z):
        return np.stack((z*(z-1)/2,1-z*z,z*(z+1)/2),axis=-1),np.stack((z-.5,-2*z,z+.5),axis=-1)
    x,dx=basis(points[:,0]);y,dy=basis(points[:,1])
    N=np.einsum('ga,gb->gab',x,y).reshape(-1,9)
    gradient=np.stack((np.einsum('ga,gb->gab',dx,y),np.einsum('ga,gb->gab',x,dy)),axis=-1).reshape(-1,9,2)*2/h
    return N,gradient


class QuadraticMesh:
    def __init__(self,n,length=1.):
        if int(n)!=n or n<2 or not np.isfinite(length) or length<=0:raise ValueError('Invalid Q2 mesh.')
        self.n=int(n);self.length=float(length);self.h=length/n;self.degree=2
        x=np.linspace(0,length,2*n+1);self.coordinates=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1).reshape(-1,2)
        nodes=np.arange((2*n+1)**2).reshape(2*n+1,2*n+1)
        self.cells=np.array([nodes[2*i:2*i+3,2*j:2*j+3].ravel() for i in range(n) for j in range(n)])
        self.dofs=(2*self.cells[...,None]+np.arange(2)).reshape(-1,18)
        q,w=np.polynomial.legendre.leggauss(3)
        self.reference_points=np.stack(np.meshgrid(q,q,indexing='ij'),axis=-1).reshape(-1,2)
        self.weight=self.h**2/4;self.weights=self.weight*np.outer(w,w).ravel();self.quadrature_count=9
        self.N,self.gradient=quadratic_shape(self.reference_points,self.h)
        self.quadrature_points=np.einsum('ga,eai->egi',self.N,self.coordinates[self.cells])
        corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        self.corner_gradient=quadratic_shape(corners,self.h)[1]
        t=np.linspace(0,1,4);sample=np.stack(np.meshgrid(2*t-1,2*t-1,indexing='ij'),axis=-1).reshape(-1,2)
        self.determinant_gradient=quadratic_shape(sample,self.h)[1]
        B=np.stack(((1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3),axis=-1)
        self.inverse_bernstein=np.linalg.inv(B)

    @property
    def node_count(self):return len(self.coordinates)

    def matrix(self,element,indices,size):
        rows=np.broadcast_to(indices[:,:,None],element.shape).ravel();cols=np.broadcast_to(indices[:,None,:],element.shape).ravel()
        return coo_matrix((element.ravel(),(rows,cols)),shape=(size,size)).tocsr()

    def determinant_coefficients(self,u):
        F=np.eye(2)+np.einsum('gaJ,eai->egiJ',self.determinant_gradient,u.reshape(-1,2)[self.cells])
        values=determinant(F).reshape(self.n*self.n,4,4)
        return np.einsum('ia,eab,jb->eij',self.inverse_bernstein,values,self.inverse_bernstein)

    def orientation_lower_bound(self,u):
        return float(np.min(self.determinant_coefficients(u)))
