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


_HALF=np.array([[1,0,0,0],[.5,.5,0,0],[.25,.5,.25,0],[.125,.375,.375,.125]])
_SPLIT=(_HALF,_HALF[::-1,::-1])


def subdivided_bound(C,depth):
    """Lower bound of a bicubic Bernstein patch by recursive de Casteljau subdivision at t=1/2.

    Returns a positive value when every leaf patch has positive coefficients; a leaf whose corner value (an exact
    determinant) is not positive is a genuine inversion and ends the search.
    """
    if C.min()>0:return float(C.min())
    if min(C[0,0],C[0,-1],C[-1,0],C[-1,-1])<=0 or depth==0:return float(C.min())
    return min(subdivided_bound(A@C@B.T,depth-1) for A in _SPLIT for B in _SPLIT)


def graded_edges(n,length=1.,band=(0.,1.),ratio=1.,width=.03,samples=20001):
    """Edges placing n cells with relative density 1+(ratio-1)phi(s) (Section 46.1)."""
    a,b=band
    # A band may extend past the domain so that a boundary region is fully refined.
    if int(n)!=n or n<2 or length<=0 or ratio<1 or width<=0 or not (a<b and b>0 and a<length):raise ValueError('Invalid grading.')
    s=np.linspace(0,length,samples)
    phi=(np.tanh((s-a)/width)-np.tanh((s-b)/width))/(2*np.tanh((b-a)/(2*width)))
    rho=1+(ratio-1)*np.clip(phi,0,1)
    cumulative=np.r_[0,np.cumsum((rho[1:]+rho[:-1])/2*np.diff(s))];cumulative/=cumulative[-1]
    edges=np.interp(np.linspace(0,1,int(n)+1),cumulative,s);edges[0]=0.;edges[-1]=length
    return edges


def uniform_geometry(mesh):
    """Unit per-cell scale factors: uniform products are reproduced exactly."""
    n=mesh.n;mesh.uniform=True;mesh.hx=np.full(n,mesh.h);mesh.hy=np.full(n,mesh.h)
    mesh.grad_scale=np.ones((n*n,2));mesh.weight_scale=np.ones(n*n)
    mesh.cell_weights=np.broadcast_to(mesh.weights,(n*n,len(mesh.weights)))
    mesh.cell_volume=np.full(n*n,mesh.h**2);mesh.edges=None


# Per-direction 3-point rules on [-1,1]: interior Gauss points, or Gauss-Lobatto points at the Q2 nodes (Section 47.2).
RULES=dict(gauss=lambda:np.polynomial.legendre.leggauss(3),lobatto=lambda:(np.array([-1.,0.,1.]),np.array([1.,4.,1.])/3))


class QuadraticMesh:
    def __init__(self,n,length=1.,edges=None,quadrature='gauss'):
        if int(n)!=n or n<2 or not np.isfinite(length) or length<=0:raise ValueError('Invalid Q2 mesh.')
        if quadrature not in RULES:raise ValueError('Unknown quadrature rule.')
        self.quadrature=quadrature
        self.n=int(n);self.length=float(length);self.h=length/n;self.degree=2
        if edges is None:
            x=np.linspace(0,length,2*n+1);self.coordinates=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1).reshape(-1,2)
        else:
            xe,ye=(np.asarray(e,dtype=float) for e in edges)
            for e in (xe,ye):
                if e.shape!=(n+1,) or not np.isfinite(e).all() or e[0]!=0 or e[-1]!=length or np.any(np.diff(e)<=0):
                    raise ValueError('Graded edges must increase from 0 to length with n+1 entries.')
            axes=[]
            for e in (xe,ye):
                nodes=np.empty(2*n+1);nodes[0::2]=e;nodes[1::2]=(e[:-1]+e[1:])/2;axes.append(nodes)
            self.coordinates=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,2)
        nodes=np.arange((2*n+1)**2).reshape(2*n+1,2*n+1)
        self.cells=np.array([nodes[2*i:2*i+3,2*j:2*j+3].ravel() for i in range(n) for j in range(n)])
        self.dofs=(2*self.cells[...,None]+np.arange(2)).reshape(-1,18)
        q,w=RULES[quadrature]()
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
        if edges is None:uniform_geometry(self)
        else:
            self.uniform=False;self.edges=(xe,ye);self.hx=np.diff(xe);self.hy=np.diff(ye)
            HX,HY=np.meshgrid(self.hx,self.hy,indexing='ij');HX=HX.ravel();HY=HY.ravel()
            self.grad_scale=np.stack((self.h/HX,self.h/HY),axis=-1);self.cell_volume=HX*HY
            self.weight_scale=self.cell_volume/self.h**2;self.cell_weights=self.weights[None,:]*self.weight_scale[:,None]

    def scale_gradient(self,values):
        """Apply per-cell physical column scaling to (E,G,...,2) reference-h gradients."""
        return values if self.uniform else values*self.grad_scale[:,None,None,:]

    def edge_conductances(self):
        """Two-point face/distance conductances for x- and y-neighbor cells (Section 46.2)."""
        if self.uniform:return np.ones((self.n-1,self.n)),np.ones((self.n,self.n-1))
        ax=self.hy[None,:]/((self.hx[:-1]+self.hx[1:])/2)[:,None]
        ay=self.hx[:,None]/((self.hy[:-1]+self.hy[1:])/2)[None,:]
        return ax,ay

    @property
    def node_count(self):return len(self.coordinates)

    def matrix(self,element,indices,size):
        rows=np.broadcast_to(indices[:,:,None],element.shape).ravel();cols=np.broadcast_to(indices[:,None,:],element.shape).ravel()
        return coo_matrix((element.ravel(),(rows,cols)),shape=(size,size)).tocsr()

    def determinant_coefficients(self,u):
        F=np.eye(2)+self.scale_gradient(np.einsum('gaJ,eai->egiJ',self.determinant_gradient,u.reshape(-1,2)[self.cells]))
        values=determinant(F).reshape(self.n*self.n,4,4)
        return np.einsum('ia,eab,jb->eij',self.inverse_bernstein,values,self.inverse_bernstein)

    def orientation_lower_bound(self,u,cells=None,subdivide=0,per_cell=False):
        """Minimum Bernstein coefficient of the bicubic determinant, a lower bound of J over each cell.

        subdivide>0 (Section 64 outcome, opt-in): cells whose coefficients are not all positive are split by de Casteljau
        at the midpoint in both directions, recursively up to that depth. Sub-patch coefficients still bound J from below
        and converge to its minimum, so the certificate stays rigorous but is much less conservative.
        """
        coefficients=self.determinant_coefficients(u)
        if cells is not None:coefficients=coefficients[cells]
        bounds=coefficients.reshape(len(coefficients),-1).min(axis=1)
        if subdivide>0:
            for k in np.flatnonzero(bounds<=0):bounds[k]=subdivided_bound(coefficients[k],subdivide)
        return bounds if per_cell else float(np.min(bounds))
