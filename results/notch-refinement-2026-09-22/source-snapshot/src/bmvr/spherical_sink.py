"""Finite-width spherical trapping layers and reciprocal population storage."""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse import bmat, csc_matrix, diags
from scipy.sparse.linalg import factorized, spsolve


def reaction_profile(r, radius=2., width=1., reactivity=1.):
    """Rate per unit time, normalized to a surface of area 4*pi*radius**2."""
    if radius <= width/2 or width <= 0 or reactivity < 0:
        raise ValueError('Require radius > width/2 > 0 and nonnegative reactivity.')
    offset=np.asarray(r)-radius
    delta=np.where(np.abs(offset)<width/2,2/width*np.cos(np.pi*offset/width)**2,0.)
    second=width**2*(1/12-1/(2*np.pi**2))
    return reactivity*radius**2/(radius**2+second)*delta


@dataclass
class SphericalSink:
    cells: int = 128
    length: float = 8.
    radius: float = 2.
    width: float = 1.
    diffusion: float = 1.
    reactivity: float = 1.

    def __post_init__(self):
        if self.cells < 4 or self.length <= self.radius+self.width/2 or self.diffusion <= 0:
            raise ValueError('Require at least four cells, positive diffusion and a contained layer.')
        self.edges=np.linspace(0,self.length,self.cells+1)
        self.r=(self.edges[1:]+self.edges[:-1])/2
        self.vol=4*np.pi/3*np.diff(self.edges**3)
        self.q=reaction_profile(self.r,self.radius,self.width,self.reactivity)
        self.exchange=self.vol*self.q
        self.conductance=4*np.pi*self.diffusion*self.edges[1:-1]**2/np.diff(self.r)
        diagonal=np.r_[self.conductance,0.]+np.r_[0.,self.conductance]
        self.stiffness=diags((-self.conductance,diagonal,-self.conductance),(-1,0,1),format='csc')
        self.outer_conductance=4*np.pi*self.diffusion*self.length**2/(self.length-self.r[-1])

    def steady(self, concentration=1., equilibrium=0.):
        diagonal=self.exchange.copy();diagonal[-1]+=self.outer_conductance
        rhs=self.exchange*equilibrium;rhs[-1]+=self.outer_conductance*concentration
        c=spsolve(self.stiffness+diags(diagonal),rhs)
        capture=float(self.exchange@(c-equilibrium))
        incoming=float(self.outer_conductance*(concentration-c[-1]))
        return c, capture, incoming

    def closed_system(self, capacity):
        """Return diagonal mass and symmetric diffusion/exchange stiffness."""
        if capacity <= 0:
            raise ValueError('Surface capacity must be positive.')
        column=csc_matrix(-self.exchange[:,None])
        stiffness=bmat([[self.stiffness+diags(self.exchange),column],
                        [column.T,csc_matrix([[self.exchange.sum()]])]],format='csc')
        return np.r_[self.vol,capacity],stiffness

    def stepper(self, capacity, dt):
        if dt <= 0:
            raise ValueError('Timestep must be positive.')
        mass,stiffness=self.closed_system(capacity)
        solve=factorized(diags(mass,format='csc')+dt*stiffness)
        return lambda state:solve(mass*state)


def collocation_reference(radii, *, length=8., radius=2., width=1., diffusion=1.,
                          reactivity=1., concentration=1., equilibrium=0., tolerance=1e-9):
    """Independent strong-form ODE reference, with the singularity removed."""
    mesh=np.unique(np.r_[np.linspace(0,length,257),
                         np.linspace(radius-width/2,radius+width/2,129)])
    delta=concentration-equilibrium
    def rhs(r,y):
        return np.vstack((y[1],reaction_profile(r,radius,width,reactivity)*y[0]/diffusion))
    result=solve_bvp(rhs,lambda left,right:np.array([left[0],right[0]-length*delta]),
                     mesh,np.vstack((delta*mesh,np.full_like(mesh,delta))),tol=tolerance,max_nodes=20000)
    if not result.success:
        raise RuntimeError(result.message)
    radii=np.asarray(radii);u=result.sol(radii)
    c=equilibrium+np.divide(u[0],radii,out=u[1].copy(),where=radii!=0)
    outer=result.sol(length)
    capture=float(4*np.pi*diffusion*(length*outer[1]-outer[0]))
    return c,capture,{'nodes':len(result.x),'max_rms_residual':float(result.rms_residuals.max()),
                      'tolerance':tolerance}


def robin_capture(*, length=8., radius=2., diffusion=1., reactivity=1., concentration=1., equilibrium=0.):
    if reactivity==0:
        return 0.
    return 4*np.pi*diffusion*radius*(concentration-equilibrium)/(1+diffusion/(reactivity*radius)-radius/length)


def exact_discrete_evolution(model, capacity, state, time):
    """Dense symmetric eigensolve used only as a small-system time reference."""
    from scipy.linalg import eigh
    mass,stiffness=model.closed_system(capacity)
    root=np.sqrt(mass)
    generator=stiffness.toarray()/root[:,None]/root[None,:]
    eigenvalues,basis=eigh(generator)
    return (basis@(np.exp(-time*eigenvalues)*(basis.T@(root*state))))/root
