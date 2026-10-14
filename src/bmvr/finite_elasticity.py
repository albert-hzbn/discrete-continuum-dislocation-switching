"""Reference Q1/Q2 finite elasticity with prescribed Section 26 internal fields.

Spatial mechanics only: composition, phase, temperature and plastic map are frozen.
"""
from dataclasses import asdict, dataclass
import json
import numpy as np
from scipy.sparse.linalg import spsolve
from .fracture2d import Fracture2D
from .quadratic_mesh import QuadraticMesh,uniform_geometry
from .small_matrix import determinant,inverse


@dataclass(frozen=True)
class FiniteParameters:
    lam: float = 2.
    mu: float = 1.
    cv: float = 10.
    alpha_c: float = .02
    alpha_eta: float = .015
    alpha_T: float = .005
    c0: float = .2
    T0: float = 1.
    split_smoothing: float = 0.  # Section 64 outcome: C2 band width in ln J_e for the compressive split; 0 keeps the C1 split

    def __post_init__(self):
        if (not all(np.isfinite(v) for v in asdict(self).values())
                or self.lam < 0 or min(self.mu, self.cv, self.T0) <= 0
                or not 0 <= self.c0 <= 1 or self.split_smoothing < 0):
            raise ValueError('Invalid finite-elasticity coefficients.')


def constitutive(F, Fp=None, c=.2, eta=0., T=1., parameters=None, tangent=True, elastic_scale=1., elastic_split='isotropic', mechanical=None):
    """Return 3D free energy and exact first/second deformation derivatives.

    mechanical (Section 64): optional 0/1 factor per point; 0 removes every elastic and thermoelastic term
    (a deleted cell) while keeping the chemical, phase and purely thermal parts unchanged.
    """
    p = parameters or FiniteParameters()
    F = np.asarray(F, dtype=float)
    Fp = np.asarray(np.eye(3) if Fp is None else Fp, dtype=float)
    c, eta, T = (np.asarray(v, dtype=float) for v in (c, eta, T))
    scale=np.asarray(elastic_scale,dtype=float)
    if not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid elastic scale.')
    if elastic_split not in ('isotropic','volumetric-deviatoric'):raise ValueError('Unknown elastic split.')
    if F.shape[-2:] != (3, 3) or Fp.shape[-2:] != (3, 3):
        raise ValueError('F and Fp must end in (3, 3).')
    if not all(np.isfinite(v).all() for v in (F, Fp, c, eta, T)):
        raise ValueError('Nonfinite state.')
    J = determinant(F)
    if (np.any(J <= 0) or not np.allclose(determinant(Fp), 1., atol=1e-10, rtol=0)
            or np.any(T <= 0) or np.any((c < 0) | (c > 1))
            or np.any((eta < 0) | (eta > 1))):
        raise ValueError('State violates orientation, plastic volume or scalar bounds.')
    b = p.alpha_c*(c-p.c0)+p.alpha_eta*eta**2*(3-2*eta)+p.alpha_T*(T-p.T0)
    with np.errstate(over='ignore', under='ignore'):
        a = np.exp(b)
    if not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError('Invalid growth factor.')
    B = inverse(Fp)/a[..., None, None]
    Fe = F@B
    Je = determinant(Fe)
    if not np.isfinite(Fe).all() or np.any(Je <= 0) or not np.isfinite(Je).all():
        raise ValueError('Invalid elastic deformation.')
    log = np.log(Je)
    Ce = Fe.swapaxes(-1, -2)@Fe
    trace = np.trace(Ce, axis1=-2, axis2=-1)
    t = p.lam*log-p.mu
    invT = inverse(F).swapaxes(-1, -2)
    A = B@B.swapaxes(-1, -2)
    FA=F@A
    P = p.mu*FA+t[..., None, None]*invT
    M = p.mu*Ce+t[..., None, None]*np.eye(3)
    trM0 = np.trace(M, axis1=-2, axis2=-1)
    trM=scale*trM0;P=scale[...,None,None]*P;M=scale[...,None,None]*M
    elastic0 = p.mu*(trace-3)/2-p.mu*log+p.lam*log**2/2
    elastic=scale*elastic0
    qminus=np.zeros_like(log);Kminus=np.zeros_like(log);Wminus=np.zeros_like(log)
    if elastic_split=='volumetric-deviatoric':
        z=np.exp(2*log/3);compression=log<0
        Wminus=np.where(compression,1.5*p.mu*np.expm1(2*log/3)-p.mu*log+.5*p.lam*log**2,0.)
        qminus=np.where(compression,p.mu*np.expm1(2*log/3)+p.lam*log,0.)
        Kminus=np.where(compression,2*p.mu*z/3+p.lam,0.)
        if p.split_smoothing>0:
            # C2 split: the compressive energy times a quintic smoothstep S(-ln J_e / eps). It is unchanged for
            # ln J_e <= -eps and for ln J_e >= 0, and its second derivative (heat capacity, thermoelastic coupling)
            # is continuous at J_e = 1. Derivatives are with respect to ln J_e.
            eps=p.split_smoothing;W0=1.5*p.mu*np.expm1(2*log/3)-p.mu*log+.5*p.lam*log**2
            q0=p.mu*np.expm1(2*log/3)+p.lam*log;K0=2*p.mu*z/3+p.lam
            r=np.clip(-log/eps,0.,1.);S=r**3*(10-15*r+6*r*r);dS=30*r*r*(1-r)**2;d2S=60*r*(1-r)*(1-2*r)
            Wminus=W0*S;qminus=q0*S-W0*dS/eps;Kminus=K0*S-2*q0*dS/eps+W0*d2S/eps**2
        elastic+=(1-scale)*Wminus
        correction=(1-scale)*qminus
        P+=correction[...,None,None]*invT;M+=correction[...,None,None]*np.eye(3)
        trM+=3*correction
    if mechanical is not None:
        mm=np.broadcast_to(np.asarray(mechanical,dtype=float),log.shape)
        elastic=mm*elastic;P=mm[...,None,None]*P;M=mm[...,None,None]*M;trM=mm*trM
        elastic0=mm*elastic0;trM0=mm*trM0;Wminus=mm*Wminus;qminus=mm*qminus;Kminus=mm*Kminus
        scale=mm*scale
    psi = (elastic+(c-p.c0)**2/2+eta**2*(1-eta)**2
           +p.cv*(T-p.T0-T*np.log(T/p.T0)))
    entropy = p.cv*np.log(T/p.T0)+p.alpha_T*trM
    thermal_factor=scale*(2*p.mu*trace+9*p.lam)+9*(1-scale)*Kminus
    PT=-p.alpha_T*(scale[...,None,None]*(2*p.mu*FA+3*p.lam*invT)
        +3*((1-scale)*Kminus)[...,None,None]*invT)
    capacity = p.cv-T*p.alpha_T**2*thermal_factor
    if not np.isfinite(capacity).all() or np.any(capacity <= 0):
        raise ValueError('Nonpositive effective heat capacity.')
    result = dict(energy=psi, elastic=elastic, P=P, M=M,
                  sigma=(P@F.swapaxes(-1, -2))/J[..., None, None],
                  gc=c-p.c0-p.alpha_c*trM,
                  geta=2*eta*(1-eta)*(1-2*eta)-p.alpha_eta*6*eta*(1-eta)*trM,
                  entropy=entropy, internal_energy=psi+T*entropy,
                  heat_capacity=capacity,J=J,Je=Je,elastic_C=Ce,elastic_trace=trace,
                  elastic_FA=FA,inverse_F_transpose=invT,elastic_scale=scale,
                  undamaged_elastic=elastic0,undamaged_M_trace=trM0,thermal_factor=thermal_factor,
                  damage_elastic=elastic0-Wminus,damage_M_trace=trM0-3*qminus,PT=PT,
                  compressive_bulk_energy=Wminus)
    if tangent:
        result['tangent'] = (1. if mechanical is None else mm[...,None,None,None,None])*(p.mu*np.einsum('ik,...lj->...ijkl', np.eye(3), A)
            +p.lam*np.einsum('...ij,...kl->...ijkl', invT, invT)
            -t[..., None, None, None, None]*np.einsum('...il,...kj->...ijkl', invT, invT))
        result['tangent']*=scale[...,None,None,None,None]
        if elastic_split=='volumetric-deviatoric':
            result['tangent']+=(1-scale)[...,None,None,None,None]*(
                Kminus[...,None,None,None,None]*np.einsum('...ij,...kl->...ijkl',invT,invT)
                -qminus[...,None,None,None,None]*np.einsum('...il,...kj->...ijkl',invT,invT))
    return result


class FiniteElasticity2D:
    """Plane-strain equilibrium on square Q1/Q2 elements, per unit thickness."""
    def __init__(self, n, length=1., parameters=None, c=.2, eta=0., T=1., Fp=None, body=None, degree=1, elastic_scale=1., elastic_split='isotropic', edges=None, quadrature='gauss'):
        if int(n) != n or n < 2 or not np.isfinite(length) or length <= 0:
            raise ValueError('Invalid mesh.')
        if degree not in (1,2):raise ValueError('Displacement degree must be one or two.')
        self.degree=int(degree)
        if degree==1 and edges is not None:raise ValueError('Graded meshes require Q2 mechanics.')
        if degree==1 and quadrature!='gauss':raise ValueError('Alternative quadrature requires Q2 mechanics.')
        self.mesh = Fracture2D(int(n), length) if degree==1 else QuadraticMesh(int(n),length,edges,quadrature)
        if degree==1:
            self.mesh.degree=1;self.mesh.quadrature_count=4;self.mesh.weights=np.full(4,self.mesh.weight)
            uniform_geometry(self.mesh)
            self.mesh.scale_gradient=lambda values:values
            self.mesh.edge_conductances=lambda:(np.ones((int(n)-1,int(n))),np.ones((int(n),int(n)-1)))
        self.p = parameters or FiniteParameters()
        self.elastic_scale=self.sample(elastic_scale)
        self.elastic_split=elastic_split
        m = self.mesh
        self.c, self.eta, self.T = (self.sample(v) for v in (c, eta, T))
        self.Fp = self.sample(np.eye(3) if Fp is None else Fp, (3, 3))
        if (np.any(abs(self.Fp[..., :2, 2]) > 1e-12)
                or np.any(abs(self.Fp[..., 2, :2]) > 1e-12)):
            raise ValueError('Plane strain requires Fp without out-of-plane shears.')
        self.body = self.sample(np.zeros(2) if body is None else body, (2,))
        self.G = np.zeros((m.quadrature_count, 2, 2, m.dofs.shape[1]))
        self.G[:, 0, :, 0::2] = m.gradient.transpose(0, 2, 1)
        self.G[:, 1, :, 1::2] = m.gradient.transpose(0, 2, 1)
        sx = np.array([-1, 1, 1, -1]); sy = np.array([-1, -1, 1, 1])
        self.corner_gradient = np.stack((sx*(1+sy[:, None]*sy),
                                        sy*(1+sx[:, None]*sx)), axis=-1)/(2*m.h)
        if degree==2:self.corner_gradient=m.corner_gradient
        body = self.body if m.uniform else self.body*m.weight_scale[:,None,None]
        element = np.einsum('g,ga,egi->eai',m.weights,m.N,body)
        self.external = np.bincount(m.dofs.ravel(), weights=element.ravel(), minlength=2*m.node_count)
        # Validate prescribed fields before the first nonlinear solve.
        constitutive(np.eye(3), self.Fp, self.c, self.eta, self.T, self.p, False,self.elastic_scale,self.elastic_split)

    def sample(self, value, tail=()):
        x, y = self.mesh.quadrature_points.transpose(2, 0, 1)
        value = value(x, y) if callable(value) else value
        sampled = np.broadcast_to(value, x.shape+tail).copy()
        if not np.isfinite(sampled).all():
            raise ValueError('Nonfinite prescribed field.')
        return sampled

    def boundary(self):
        m = self.mesh; x, y = m.coordinates.T
        nodes = np.flatnonzero((x == 0) | (x == m.length) | (y == 0) | (y == m.length))
        return (2*nodes[:, None]+np.arange(2)).ravel()

    def evaluate(self, u, tangent=False):
        m = self.mesh; u = np.asarray(u, dtype=float)
        if u.shape != (2*m.node_count,) or not np.isfinite(u).all():
            raise ValueError('Invalid displacement vector.')
        active = getattr(self, 'active', None)  # Section 64: deleted cells carry no mechanics and no orientation check
        corner = np.eye(2)+m.scale_gradient(np.einsum('gaJ,eai->egiJ', self.corner_gradient, u.reshape(-1, 2)[m.cells]))
        cells = np.arange(m.n*m.n) if active is None else np.flatnonzero(active)
        cornerJ = determinant(corner)[cells].min(axis=1)
        minJ = float(np.min(cornerJ))
        if minJ <= 0:
            self.refused_cells = cells[cornerJ <= 0]  # Section 68: the cells that stopped the step
            raise ValueError('Element corner inversion.')
        if self.degree==1:bound=minJ
        else:
            bounds=m.orientation_lower_bound(u, active, getattr(self, 'subdivide', 0), per_cell=True);bound=float(np.min(bounds))
            if bound<=0:self.refused_cells=cells[bounds<=0]
        if bound<=0:raise ValueError('Element orientation not certified by Bernstein bound.')
        F = np.broadcast_to(np.eye(3), (m.n*m.n, m.quadrature_count, 3, 3)).copy()
        F[..., :2, :2] += m.scale_gradient(np.einsum('gijd,ed->egij', self.G, u[m.dofs]))
        if active is not None:
            F[~active] = np.eye(3)
        # Section 71: discrete-crack cells carry a reference reset R per point, so the constitutive law sees F_eff = F R.
        # The energy is W(F R); its F-derivative is P_eff R^T and the tangent contracts both reference indices with R.
        cells_r, R = getattr(self, 'reset_cells', None), getattr(self, 'reset', None)
        F_eff = F
        if cells_r is not None and len(cells_r):
            F_eff = F.copy(); F_eff[cells_r] = F[cells_r]@R
        local = constitutive(F_eff, self.Fp, self.c, self.eta, self.T, self.p, tangent,self.elastic_scale,self.elastic_split,
                             None if active is None else active[:, None].astype(float))
        if cells_r is not None and len(cells_r):
            Rt = R.swapaxes(-1, -2); detR = determinant(R)
            for key in ('P', 'PT'):
                local[key] = local[key].copy(); local[key][cells_r] = local[key][cells_r]@Rt
            local['sigma'] = local['sigma'].copy(); local['sigma'][cells_r] *= detR[..., None, None]
            local['J'] = local['J'].copy(); local['J'][cells_r] /= detR
            if tangent:
                local['tangent'] = local['tangent'].copy()
                local['tangent'][cells_r] = np.einsum('...iKmL,...JK,...NL->...iJmN', local['tangent'][cells_r], R, R)
        P = local['P'][..., :2, :2]
        if not m.uniform:P = P*m.grad_scale[:,None,None,:]*m.weight_scale[:,None,None,None]
        element = np.einsum('g,gijd,egij->ed',m.weights,self.G,P)
        internal = np.bincount(m.dofs.ravel(), weights=element.ravel(), minlength=len(u))
        energy = float(np.sum(m.cell_weights*local['energy']))
        result = dict(energy=energy, potential=energy-float(self.external@u),
                      elastic_energy=float(np.sum(m.cell_weights*local['elastic'])),
                      force=internal-self.external, F=F, local=local,
                      min_corner_J=minJ,min_J_bound=bound)
        if tangent:
            C = local['tangent'][..., :2, :2, :2, :2]
            if not m.uniform:
                s = m.grad_scale
                C = C*(s[:,None,None,:,None,None]*s[:,None,None,None,None,:]*m.weight_scale[:,None,None,None,None,None])
            element = np.einsum('g,gija,egijkl,gklb->eab',m.weights,self.G,C,self.G)
            result['stiffness'] = m.matrix(element, m.dofs, len(u))
        return result

    def equilibrate(self, u, fixed, values, tolerance=1e-9, max_iterations=60):
        u = np.array(u, dtype=float, copy=True); fixed = np.asarray(fixed, dtype=int)
        if tolerance <= 0 or not np.isfinite(tolerance) or max_iterations < 0:
            raise ValueError('Invalid solver controls.')
        u[fixed] = values
        free = np.ones(len(u), dtype=bool); free[fixed] = False
        history = []
        for iteration in range(max_iterations+1):
            state = self.evaluate(u, tangent=True)
            r = state['force']; history.append(state['potential'])
            scale = max(np.linalg.norm(r[fixed]), np.linalg.norm(self.external), 1e-12)
            error = float(np.linalg.norm(r[free])/scale)
            if error <= tolerance:
                return u, state, dict(iterations=iteration, relative_residual=error, potential=history)
            if iteration == max_iterations:
                break
            K = state['stiffness'][free][:, free]
            direction = np.zeros_like(u); direction[free] = spsolve(K, -r[free])
            slope = float(r@direction)
            if not np.isfinite(direction).all() or slope >= 0:
                direction[free] = -r[free]/np.maximum(abs(K.diagonal()), 1e-12)
                slope = float(r@direction)
            step = 1.
            for _ in range(40):
                candidate = u+step*direction
                try:
                    following = self.evaluate(candidate)
                except ValueError:
                    step *= .5
                    continue
                if following['potential'] <= state['potential']+1e-4*step*slope+1e-14:
                    break
                step *= .5
            else:
                raise RuntimeError('Finite-elasticity line search failed.')
            u = candidate
        raise RuntimeError(f'Finite equilibrium failed: relative residual {error:.3e}.')

    def save(self, path, u, load=0., step=0):
        self.evaluate(u)
        metadata = dict(n=self.mesh.n,length=self.mesh.length,parameters=asdict(self.p),degree=self.degree,elastic_split=self.elastic_split,load=load,step=step)
        if not self.mesh.uniform:metadata['edges']=[e.tolist() for e in self.mesh.edges]
        if getattr(self.mesh,'quadrature','gauss')!='gauss':metadata['quadrature']=self.mesh.quadrature
        np.savez_compressed(path, u=u, c=self.c, eta=self.eta, T=self.T, Fp=self.Fp,
                            body=self.body, elastic_scale=self.elastic_scale, metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            clock = {k: metadata.pop(k) for k in ('load', 'step')}
            fields = {k: data[k].copy() for k in ('c', 'eta', 'T', 'Fp', 'body')}
            fields['elastic_scale']=data['elastic_scale'].copy() if 'elastic_scale' in data else 1.
            model = cls(parameters=FiniteParameters(**metadata.pop('parameters')), **metadata, **fields)
            return model, data['u'].copy(), clock
