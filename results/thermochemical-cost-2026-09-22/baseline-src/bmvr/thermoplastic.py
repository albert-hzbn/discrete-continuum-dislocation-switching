"""Inertial thermoplasticity on Q1/Q2 mechanics and cell heat volumes."""
from dataclasses import asdict, dataclass
import json
import numpy as np
from .finite_elasticity import FiniteElasticity2D, FiniteParameters


@dataclass(frozen=True)
class ThermalPlasticParameters:
    density: float = 1.
    conductivity: float = .4
    hardening: tuple = (.2, .3)
    yield_stress: tuple = (.015, .025)
    drag: tuple = (1., 1.7)

    def __post_init__(self):
        if not np.isfinite(self.density) or self.density <= 0 or not np.isfinite(self.conductivity) or self.conductivity < 0:
            raise ValueError('Invalid density or conductivity.')
        for name in ('hardening', 'yield_stress', 'drag'):
            a = np.asarray(getattr(self, name))
            if a.shape != (2,) or not np.isfinite(a).all() or np.any(a < 0) or (name == 'drag' and np.any(a == 0)):
                raise ValueError(f'Invalid two-system {name}.')


@dataclass
class ThermalPlasticState:
    u: np.ndarray
    v: np.ndarray
    T: np.ndarray
    Fp: np.ndarray
    gamma: np.ndarray


def exponential_slip(L, dt):
    """Exact exponential for the two off-diagonal plane-strain generators."""
    L = np.asarray(L)
    if L.shape[-2:] != (3, 3) or not np.isfinite(L).all() or not np.isfinite(dt):
        raise ValueError('Invalid slip generator or interval.')
    other = L.copy(); other[..., 0, 1] = 0.; other[..., 1, 0] = 0.
    if np.any(abs(other) > 1e-14):
        raise ValueError('Generator must contain only the two declared slip systems.')
    squared = dt*dt*L[..., 0, 1]*L[..., 1, 0]
    z = np.sqrt(abs(squared)); positive = squared >= 0
    diagonal = np.where(positive, np.cosh(z), np.cos(z))
    numerator = np.where(positive, np.sinh(z), np.sin(z))
    ratio = np.divide(numerator, z, out=np.ones_like(z), where=z != 0)
    result = dt*ratio[..., None, None]*L
    result[..., 0, 0] = result[..., 1, 1] = diagonal
    result[..., 2, 2] = 1.
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite plastic exponential.')
    return result


def conductive_power(T, conductivity):
    """Insulated square-cell heat powers and exact pairwise entropy production."""
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or not np.isfinite(T).all() or np.any(T <= 0) or not np.isfinite(conductivity) or conductivity < 0:
        raise ValueError('Invalid conduction state.')
    power = np.zeros_like(T); production = 0.
    for axis in (0, 1):
        left = [slice(None), slice(None)]; right = left.copy()
        left[axis] = slice(None, -1); right[axis] = slice(1, None)
        left, right = tuple(left), tuple(right)
        delta = T[right]-T[left]; flux = conductivity*delta
        power[left] += flux; power[right] -= flux
        production += float(np.sum(conductivity*delta**2/(T[left]*T[right])))
    return power, production


class ThermalPlastic2D:
    def __init__(self, n, length=1., mechanical=None, transport=None, c=.2, eta=0., fixed=None, values=None, degree=1):
        self.mechanics = FiniteElasticity2D(n,length,mechanical,c=c,eta=eta,degree=degree)
        self.p = transport or ThermalPlasticParameters(); m = self.mechanics.mesh
        self.fixed = self.mechanics.boundary() if fixed is None else np.asarray(fixed, dtype=int).copy()
        if (self.fixed.ndim != 1 or len(np.unique(self.fixed)) != len(self.fixed)
                or np.any(self.fixed < 0) or np.any(self.fixed >= 2*m.node_count)):
            raise ValueError('Invalid fixed degrees of freedom.')
        self.values = np.zeros(len(self.fixed)) if values is None else np.asarray(values, dtype=float).copy()
        if self.values.shape != self.fixed.shape or not np.isfinite(self.values).all():
            raise ValueError('Invalid fixed displacements.')
        mass_element = np.broadcast_to(self.p.density*np.einsum('g,ga->a',m.weights,m.N),m.cells.shape)
        nodal = np.bincount(m.cells.ravel(), weights=mass_element.ravel(), minlength=m.node_count)
        self.mass = np.repeat(nodal, 2)
        self.S = np.zeros((2, 3, 3)); self.S[0, 0, 1] = 1.; self.S[1, 1, 0] = 1.

    def initial(self, T=1., u=None, equilibrate=False):
        m = self.mechanics.mesh
        if callable(T):
            centers = m.coordinates[m.cells].mean(axis=1); T = T(*centers.T)
        state = ThermalPlasticState(np.zeros(2*m.node_count) if u is None else np.array(u, dtype=float, copy=True),
            np.zeros(2*m.node_count), np.broadcast_to(T, (m.n*m.n,)).copy(),
            np.broadcast_to(np.eye(3), (m.n*m.n,m.quadrature_count,3,3)).copy(),np.zeros((m.n*m.n,m.quadrature_count,2)))
        state.u[self.fixed] = self.values
        self.install(state)
        if equilibrate:
            state.u, _, _ = self.mechanics.equilibrate(state.u, self.fixed, self.values)
        self.rate(state)
        return state

    def install(self, state):
        m = self.mechanics.mesh
        shapes = dict(u=(2*m.node_count,), v=(2*m.node_count,), T=(m.n*m.n,),
                      Fp=(m.n*m.n,m.quadrature_count,3,3),gamma=(m.n*m.n,m.quadrature_count,2))
        for key, shape in shapes.items():
            a = getattr(state, key)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'Invalid state {key}.')
        if np.any(state.T <= 0) or not np.allclose(state.u[self.fixed], self.values, atol=1e-13, rtol=0) or np.any(abs(state.v[self.fixed]) > 1e-13):
            raise ValueError('Invalid temperature or fixed-boundary state.')
        if np.any(abs(state.Fp[..., :2, 2]) > 1e-12) or np.any(abs(state.Fp[..., 2, :2]) > 1e-12):
            raise ValueError('Plastic map violates plane strain.')
        self.mechanics.T = np.broadcast_to(state.T[:,None],(m.n*m.n,m.quadrature_count))
        self.mechanics.Fp = state.Fp

    def rate(self, state):
        self.install(state); model = self.mechanics; m = model.mesh; p = model.p
        mechanical = model.evaluate(state.u); local = mechanical['local']; F = mechanical['F']
        q = np.einsum('egij,sij->egs', local['M'], self.S)-np.asarray(self.p.hardening)*state.gamma
        slip = np.sign(q)*np.maximum(abs(q)-self.p.yield_stress, 0)/self.p.drag
        Lp = np.einsum('egs,sij->egij', slip, self.S)
        dissipation = np.sum(q*slip, axis=-1)
        invFp = np.linalg.inv(state.Fp)
        growth = np.exp(p.alpha_c*(model.c-p.c0)+p.alpha_eta*model.eta**2*(3-2*model.eta)+p.alpha_T*(model.T-p.T0))
        B = invFp/growth[..., None, None]; Fe = F@B; Ce = Fe.swapaxes(-1, -2)@Fe
        invFT = np.linalg.inv(F).swapaxes(-1, -2)
        PT = -p.alpha_T*(2*p.mu*F@(B@B.swapaxes(-1, -2))+3*p.lam*invFT)
        chi = 2*p.alpha_T*p.mu*np.einsum('egij,egij->eg', Ce, Lp)
        Fdot = np.zeros_like(F)
        Fdot[..., :2, :2] = np.einsum('gijd,ed->egij', model.G, state.v[m.dofs])
        mixed = np.einsum('egij,egij->eg', PT, Fdot)+chi
        heat, conductive_entropy = conductive_power(state.T.reshape(m.n, m.n), self.p.conductivity)
        capacity = np.sum(m.weights*local['heat_capacity'],axis=1)
        Tdot = (heat.ravel()+np.sum(m.weights*(dissipation+model.T*mixed),axis=1))/capacity
        acceleration = -mechanical['force']/self.mass; acceleration[self.fixed] = 0.
        hardening = float(np.sum(m.weights[None,:,None]*np.asarray(self.p.hardening)*state.gamma**2)/2)
        kinetic = float(np.sum(self.mass*state.v**2)/2)
        internal = float(np.sum(m.weights*local['internal_energy']))+hardening
        diagnostics = dict(total_energy=internal+kinetic, internal_energy=internal, kinetic_energy=kinetic,
            elastic_energy=mechanical['elastic_energy'], hardening_energy=hardening,
            free_energy=mechanical['energy']+hardening,entropy=float(np.sum(m.weights*local['entropy'])),
            entropy_production=conductive_entropy+float(np.sum(m.weights*dissipation/model.T)),
            conductive_entropy_production=conductive_entropy,plastic_power=float(np.sum(m.weights*dissipation)),
            min_T=float(np.min(state.T)), max_T=float(np.max(state.T)),
            min_J=mechanical['min_J_bound'],min_capacity=float(np.min(local['heat_capacity'])),
            plastic_volume_error=float(np.max(abs(np.linalg.det(state.Fp)-1))),
            heat_power_sum=float(np.sum(heat)), max_speed=float(np.max(abs(state.v))),
            max_slip=float(np.max(abs(state.gamma))))
        return dict(u=state.v.copy(), v=acceleration, T=Tdot, gamma=slip, Lp=Lp,
                    PT=PT, chi=chi, Fdot=Fdot, diagnostics=diagnostics,
                    local=local, elastic_trace=np.trace(Ce,axis1=-2,axis2=-1), capacity=capacity,
                    cell_heat_power=heat.ravel(),
                    cell_mechanical_power=np.sum(m.weights*np.einsum('egij,egij->eg',local['P'],Fdot),axis=1),
                    cell_internal_energy=np.sum(m.weights*(local['internal_energy']+
                        np.sum(np.asarray(self.p.hardening)*state.gamma**2,axis=-1)/2),axis=1))

    def _advance(self, state, rate, dt):
        fields={key:value+dt*rate[key] for key,value in vars(state).items() if key!='Fp'}
        fields['Fp']=exponential_slip(rate['Lp'],dt)@state.Fp
        return type(state)(**fields)

    def step(self, state, dt, tolerance=1e-12, max_iterations=40):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite time step required.')
        if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
            raise ValueError('Invalid midpoint solve controls.')
        self.rate(state)
        middle = type(state)(**{key:value.copy() for key,value in vars(state).items()})
        for iteration in range(max_iterations):
            mid = self.rate(middle)
            following = self._advance(state,mid,.5*dt)
            error = max(float(np.max(abs(getattr(following,key)-getattr(middle,key)))) for key in vars(state))
            middle = following
            if error <= tolerance:
                break
        else:
            raise RuntimeError(f'Implicit thermoplastic midpoint did not converge: {error:.3e}.')
        mid = self.rate(middle)
        result = self._advance(state,mid,dt)
        final = self.rate(result)
        final['diagnostics']['midpoint_iterations'] = iteration+1
        final['diagnostics']['midpoint_residual'] = error
        final['diagnostics']['midpoint_entropy_production'] = mid['diagnostics']['entropy_production']
        final['diagnostics']['midpoint_plastic_power'] = mid['diagnostics']['plastic_power']
        return result, final['diagnostics']

    def save(self, path, state, time=0., step=0):
        self.rate(state); m = self.mechanics.mesh
        metadata = dict(n=m.n,length=m.length,degree=self.mechanics.degree,mechanical=asdict(self.mechanics.p),
                        transport=asdict(self.p), time=time, step=step)
        np.savez_compressed(path, **vars(state), c=self.mechanics.c, eta=self.mechanics.eta,
                            fixed=self.fixed, values=self.values, metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata'])); clock = {k:meta.pop(k) for k in ('time', 'step')}
            model = cls(mechanical=FiniteParameters(**meta.pop('mechanical')),
                transport=ThermalPlasticParameters(**meta.pop('transport')),
                c=data['c'].copy(), eta=data['eta'].copy(), fixed=data['fixed'], values=data['values'], **meta)
            state = ThermalPlasticState(**{key:data[key].copy() for key in ('u','v','T','Fp','gamma')})
            model.rate(state)
            return model, state, clock
