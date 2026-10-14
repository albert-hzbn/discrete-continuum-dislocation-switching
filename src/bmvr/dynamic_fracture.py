"""Section 66: explicit elastodynamics with AT2 damage (Section 36 energy) for dynamic crack branching.

Plane strain on the Section 36 Q1 grid with the spectral tension/compression split. The displacement advances by
explicit central differences with a lumped mass; after each step the damage minimizes the Section 36 functional at the
new displacement subject to irreversibility (the staggered scheme of dynamic phase-field fracture). Tractions are
applied on the top and bottom edges; no displacement is prescribed, and the self-equilibrated load leaves no rigid motion.
"""

import numpy as np
from .fracture2d import Fracture2D, FractureParameters, bound_quadratic


class DynamicFracture2D:
    def __init__(self, n, parameters=None, density=1.0, traction=.05, notch=.5, nx=None):
        self.base = Fracture2D(n, parameters=parameters or FractureParameters(), nx=nx)
        b = self.base
        self.density, self.traction, self.notch = density, float(traction), float(notch)
        counts = np.bincount(b.cells.ravel(), minlength=b.node_count)
        self.mass = np.repeat(density * b.h ** 2 * counts / 4, 2)
        x, y = b.coordinates.T
        self.external = np.zeros(2 * b.node_count)
        for edge, sign in ((y == b.length, 1.0), (y == 0, -1.0)):
            nodes = np.flatnonzero(edge)
            weight = np.where((x[nodes] == 0) | (x[nodes] == b.width), b.h / 2, b.h)
            self.external[2 * nodes + 1] += sign * self.traction * weight

    def initial(self):
        b = self.base
        x, y = b.coordinates.T
        distance = np.sqrt(np.maximum(x - self.notch, 0) ** 2 + (y - b.length / 2) ** 2)
        return dict(u=np.zeros(2 * b.node_count), v=np.zeros(2 * b.node_count), q=np.exp(-distance / b.p.ell), time=0.0)

    def energies(self, state):
        e = self.base.evaluate(state['u'], state['q'])
        return dict(kinetic=float(.5 * np.sum(self.mass * state['v'] ** 2)), elastic=e['elastic_energy'],
                    fracture=e['fracture_energy'], work_potential=float(self.external @ state['u']))

    def step(self, state, dt):
        """Explicit central difference for u, then the irreversible damage minimization at the new u."""
        b = self.base
        force = b.evaluate(state['u'], state['q'])['force']
        a = (self.external - force) / self.mass
        v_half = state['v'] + .5 * dt * a
        u = state['u'] + dt * v_half
        A, rhs = b.damage_system(u)
        q, info = bound_quadratic(A, rhs, state['q'], state['q'])
        q = np.clip(q, state['q'], 1.0)
        force = b.evaluate(u, q)['force']
        v = v_half + .5 * dt * (self.external - force) / self.mass
        return dict(u=u, v=v, q=q, time=state['time'] + dt), info

    def centerline_tip(self, q, level=.8):
        b = self.base
        grid = q.reshape(b.nx + 1, b.n + 1)  # [x index, y index]
        broken = grid >= level
        columns = np.flatnonzero(broken.any(axis=1))
        return float(columns.max() * b.h) if len(columns) else 0.0

    def branches(self, q, x, level=.8):
        """Number of separate damage bands crossing the vertical line at x."""
        b = self.base
        column = q.reshape(b.nx + 1, b.n + 1)[int(round(x / b.h)), :] >= level
        return int(np.count_nonzero(np.diff(np.r_[0, column.astype(int), 0]) == 1))


class ThermalDynamicFracture2D(DynamicFracture2D):
    """Section 81: dynamic AT2 fracture with the thermochemical model's fracture-heat coupling.

    The fracture energy dissipated in a step, G_c(T) times the increase of the crack density, becomes heat, allocated to cells
    in proportion to their damage growth so local heat is never negative (capacity c_v per volume); heat conducts between
    cells; and toughness softens with temperature,
    G_c(T) = G_c0 max(1 - softening (T - T0), floor). Heat booked at the toughness used in the step keeps
    H = K + E_el + Q - W conserved to the accuracy of the staggered scheme.
    """

    def __init__(self, n, parameters=None, density=1.0, traction=.05, notch=.5, nx=None, cv=.1, softening=1., conductivity=1e-4, T0=1., floor=.1):
        super().__init__(n, parameters, density, traction, notch, nx)
        self.cv, self.softening, self.conductivity, self.T0, self.floor = cv, softening, conductivity, T0, floor
        self.G0 = float(np.mean(self.base.toughness)); b = self.base
        self.volume = b.h**2

    def initial(self):
        s = super().initial(); s['T'] = np.full(self.base.nx*self.base.n, self.T0); return s

    def toughness(self, T):
        return self.G0*np.maximum(1-self.softening*(T-self.T0), self.floor)

    def crack_density(self, q):
        b = self.base; d = q[b.cells]@b.N.T; g = np.einsum('gai,ea->egi', b.gradient, q[b.cells])
        return b.weight*np.sum((d**2/b.p.ell+b.p.ell*np.sum(g**2, axis=-1))/2, axis=1)

    def energies(self, state):
        e = super().energies(state); e['heat'] = float(self.cv*self.volume*np.sum(state['T']-self.T0)); return e

    def step(self, state, dt):
        b = self.base; G = self.toughness(state['T']); b.toughness = np.broadcast_to(G[:, None], b.toughness.shape).copy()
        f0 = self.crack_density(state["q"]); new, info = super().step(state, dt)
        dissipated = float(np.sum(G*(self.crack_density(new["q"])-f0)))  # fracture energy dissipated in this step
        dq = np.maximum(new['q']-state['q'], 0.)[b.cells].sum(axis=1)  # damage growth per cell (never negative)
        share = dq/dq.sum() if dq.sum() > 0 else np.zeros_like(dq)
        T = state['T']+dissipated*share/(self.cv*self.volume)  # positive allocation: heat where damage grows
        grid = T.reshape(b.nx, b.n); lap = np.zeros_like(grid)
        lap[1:] += grid[:-1]-grid[1:]; lap[:-1] += grid[1:]-grid[:-1]; lap[:, 1:] += grid[:, :-1]-grid[:, 1:]; lap[:, :-1] += grid[:, 1:]-grid[:, :-1]
        new['T'] = (grid+dt*self.conductivity*lap/b.h**2).ravel()  # no-flux conduction; conserves heat exactly
        return new, info
