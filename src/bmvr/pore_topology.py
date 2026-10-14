"""Section 58: pore collapse, coalescence and a conserving nucleation event rule on the Section 52 pore model."""

import numpy as np
from scipy import ndimage
from .grid import Grid
from .energy import pore_energy, pore_forces

GAMMA = np.sqrt(2) / 6
A = np.sqrt(2)


class PoreModel:
    def __init__(self, length=64.0, h=.5, k_ex=.5, power=1):
        self.grid = Grid.uniform(length, int(round(length / h)), periodic=True)
        self.length, self.h, self.k_ex, self.power, self.dt = length, h, k_ex, power, .2 * h * h
        self.floor = -np.inf  # smallest admissible n for evaluating the energy (Section 75 entropy form needs n > 0)

    def distance(self, center):
        d = [(c - x0 + self.length / 2) % self.length - self.length / 2 for c, x0 in ((self.grid.X, center[0]), (self.grid.Y, center[1]))]
        return np.hypot(*d)

    def pore(self, center, radius):
        return 1 / (1 + np.exp(A * (self.distance(center) - radius)))

    def energy(self, n, p):
        return pore_energy(self.grid, n, p)

    def forces(self, n, p):
        return pore_forces(self.grid, n, p)

    def inventory(self, n, p):
        return self.grid.integrate(n + p)

    def step(self, n, p, energy, dt=None, limiter=True):
        """The Section 52 explicit step with energy and bound backtracking, plus the Section 58 availability limiter.

        The quadratic vacancy energy has no entropic barrier at n=0, so a vacancy-consuming exchange (rate<0) can
        drive n negative for every step size. The limiter scales that exchange, cell by cell, so that n stays >= 0.
        Each cell's exchange term keeps its sign in the first-order energy change, so the step stays a descent
        direction, and n+p is conserved exactly. It is inactive wherever the bound is not reached.
        """
        dt = self.dt if dt is None else dt
        un, up = self.forces(n, p)
        rate = self.k_ex * (p * (1 - p)) ** self.power * (up - un)
        diffusion = self.grid.lap(un)
        used = dt
        for _ in range(30):
            r = rate
            nn, pp = n + used * (diffusion + rate), p - used * rate
            if limiter:
                supply = n + used * diffusion
                limited = (rate < 0) & (nn < 0) & (supply >= 0)
                # On a limited cell all of its supply converts to pore: n is set to exactly zero, avoiding roundoff below it.
                nn, pp = np.where(limited, 0.0, nn), np.where(limited, p + supply, pp)
                r = np.where(limited, -supply / used, rate)
            value = self.energy(nn, pp) if nn.min() > self.floor else np.inf
            if nn.min() >= 0 and pp.min() >= 0 and pp.max() <= 1 and value <= energy + 1e-11:
                self.limited = int(np.count_nonzero(r != rate))
                return nn, pp, value, used
            used *= .5
        raise RuntimeError("Pore step failed bounds or energy.")

    def components(self, p):
        """Connected components of P > 1/2 with periodic identification; returns labels and count."""
        labels, count = ndimage.label(p > .5)
        parent = list(range(count + 1))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for a, b in ((labels[0, :], labels[-1, :]), (labels[:, 0], labels[:, -1])):
            for i, j in zip(a, b):
                if i and j:
                    parent[find(i)] = find(j)
        roots = {find(i) for i in range(1, count + 1)}
        remap = np.array([0] + [find(i) for i in range(1, count + 1)])
        return remap[labels], len(roots)

    def critical_radius(self, mu):
        return np.inf if mu <= 0 else GAMMA / mu

    def insert(self, n, p, center, factor=1.5, exclusion=None):
        """Conserving embryo insertion (Section 58.3). Returns (n, p, record) or (None, None, record) if refused.

        exclusion (post hoc, Section 58 outcome): refuse sites whose embryo would come within this margin of an existing
        pore, because such an insertion lowers F and turns growth of that pore into a jump event.
        """
        distance = self.distance(center)
        mu = float(n[np.unravel_index(np.argmin(distance), n.shape)])
        record = dict(center=[float(c) for c in center], local_mu=mu)
        if mu <= 0:
            return None, None, dict(record, refused="undersaturated")
        radius = factor * self.critical_radius(mu)
        if exclusion is not None and np.any(p > .5) and distance[p > .5].min() < radius + exclusion:
            return None, None, dict(record, refused="overlaps_existing_pore", radius=radius)
        new_p = np.maximum(p, self.pore(center, radius))
        content = self.grid.integrate(new_p - p)
        available = self.grid.integrate(n)
        if content >= available:
            return None, None, dict(record, refused="insufficient_vacancies", radius=radius)
        new_n = n * (1 - content / available)
        depleted = self.grid.integrate(new_n) / self.grid.area
        if radius <= self.critical_radius(depleted):
            return None, None, dict(record, refused="subcritical_after_depletion", radius=radius)
        return new_n, new_p, dict(record, radius=radius, content=content, depleted_mean=depleted)


class EntropyPoreModel(PoreModel):
    """Section 75.1: dilute ideal-solution vacancy energy theta[n ln(n/n_eq) - n + n_eq] instead of the quadratic form.

    Its chemical potential theta ln(n/n_eq) diverges to -inf as n -> 0, so a vacancy-consuming exchange stops by itself:
    the natural n >= 0 barrier that the quadratic form lacks. With theta = chi n_eq it has the quadratic form's curvature
    at n_eq.
    """

    def __init__(self, n_eq=.005, theta=None, **kw):
        super().__init__(**kw)
        self.n_eq = n_eq; self.theta = n_eq if theta is None else theta; self.floor = 0.

    def energy(self, n, p):
        if np.any(n <= 0): return np.inf
        f = self.theta * (n * np.log(n / self.n_eq) - n + self.n_eq)
        return self.grid.integrate(f) + pore_energy(self.grid, np.zeros_like(n), p)

    def forces(self, n, p):
        return self.theta * np.log(n / self.n_eq), pore_forces(self.grid, n, p)[1]


class GasPoreModel(PoreModel):
    """Section 75.2: a pore holding N_g conserved gas atoms with ideal-gas free energy N_g theta_g [ln(N_g/A_p) - 1],
    A_p = integral of P. Its variation adds the uniform pressure -N_g theta_g/A_p to the pore force."""

    def __init__(self, gas=1., theta_g=1., **kw):
        super().__init__(**kw); self.gas = gas; self.theta_g = theta_g

    def pressure(self, p):
        return self.gas * self.theta_g / self.grid.integrate(p)

    def energy(self, n, p):
        area = self.grid.integrate(p)
        return pore_energy(self.grid, n, p) + (np.inf if area <= 0 else self.gas * self.theta_g * (np.log(self.gas / area) - 1))

    def forces(self, n, p):
        un, up = pore_forces(self.grid, n, p)
        return un, up - self.pressure(p)

