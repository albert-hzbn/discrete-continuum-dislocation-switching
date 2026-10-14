"""Section 57: solute segregation to a diffuse planar interface, with exact discrete thermodynamics."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Segregation1D:
    n: int = 160
    length: float = 40.0
    W: float = 1.0
    kappa: float = 1.0
    theta: float = .1
    binding: float = .3
    mobility: float = 10.0
    phase_mobility: float = 1.0

    @property
    def h(self):
        return self.length / self.n

    @property
    def x(self):
        return (np.arange(self.n) + .5) * self.h - self.length / 2

    def initial(self, mean_solute, width=np.sqrt(2)):
        if not 0 < mean_solute < 1:
            raise ValueError("Mean solute fraction must lie in (0, 1).")
        return np.tanh(self.x / width), np.full(self.n, float(mean_solute))

    def mixing(self, c):
        return self.theta * (c * np.log(c) + (1 - c) * np.log1p(-c))

    def energy(self, eta, c):
        bulk = self.W * (1 - eta ** 2) ** 2 / 4 + self.mixing(c) - self.binding * c * (1 - eta ** 2)
        return float(self.h * bulk.sum() + self.kappa / 2 * np.sum(np.diff(eta) ** 2) / self.h)

    def phase_force(self, eta, c):
        """(1/h) dF/d eta_i, with no-flux ends."""
        lap = np.zeros_like(eta)
        d = np.diff(eta) / self.h ** 2
        lap[:-1] += d
        lap[1:] -= d
        return self.W * (eta ** 3 - eta) + 2 * self.binding * c * eta - self.kappa * lap

    def potential(self, eta, c):
        """(1/h) dF/d c_i."""
        return self.theta * (np.log(c) - np.log1p(-c)) - self.binding * (1 - eta ** 2)

    def rates(self, eta, c):
        mu = self.potential(eta, c)
        face = (c[1:] + c[:-1]) / 2
        flux = -self.mobility * face * (1 - face) * np.diff(mu) / self.h
        dc = np.zeros_like(c)
        dc[:-1] -= flux / self.h
        dc[1:] += flux / self.h
        return -self.phase_mobility * self.phase_force(eta, c), dc, mu

    def equilibrate(self, eta, c, dt, tolerance=1e-10, max_steps=4_000_000, check=2000):
        mass0, energy = self.h * c.sum(), [self.energy(eta, c)]
        for step in range(1, max_steps + 1):
            deta, dc, mu = self.rates(eta, c)
            eta, c = eta + dt * deta, c + dt * dc
            if np.any(c <= 0) or np.any(c >= 1):
                raise RuntimeError("Solute fraction left (0, 1); reduce the time step.")
            if step % check == 0:
                energy.append(self.energy(eta, c))
                mu = self.potential(eta, c)
                if max(np.ptp(mu), np.max(np.abs(self.phase_force(eta, c)))) < tolerance:
                    break
        else:
            raise RuntimeError("Equilibrium tolerance not reached.")
        return eta, c, dict(steps=step, time=step * dt, mass_drift=abs(self.h * c.sum() - mass0) / mass0,
                            max_energy_increase=float(max(0.0, np.max(np.diff(energy)))), energies=len(energy))

    def isotherm(self, eta, mu):
        return 1 / (1 + np.exp(-(mu + self.binding * (1 - eta ** 2)) / self.theta))

    def excess(self, eta, c):
        """Interface tension gamma = Omega - L omega_b and adsorption Gamma at the uniform potential."""
        mu = float(np.mean(self.potential(eta, c)))
        cb = 1 / (1 + np.exp(-mu / self.theta))
        omega_b = float(self.mixing(np.array(cb))) - mu * cb
        gamma = self.energy(eta, c) - mu * self.h * c.sum() - self.length * omega_b
        return dict(mu=mu, bulk_solute=float(cb), gamma=float(gamma), adsorption=float(self.h * np.sum(c - cb)))
