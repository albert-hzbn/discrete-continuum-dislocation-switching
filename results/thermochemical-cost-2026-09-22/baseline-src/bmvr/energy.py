"""Discrete energies and their volume-weighted derivatives."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DoubleWell:
    W: float = 1.0
    k: tuple = (1.0, 1.0)

    def value(self, grid, v):
        return grid.integrate(self.W * (1 - v * v) ** 2 / 4) + grid.gradient_energy(v, self.k)

    def force(self, grid, v):
        return self.W * (v ** 3 - v) - grid.lap(v, self.k)


def grain_energy(grid, phases, W=1.0, k=1.0):
    return grid.integrate(np.sum(.5 * W * phases * (1 - phases), axis=0)) + grid.gradient_energy(phases, (k, k))


def grain_force(grid, phases, W=1.0, k=1.0):
    return W * (.5 - phases) - grid.lap(phases, (k, k))


def pore_energy(grid, n, p, chi=1.0, W=1.0, k=1.0, neq=0.0, formation=0.0):
    return grid.integrate(chi * (n - neq) ** 2 / 2 + W * p ** 2 * (1 - p) ** 2 + formation * p) + grid.gradient_energy(p, (k, k))


def pore_forces(grid, n, p, chi=1.0, W=1.0, k=1.0, neq=0.0, formation=0.0):
    return chi * (n - neq), 2 * W * p * (1 - p) * (1 - 2 * p) - grid.lap(p, (k, k)) + formation
