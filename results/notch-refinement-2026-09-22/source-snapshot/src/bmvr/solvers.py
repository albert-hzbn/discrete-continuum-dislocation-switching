"""Small, explicit verification solvers with a uniform spectral CH reference."""

import numpy as np
from .energy import grain_force, grain_energy


def ac_step(grid, v, model, dt, mobility=1.0):
    return v - dt * mobility * model.force(grid, v)


def ch_step(grid, c, dt, W=1.0, kappa=1.0, mobility=1.0, stabilization=2.0, symbol=None):
    lam = grid.spectrum() if symbol is None else symbol
    c_hat = np.fft.fft2(c)
    nonlinear = np.fft.fft2(W * (c ** 3 - c))
    result = (c_hat + dt * mobility * lam * (stabilization * c_hat - nonlinear)) / (1 + dt * mobility * lam * (stabilization + kappa * lam))
    return np.fft.ifft2(result).real


def project_simplex(phases):
    """Euclidean projection on nonnegative fractions summing to one."""
    ordered = np.sort(phases, axis=0)[::-1]
    cumulative = np.cumsum(ordered, axis=0) - 1
    ranks = np.arange(1, len(phases) + 1)[:, None, None]
    rho = np.sum(ordered - cumulative / ranks > 0, axis=0) - 1
    theta = np.take_along_axis(cumulative / ranks, rho[None], axis=0)[0]
    return np.maximum(phases - theta, 0)


def grain_step(grid, phases, dt):
    force = grain_force(grid, phases)
    old = grain_energy(grid, phases)
    for _ in range(24):
        trial = project_simplex(phases - dt * force)
        if grain_energy(grid, trial) <= old + 1e-12 * max(1, abs(old)):
            return trial, dt
        dt *= 0.5
    raise RuntimeError("Grain step failed its energy check.")
