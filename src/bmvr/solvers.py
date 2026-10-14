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


def ch_sbdf2(grid, c, c_prev, dt, W=1.0, kappa=1.0, mobility=1.0, stabilization=2.0, symbol=None):
    """Section 79: second-order stabilized SBDF2 step for Cahn-Hilliard; c_prev=None takes a first-order step."""
    if c_prev is None:return ch_step(grid, c, dt, W, kappa, mobility, stabilization, symbol)
    lam = grid.spectrum() if symbol is None else symbol
    c_hat, p_hat = np.fft.fft2(c), np.fft.fft2(c_prev)
    n_hat, np_hat = np.fft.fft2(W * (c ** 3 - c)), np.fft.fft2(W * (c_prev ** 3 - c_prev))
    rhs = (4 * c_hat - p_hat) / (2 * dt) - mobility * lam * (2 * n_hat - np_hat - stabilization * (2 * c_hat - p_hat))
    return np.fft.ifft2(rhs / (3 / (2 * dt) + mobility * lam * (kappa * lam + stabilization))).real


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


def anisotropic_mobility(grid, v, delta, floor=1e-12):
    """Factor 1 + delta cos 4 theta_n of the normal angle theta_n (Section 56.2); 1 where grad v vanishes."""
    if not abs(delta) < 1:
        raise ValueError("A positive mobility requires |delta| < 1.")
    gy, gx = np.gradient(v, grid.y, grid.x)
    g2 = gx * gx + gy * gy
    c4 = np.divide(gx ** 4 - 6 * gx * gx * gy * gy + gy ** 4, g2 * g2, out=np.zeros_like(v), where=g2 > floor)
    return 1 + delta * c4
