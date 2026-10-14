"""Thread-parallel kernels for scaling measurements.

The Allen-Cahn kernel evaluates the same no-flux finite-volume force as
DoubleWell.force on a nonperiodic tensor grid, row-parallel with no reductions,
so its result does not depend on the thread count. The Cahn-Hilliard step is
solvers.ch_step with a threaded FFT.
"""

import numpy as np
import numba
import scipy.fft
from numba import njit, prange


@njit(parallel=True, cache=True)
def _ac_update(v, dx, dy, cx, cy, W, kx, ky, rate, out):
    ny, nx = v.shape
    for i in prange(ny):
        for j in range(nx):
            vij = v[i, j]
            lap = 0.0
            if j < nx - 1:
                lap += kx * (v[i, j + 1] - vij) / cx[j] / dx[j]
            if j > 0:
                lap -= kx * (vij - v[i, j - 1]) / cx[j - 1] / dx[j]
            if i < ny - 1:
                lap += ky * (v[i + 1, j] - vij) / cy[i] / dy[i]
            if i > 0:
                lap -= ky * (vij - v[i - 1, j]) / cy[i - 1] / dy[i]
            out[i, j] = vij - rate * (W * (vij * vij * vij - vij) - lap)


def set_threads(count):
    numba.set_num_threads(int(count))


def ac_step(grid, v, model, dt, mobility=1.0, out=None):
    """Explicit AC step equal to solvers.ac_step on a nonperiodic grid."""
    if grid.periodic:
        raise ValueError("The parallel kernel implements no-flux boundaries only.")
    out = np.empty_like(v) if out is None else out
    _ac_update(v, grid.dx, grid.dy, np.diff(grid.x), np.diff(grid.y), float(model.W),
               float(model.k[0]), float(model.k[1]), float(dt * mobility), out)
    return out


def ch_step(grid, c, dt, W=1.0, kappa=1.0, mobility=1.0, stabilization=2.0, symbol=None, workers=1):
    """solvers.ch_step with scipy's threaded FFT."""
    lam = grid.spectrum() if symbol is None else symbol
    c_hat = scipy.fft.fft2(c, workers=workers)
    nonlinear = scipy.fft.fft2(W * (c ** 3 - c), workers=workers)
    result = (c_hat + dt * mobility * lam * (stabilization * c_hat - nonlinear)) / (1 + dt * mobility * lam * (stabilization + kappa * lam))
    return scipy.fft.ifft2(result, workers=workers).real
