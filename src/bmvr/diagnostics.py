"""Physical observables and numerical errors, without pass-by-assumption defaults."""

import numpy as np
from .adaptivity import transfer


def radius(grid, v):
    return float(np.sqrt(max(0, grid.integrate((v + 1) / 2)) / np.pi))


def field_errors(reference_grid, reference, grid, field):
    mapped = transfer(grid, reference_grid, field)
    d = mapped - reference
    return {"l1": reference_grid.integrate(np.abs(d)) / reference_grid.area,
            "l2": float(np.sqrt(reference_grid.integrate(d * d) / reference_grid.area)),
            "linf": float(np.max(np.abs(d))),
            "relative_l2": float(np.sqrt(reference_grid.integrate(d * d) / max(reference_grid.integrate(reference ** 2), 1e-30)))}


def energy_increase(values):
    return float(max(0.0, np.max(np.diff(values)))) if len(values) > 1 else 0.0


def structure_factor(grid, c):
    centered = c - c.mean()
    power = np.abs(np.fft.fft2(centered)) ** 2 / c.size
    kx = 2 * np.pi * np.fft.fftfreq(len(grid.x), grid.dx[0])
    ky = 2 * np.pi * np.fft.fftfreq(len(grid.y), grid.dy[0])
    k = np.hypot(kx[None], ky[:, None])
    power[0, 0] = 0
    length = 2 * np.pi * power.sum() / max(float(np.sum(k * power)), 1e-30)
    return power, float(length)


def interface_radii(grid, v, angles, center=(0.0, 0.0), level=0.0):
    """Radius of the first level crossing along each ray, by linear interpolation (uniform grids)."""
    from scipy.ndimage import map_coordinates
    if not (np.allclose(grid.dx, grid.dx[0]) and np.allclose(grid.dy, grid.dy[0])):
        raise ValueError("Ray sampling assumes a uniform grid.")
    step = grid.hmin / 4
    r = np.arange(0.0, min(grid.xe[-1] - center[0], grid.ye[-1] - center[1]), step)
    x = center[0] + r[None, :] * np.cos(angles)[:, None]
    y = center[1] + r[None, :] * np.sin(angles)[:, None]
    values = map_coordinates(v, [(y - grid.y[0]) / grid.dy[0], (x - grid.x[0]) / grid.dx[0]], order=1, mode="nearest") - level
    radii = np.full(len(angles), np.nan)
    for a, row in enumerate(values):
        cross = np.flatnonzero((row[:-1] > 0) & (row[1:] <= 0))
        if len(cross):
            j = cross[0]
            radii[a] = r[j] + row[j] / (row[j] - row[j + 1]) * step
    return radii
