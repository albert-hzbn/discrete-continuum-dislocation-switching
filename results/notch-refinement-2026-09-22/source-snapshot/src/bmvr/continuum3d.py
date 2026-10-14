"""Periodic 3D variational fields with homogeneous elasticity and crystal slip.

Small strain, fixed crystal orientation, synthetic constitutive parameters.
The Fourier mechanics requires odd grid sizes; scalar gradients use finite volumes.
"""

from dataclasses import dataclass
from itertools import permutations
import numpy as np


@dataclass(frozen=True)
class Cube:
    length: float
    n: int

    def __post_init__(self):
        if self.length <= 0 or self.n < 3 or self.n % 2 == 0:
            raise ValueError("Use a positive cube length and an odd grid size >= 3.")

    @property
    def shape(self):
        return (self.n,) * 3

    @property
    def h(self):
        return self.length / self.n

    @property
    def volume(self):
        return self.length ** 3

    def coordinates(self):
        x = (np.arange(self.n) + .5) * self.h - self.length / 2
        return np.meshgrid(x, x, x, indexing="ij")

    def integrate(self, field):
        return float(np.sum(field) * self.h ** 3)

    def lap(self, field, coefficients=(1., 1., 1.)):
        return sum(k * (np.roll(field, 1, axis) + np.roll(field, -1, axis) - 2 * field) / self.h ** 2
                   for axis, k in zip((-3, -2, -1), coefficients))

    def gradient_energy(self, field, coefficients=(1., 1., 1.)):
        return sum(k * self.integrate(((np.roll(field, -1, axis) - field) / self.h) ** 2) / 2
                   for axis, k in zip((-3, -2, -1), coefficients))


def isotropic_stress(strain, lam, shear):
    trace = np.einsum("ii...->...", strain)
    return 2 * shear * strain + lam * np.eye(3).reshape(3, 3, 1, 1, 1) * trace


class PeriodicElasticity:
    """Condense displacement for an arbitrary symmetric eigenstrain tensor."""

    def __init__(self, grid, lam=2., shear=1.):
        if shear <= 0 or 3 * lam + 2 * shear <= 0:
            raise ValueError("Elastic moduli must give positive strain energy.")
        self.grid, self.lam, self.shear = grid, lam, shear
        wave = 2 * np.pi * np.fft.fftfreq(grid.n, grid.h)
        self.k = np.array(np.meshgrid(wave, wave, wave, indexing="ij"))
        kk = np.sum(self.k ** 2, axis=0)
        self.inverse_k2 = np.divide(1., kk, out=np.zeros_like(kk), where=kk > 0)

    def derivative(self, field, axis):
        return np.fft.ifftn(1j * self.k[axis] * np.fft.fftn(field, axes=(-3, -2, -1)),
                            axes=(-3, -2, -1)).real

    def solve(self, eigenstrain, mean_strain=None):
        eigenstrain = np.asarray(eigenstrain)
        if eigenstrain.shape != (3, 3) + self.grid.shape:
            raise ValueError("Eigenstrain must have shape (3, 3, n, n, n).")
        if not np.allclose(eigenstrain, eigenstrain.swapaxes(0, 1), rtol=0, atol=1e-12):
            raise ValueError("Eigenstrain must be symmetric.")
        mean = np.zeros((3, 3)) if mean_strain is None else np.asarray(mean_strain)
        if mean.shape != (3, 3) or not np.allclose(mean, mean.T, rtol=0, atol=1e-12):
            raise ValueError("Prescribed mean strain must be a symmetric 3x3 tensor.")
        eigenstress = isotropic_stress(eigenstrain, self.lam, self.shear)
        phat = np.fft.fftn(eigenstress, axes=(-3, -2, -1))
        f = np.einsum("j...,ij...->i...", self.k, phat)
        longitudinal = self.k * np.einsum("i...,i...->...", self.k, f) * self.inverse_k2
        uhat = -1j * (f - (self.lam + self.shear) / (self.lam + 2 * self.shear) * longitudinal)
        uhat *= self.inverse_k2 / self.shear
        grad_hat = 1j * np.einsum("j...,i...->ij...", self.k, uhat)
        strain = np.fft.ifftn((grad_hat + grad_hat.swapaxes(0, 1)) / 2, axes=(-3, -2, -1)).real
        strain += mean[..., None, None, None]
        elastic_strain = strain - eigenstrain
        stress = isotropic_stress(elastic_strain, self.lam, self.shear)
        density = .5 * np.einsum("ij...,ij...->...", elastic_strain, stress)
        div_hat = 1j * np.einsum("j...,ij...->i...", self.k, np.fft.fftn(stress, axes=(-3, -2, -1)))
        divergence = np.fft.ifftn(div_hat, axes=(-3, -2, -1)).real
        return {"energy": self.grid.integrate(density), "stress": stress, "strain": strain,
                "displacement": np.fft.ifftn(uhat, axes=(-3, -2, -1)).real,
                "equilibrium_rms": float(np.sqrt(np.mean(np.sum(divergence ** 2, axis=0))))}


def fcc_slip_systems(rotation=None):
    """Twelve unique unoriented {111}<110> systems; signed slip carries direction."""
    normals = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype=float)
    candidates = sorted(set(permutations((1, 1, 0))) | set(permutations((1, -1, 0))))
    directions, planes = [], []
    for normal in normals:
        for candidate in candidates:
            direction = np.array(candidate, dtype=float)
            if direction[np.flatnonzero(direction)[0]] < 0 or normal @ direction != 0:
                continue
            directions.append(direction / np.sqrt(2))
            planes.append(normal / np.sqrt(3))
    directions, planes = np.array(directions), np.array(planes)
    if rotation is not None:
        rotation = np.asarray(rotation)
        if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3)) or not np.isclose(np.linalg.det(rotation), 1):
            raise ValueError("Crystal orientation must be a proper rotation.")
        directions, planes = directions @ rotation.T, planes @ rotation.T
    schmid = np.einsum("ai,aj->aij", directions, planes)
    return directions, planes, (schmid + schmid.swapaxes(1, 2)) / 2


@dataclass(frozen=True)
class Parameters3D:
    W: float = 1.
    gradient: tuple = (1., 1., 1.4)
    susceptibility: float = 1.
    binding: float = .15
    reference_population: float = .2
    phase_misfit: float = .08
    solute_expansion: float = .02
    hardening: float = .2
    yield_stress: float = .015
    slip_drag: float = 1.
    phase_mobility: float = 1.
    diffusion_mobility: float = 1.


class CoupledCrystal:
    def __init__(self, grid, parameters=None, mean_strain=None, schmid=None, lam=2., shear=1.):
        self.grid, self.p = grid, parameters or Parameters3D()
        self.elastic = PeriodicElasticity(grid, lam, shear)
        self.mean_strain = np.zeros((3, 3)) if mean_strain is None else np.asarray(mean_strain)
        self.schmid = fcc_slip_systems()[2] if schmid is None else np.asarray(schmid)
        if self.schmid.ndim != 3 or self.schmid.shape[1:] != (3, 3):
            raise ValueError("Slip tensors must have shape (systems, 3, 3).")
        if not np.allclose(self.schmid, self.schmid.swapaxes(1, 2)) or not np.allclose(np.trace(self.schmid, axis1=1, axis2=2), 0):
            raise ValueError("Slip tensors must be symmetric and isochoric.")
        p = self.p
        if min(p.W, *p.gradient, p.susceptibility, p.slip_drag, p.phase_mobility, p.diffusion_mobility) <= 0 or min(p.hardening, p.yield_stress) < 0:
            raise ValueError("Energy and kinetic coefficients violate positivity assumptions.")

    def evaluate(self, v, c, slip, extra_eigenstrain=None):
        p, g = self.p, self.grid
        if v.shape != g.shape or c.shape != g.shape or slip.shape != (len(self.schmid),) + g.shape:
            raise ValueError("Field shapes do not match the grid and slip systems.")
        dc = c - p.reference_population
        eigen = np.einsum("a...,aij->ij...", slip, self.schmid)
        if extra_eigenstrain is not None:
            if extra_eigenstrain.shape != eigen.shape:
                raise ValueError("Additional eigenstrain must match the full tensor grid.")
            eigen += extra_eigenstrain
        eigen += np.eye(3)[..., None, None, None] * (p.phase_misfit * (v + 1) / 2 + p.solute_expansion * dc)
        elastic = self.elastic.solve(eigen, self.mean_strain)
        trace = np.einsum("ii...->...", elastic["stress"])
        resolved = np.einsum("ij...,aij->a...", elastic["stress"], self.schmid)
        gv = p.W * (v ** 3 - v) - g.lap(v, p.gradient) + 2 * p.binding * dc * v - p.phase_misfit * trace / 2
        chemical = p.susceptibility * dc - p.binding * (1 - v * v) - p.solute_expansion * trace
        q = resolved - p.hardening * slip
        slip_rate = np.sign(q) * np.maximum(np.abs(q) - p.yield_stress, 0) / p.slip_drag
        rates = (-p.phase_mobility * gv, p.diffusion_mobility * g.lap(chemical), slip_rate)
        components = {"chemical": g.integrate(p.susceptibility * dc ** 2 / 2),
                      "structural": g.integrate(p.W * (1 - v * v) ** 2 / 4),
                      "interface": g.gradient_energy(v, p.gradient),
                      "binding": g.integrate(-p.binding * dc * (1 - v * v)),
                      "elastic": elastic["energy"], "hardening": g.integrate(p.hardening * slip ** 2 / 2)}
        dissipation = (p.phase_mobility * g.integrate(gv ** 2)
                       + 2 * p.diffusion_mobility * g.gradient_energy(chemical)
                       + g.integrate(p.yield_stress * np.abs(slip_rate) + p.slip_drag * slip_rate ** 2))
        return {"energy": sum(components.values()), "components": components,
                "forces": (gv, chemical, -q), "rates": rates, "dissipation": dissipation,
                "stress": elastic["stress"], "strain": elastic["strain"],
                "equilibrium_rms": elastic["equilibrium_rms"]}

    def step(self, fields, dt, current=None):
        if dt <= 0:
            raise ValueError("Timestep must be positive.")
        current = self.evaluate(*fields) if current is None else current
        used, trials = dt, 0
        for trials in range(1, 31):
            candidate = tuple(f + used * rate for f, rate in zip(fields, current["rates"]))
            if np.isfinite(candidate[1]).all() and candidate[1].min() > 0 and candidate[1].max() < 1:
                following = self.evaluate(*candidate)
                bound = current["energy"] - .1 * used * current["dissipation"] + 1e-12 * max(1, abs(current["energy"]))
                if np.isfinite(following["energy"]) and following["energy"] <= bound:
                    return candidate, following, used, trials
            used *= .5
        raise RuntimeError("The coupled 3D step failed positivity or energy descent.")
