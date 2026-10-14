"""Sparse screw-line section model and reciprocal diffuse-core binding."""

from dataclasses import dataclass, replace
import numpy as np
from .energy import DoubleWell


@dataclass
class Screws:
    positions: np.ndarray
    burgers: np.ndarray
    core: float = 0.75
    shear: float = 1.0
    reference_radius: float = 64.0
    mobility: float = 1.0

    def moved(self, positions):
        return replace(self, positions=np.array(positions, dtype=float))

    def pair(self):
        count = len(self.burgers)
        forces = np.zeros((count, 2))
        # Fixed additive core convention, used explicitly by annihilation events.
        energy = self.shear * np.sum(self.burgers ** 2) * np.log(self.reference_radius / self.core) / (4 * np.pi)
        for i in range(count):
            for j in range(i + 1, count):
                d = self.positions[i] - self.positions[j]
                rr = float(d @ d + self.core ** 2)
                pref = self.shear * self.burgers[i] * self.burgers[j] / (4 * np.pi)
                energy -= pref * np.log(rr / self.reference_radius ** 2)
                f = 2 * pref * d / rr
                forces[i] += f
                forces[j] -= f
        return float(energy), forces


def screw_stress(x, y, b=1.0, shear=1.0, core=0.75):
    denom = x * x + y * y + core * core
    return -shear * b * y / (2 * np.pi * denom), shear * b * x / (2 * np.pi * denom)


def core_kernel(grid, position, width):
    dx, dy = grid.X - position[0], grid.Y - position[1]
    raw = np.exp(-(dx * dx + dy * dy) / (2 * width * width))
    norm = grid.integrate(raw)
    if norm < 1e-100:
        raise ValueError("A defect is outside its resolved domain.")
    rho = raw / norm
    # Differentiate the finite-domain quadrature normalization.
    gx, gy = dx / width ** 2, dy / width ** 2
    drx = rho * (gx - grid.integrate(rho * gx))
    dry = rho * (gy - grid.integrate(rho * gy))
    return rho, drx, dry


def binding(grid, v, c, screws, chemical=0.1, interface=0.1):
    potential = -chemical * c - interface * (1 - v * v)
    energy = 0.0
    gv, gc = np.zeros_like(v), np.zeros_like(c)
    force = np.zeros_like(screws.positions)
    for i, position in enumerate(screws.positions):
        rho, drx, dry = core_kernel(grid, position, screws.core)
        energy += grid.integrate(rho * potential)
        gv += 2 * interface * v * rho
        gc -= chemical * rho
        force[i] = -grid.integrate(drx * potential), -grid.integrate(dry * potential)
    return energy, gv, gc, force


def coupled_energy_forces(grid, v, c, screws, origins, imposed=0.025,
                          chemical=0.1, interface=0.1, c0=0.2):
    base = DoubleWell()
    epair, force = screws.pair()
    eb, bv, bc, bf = binding(grid, v, c, screws, chemical, interface)
    plastic = float(np.sum(screws.burgers * (screws.positions[:, 0] - origins[:, 0])) / grid.area)
    stress = screws.shear * (imposed - plastic)
    load_energy = grid.area * screws.shear * (imposed - plastic) ** 2 / 2
    energy = base.value(grid, v) + grid.integrate((c - c0) ** 2 / 2) + epair + eb + load_energy
    force += bf
    force[:, 0] += screws.burgers * stress
    force[:, 1] = 0  # Specified glide constraint.
    return {"energy": float(energy), "gv": base.force(grid, v) + bv,
            "mu": c - c0 + bc, "force": force, "plastic_shear": plastic,
            "mean_stress": stress, "binding_energy": float(eb), "pair_energy": epair}
