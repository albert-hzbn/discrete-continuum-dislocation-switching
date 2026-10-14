"""Section 59: case 9 with a local continuum slip field and periodic antiplane elasticity on tensor grids."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .energy import DoubleWell
from .defects import binding, screw_stress


class AntiplaneElasticity:
    """E_el(gamma, p) = min over periodic w of sum_faces (mu/2) e_f^2 V_f on a periodic tensor grid.

    x-faces carry e = dw/dx; y-faces carry e = ebar + dw/dy - gamma_f - p with gamma_f the mean of the two cells.
    Face weights V_f are dual-cell areas (centre spacing times cell width), so each family sums to the box area.
    """

    def __init__(self, grid, shear=1.0, weighted=True):
        self.weighted = weighted
        if not grid.periodic:
            raise ValueError("Section 59 elasticity requires a periodic grid.")
        self.grid, self.mu = grid, float(shear)
        ny, nx = grid.shape
        lx, ly = grid.xe[-1] - grid.xe[0], grid.ye[-1] - grid.ye[0]
        dxc = np.diff(np.append(grid.x, grid.x[0] + lx))
        dyc = np.diff(np.append(grid.y, grid.y[0] + ly))
        index = np.arange(nx * ny).reshape(ny, nx)
        right, up = np.roll(index, -1, axis=1), np.roll(index, -1, axis=0)
        rows = np.arange(nx * ny)
        self.Dx = sparse.csr_matrix((np.concatenate([-1 / np.tile(dxc, ny), 1 / np.tile(dxc, ny)]),
                                     (np.concatenate([rows, rows]), np.concatenate([index.ravel(), right.ravel()]))), shape=(nx * ny, nx * ny))
        self.Dy = sparse.csr_matrix((np.concatenate([-1 / np.repeat(dyc, nx), 1 / np.repeat(dyc, nx)]),
                                     (np.concatenate([rows, rows]), np.concatenate([index.ravel(), up.ravel()]))), shape=(nx * ny, nx * ny))
        # gamma on a y-face is the half-cell-volume average, the exact quadrature of cellwise-constant gamma over the
        # face's dual cell; each cell then receives exactly its own volume of face stress. Equal spacing gives 1/2 each.
        # The declared run used 1/2 on every face and a post hoc attempt used linear interpolation (weights swapped);
        # both give a spurious stress factor where the spacing jumps (Section 59 outcome).
        dy_lo, dy_hi = grid.dy, np.roll(grid.dy, -1)
        w_lo, w_hi = np.repeat(dy_lo / (dy_lo + dy_hi), nx), np.repeat(dy_hi / (dy_lo + dy_hi), nx)
        if not getattr(self, 'weighted', True):
            w_lo, w_hi = np.full(nx * ny, .5), np.full(nx * ny, .5)
        self.Ay = sparse.csr_matrix((np.concatenate([w_lo, w_hi]), (np.concatenate([rows, rows]), np.concatenate([index.ravel(), up.ravel()]))), shape=(nx * ny, nx * ny))
        self.Vx = (dxc[None, :] * grid.dy[:, None]).ravel()
        self.Vy = (dyc[:, None] * grid.dx[None, :]).ravel()
        K = (self.Dx.T @ sparse.diags(self.Vx) @ self.Dx + self.Dy.T @ sparse.diags(self.Vy) @ self.Dy).tocsc()
        self.solver = splu(K[1:, 1:])  # w is defined up to a constant; pin the first cell.
        self.size = nx * ny

    def solve(self, gamma, ebar, plastic):
        s = ebar - plastic - self.Ay @ gamma.ravel()
        w = np.zeros(self.size)
        w[1:] = self.solver.solve(-(self.Dy.T @ (self.Vy * s))[1:])
        ex, ey = self.Dx @ w, self.Dy @ w + s
        energy = .5 * self.mu * (np.sum(self.Vx * ex * ex) + np.sum(self.Vy * ey * ey))
        face_stress = self.mu * self.Vy * ey
        d_gamma = -(self.Ay.T @ face_stress).reshape(self.grid.shape)  # envelope theorem
        d_plastic = -float(face_stress.sum())
        return dict(energy=float(energy), d_gamma=d_gamma, d_plastic=d_plastic, mean_stress=float(face_stress.sum() / self.grid.area),
                    stress_yz=(self.Ay.T @ (self.mu * ey)).reshape(self.grid.shape))


def screw_fields(grid, screws):
    """Regularized sigma_yz of all screws and its x-derivatives per screw (for the exact interaction force)."""
    total = np.zeros(grid.shape)
    derivatives = []
    for (x0, y0), b in zip(screws.positions, screws.burgers):
        x, y = grid.X - x0, grid.Y - y0
        d = x * x + y * y + screws.core ** 2
        total += screws.shear * b * x / (2 * np.pi * d)
        derivatives.append(screws.shear * b / (2 * np.pi) * (d - 2 * x * x) / d ** 2)
    return total, derivatives


def energy_forces(grid, elastic, v, c, gamma, screws, origins, imposed=.15, chemical=.3, interface=.2, c0=.2,
                  hardening=.5, gradient=1.0):
    base = DoubleWell()
    epair, force = screws.pair()
    eb, bv, bc, bf = binding(grid, v, c, screws, chemical, interface)
    area = grid.area
    plastic_s = float(np.sum(screws.burgers * (screws.positions[:, 0] - origins[:, 0])) / area) if len(screws.burgers) else 0.0
    el = elastic.solve(gamma, imposed, plastic_s)
    sigma, dsigma = screw_fields(grid, screws)
    mean_gamma = grid.integrate(gamma) / area
    tilde = gamma - mean_gamma
    e_int = -grid.integrate(sigma * tilde)
    e_hard = grid.integrate(hardening * gamma * gamma / 2) + grid.gradient_energy(gamma, (gradient, gradient))
    energy = base.value(grid, v) + grid.integrate((c - c0) ** 2 / 2) + epair + eb + el["energy"] + e_int + e_hard
    # Per-volume derivative of F with respect to gamma.
    sigma_mean = grid.integrate(sigma) / area
    g_gamma = el["d_gamma"] / grid.vol - (sigma - sigma_mean) + hardening * gamma - grid.lap(gamma, (gradient, gradient))
    force = force + bf
    if len(screws.burgers):
        force[:, 0] += screws.burgers * el["mean_stress"]
        for k, ds in enumerate(dsigma):
            force[k, 0] -= grid.integrate(tilde * ds)
    force[:, 1] = 0.0  # glide constraint
    return dict(energy=float(energy), gv=base.force(grid, v) + bv, mu=c - c0 + bc, g_gamma=g_gamma, force=force,
                plastic_screws=plastic_s, mean_gamma=float(mean_gamma), mean_stress=el["mean_stress"],
                elastic_energy=el["energy"], interaction_energy=float(e_int), hardening_energy=float(e_hard))
