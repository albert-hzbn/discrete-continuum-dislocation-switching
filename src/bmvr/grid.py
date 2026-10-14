"""Volume-weighted finite volumes on uniform or nonuniform tensor grids."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Grid:
    xe: np.ndarray
    ye: np.ndarray
    periodic: bool = False

    def __post_init__(self):
        self.xe = np.asarray(self.xe, dtype=float)
        self.ye = np.asarray(self.ye, dtype=float)
        self.dx, self.dy = np.diff(self.xe), np.diff(self.ye)
        if np.any(self.dx <= 0) or np.any(self.dy <= 0):
            raise ValueError("Cell edges must increase strictly.")
        self.x = (self.xe[1:] + self.xe[:-1]) / 2
        self.y = (self.ye[1:] + self.ye[:-1]) / 2
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.vol = self.dy[:, None] * self.dx[None, :]
        self.shape = self.vol.shape
        self.size = self.vol.size
        self.area = float(self.vol.sum())
        self.hmin = float(min(self.dx.min(), self.dy.min()))

    @classmethod
    def uniform(cls, length=64.0, n=128, periodic=False):
        edges = np.linspace(-length / 2, length / 2, n + 1)
        return cls(edges, edges.copy(), periodic)

    def integrate(self, value):
        return float(np.sum(self.vol * value))

    def lap(self, v, k=(1.0, 1.0)):
        """No-flux boundaries unless periodic; one shared flux on each face."""
        out = np.zeros_like(v, dtype=float)
        fx = k[0] * np.diff(v, axis=-1) / np.diff(self.x)
        out[..., :, :-1] += fx / self.dx[:-1]
        out[..., :, 1:] -= fx / self.dx[1:]
        fy = k[1] * np.diff(v, axis=-2) / np.diff(self.y)[:, None]
        out[..., :-1, :] += fy / self.dy[:-1, None]
        out[..., 1:, :] -= fy / self.dy[1:, None]
        if self.periodic:
            fx = k[0] * (v[..., :, 0] - v[..., :, -1]) / ((self.dx[0] + self.dx[-1]) / 2)
            out[..., :, -1] += fx / self.dx[-1]
            out[..., :, 0] -= fx / self.dx[0]
            fy = k[1] * (v[..., 0, :] - v[..., -1, :]) / ((self.dy[0] + self.dy[-1]) / 2)
            out[..., -1, :] += fy / self.dy[-1]
            out[..., 0, :] -= fy / self.dy[0]
        return out

    def gradient_energy(self, v, k=(1.0, 1.0)):
        e = k[0] * np.sum(np.diff(v, axis=-1) ** 2 * self.dy[:, None] / np.diff(self.x))
        e += k[1] * np.sum(np.diff(v, axis=-2) ** 2 * self.dx[None, :] / np.diff(self.y)[:, None])
        if self.periodic:
            e += k[0] * np.sum((v[..., :, 0] - v[..., :, -1]) ** 2 * self.dy / ((self.dx[0] + self.dx[-1]) / 2))
            e += k[1] * np.sum((v[..., 0, :] - v[..., -1, :]) ** 2 * self.dx / ((self.dy[0] + self.dy[-1]) / 2))
        return float(e / 2)

    def spectrum(self):
        if not self.periodic or not np.allclose(self.dx, self.dx[0]) or not np.allclose(self.dy, self.dy[0]):
            raise ValueError("The Fourier solver requires a uniform periodic grid.")
        kx = 2 * np.pi * np.fft.fftfreq(len(self.x))
        ky = 2 * np.pi * np.fft.fftfreq(len(self.y))
        return 4 * np.sin(kx / 2)[None, :] ** 2 / self.dx[0] ** 2 + 4 * np.sin(ky / 2)[:, None] ** 2 / self.dy[0] ** 2


def circle(grid, radius, width=np.sqrt(2), center=(0.0, 0.0)):
    r = np.hypot(grid.X - center[0], grid.Y - center[1])
    return np.tanh((radius - r) / width)
