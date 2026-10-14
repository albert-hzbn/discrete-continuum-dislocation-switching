"""Wedge-disclination dynamics in plane strain (theory Section 88): biharmonic energy on a periodic grid, exact forces, overdamped motion."""
import numpy as np
from .extended_defects import wavevectors


class Disclinations:
    def __init__(self, n, length, mu=1., nu=.3, width=1.):
        self.n, self.L, self.mu, self.nu, self.w = n, length, mu, nu, width; self.E = 2*mu/(1 - nu); self.h = length/n
        KX, KY = wavevectors(n, length); k2 = KX**2 + KY**2; k2[0, 0] = np.inf; self.k4 = k2**2; self.KX, self.KY = KX, KY
        x = (np.arange(n) + .5)*self.h; self.X, self.Y = np.meshgrid(x, x, indexing='ij')

    def _g(self, p):
        """Periodic Gaussian (nearest image) of unit integral and its derivative with respect to its centre."""
        dx = self.X - p[0]; dy = self.Y - p[1]; dx -= np.round(dx/self.L)*self.L; dy -= np.round(dy/self.L)*self.L
        g = np.exp(-(dx*dx + dy*dy)/(2*self.w**2))/(2*np.pi*self.w**2); return g, g*dx/self.w**2, g*dy/self.w**2

    def chi(self, pos, omega):
        eta = sum(o*self._g(p)[0] for p, o in zip(pos, omega)); eh = np.fft.fftn(eta)*self.h**2
        return self.E*eh/self.k4, eh

    def energy(self, pos, omega, ext=None):
        ch, eh = self.chi(pos, omega); W = .5*self.E*np.sum(abs(eh)**2/self.k4)/self.L**2
        return float(W + (sum(o*ext(p)[0] for p, o in zip(pos, omega)) if ext else 0.))

    def forces(self, pos, omega, ext=None):
        """F_i = -omega_i int g_i grad chi = omega_i int chi grad_{x_i} g_i (exact derivative of the discrete energy)."""
        ch, _ = self.chi(pos, omega); c = np.real(np.fft.ifftn(ch))/self.h**2; F = np.zeros((len(pos), 2))
        for i, (p, o) in enumerate(zip(pos, omega)):
            _, gx, gy = self._g(p); F[i] = -o*self.h**2*np.array([np.sum(c*gx), np.sum(c*gy)])
            if ext: F[i] -= o*np.asarray(ext(p)[1])
        return F


def applied_shear(tau):
    """chi_ext = -tau x y gives sigma_xy = tau; returns (chi, grad chi) at a point."""
    return lambda p: (-tau*p[0]*p[1], (-tau*p[1], -tau*p[0]))


def edge_force(mu, nu, b1, b2, x, y):
    """Glide force on edge dislocation 2 from edge dislocation 1 (both b along x) at relative position (x, y)."""
    return mu*b1*b2/(2*np.pi*(1 - nu))*x*(x*x - y*y)/(x*x + y*y)**2
