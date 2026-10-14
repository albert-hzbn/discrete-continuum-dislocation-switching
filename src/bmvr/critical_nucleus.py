"""Finite-width critical nucleus of a non-conserved phase field (theory Section 84).

F[phi] = sum h^2 [kappa/2 |grad_h phi|^2 + W phi^2 (1 - phi)^2 - dg p(phi)], p = phi^3 (10 - 15 phi + 6 phi^2), on a periodic
square with the 5-point Laplacian. The critical nucleus is the stationary point found by Newton's method from the classical
droplet; its index is certified from the sparse Hessian spectrum."""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import eigsh, minres


def p(phi): return phi**3*(10 - 15*phi + 6*phi**2)
def dp(phi): return 30*phi**2*(1 - phi)**2
def d2p(phi): return 60*phi*(1 - phi)*(1 - 2*phi)


class NucleationEnergy:
    def __init__(self, n, h=.5, kappa=1., W=1., dg=.01):
        self.n, self.h, self.kappa, self.W, self.dg = n, h, kappa, W, dg; self.L = n*h

    def bulk(self, phi): return self.W*phi**2*(1 - phi)**2 - self.dg*p(phi)
    def dbulk(self, phi): return 2*self.W*phi*(1 - phi)*(1 - 2*phi) - self.dg*dp(phi)
    def d2bulk(self, phi): return 2*self.W*(1 - 6*phi + 6*phi**2) - self.dg*d2p(phi)

    def energy(self, phi):
        dx = np.roll(phi, -1, 0) - phi; dy = np.roll(phi, -1, 1) - phi
        return float(self.kappa/2*np.sum(dx*dx + dy*dy) + self.h**2*np.sum(self.bulk(phi)))

    def gradient(self, phi):
        """dF/dphi at each node; the functional derivative is this divided by h^2."""
        nb = np.roll(phi, 1, 0) + np.roll(phi, -1, 0) + np.roll(phi, 1, 1) + np.roll(phi, -1, 1)
        return -self.kappa*(nb - 4*phi) + self.h**2*self.dbulk(phi)

    def hessian(self, phi):
        """Sparse Hessian of F divided by h^2, i.e. the operator -kappa Lap_h + f''(phi)."""
        n = self.n; one = np.ones(n); D = diags([one[:-1], -2*one, one[:-1]], [-1, 0, 1], format='lil'); D[0, n - 1] = D[n - 1, 0] = 1
        I = identity(n, format='csr'); lap = (kron(D.tocsr(), I) + kron(I, D.tocsr()))/self.h**2
        return (-self.kappa*lap + diags(self.d2bulk(phi).ravel())).tocsr()

    def droplet(self, R, width=1.):
        c = (np.arange(self.n) + .5)*self.h - self.L/2; X, Y = np.meshgrid(c, c, indexing='ij')
        return .5*(1 - np.tanh((np.hypot(X, Y) - R)/(2*width*np.sqrt(self.kappa/(2*self.W)))))

    def newton(self, phi, tol=1e-11, steps=40):
        """Newton's method on the full gradient; converges to the nearby stationary point whatever its index."""
        for k in range(steps):
            g = self.gradient(phi)/self.h**2
            if np.max(abs(g)) <= tol: return phi, k
            d, _ = minres(self.hessian(phi), -g.ravel(), rtol=1e-12, maxiter=20000); phi = phi + d.reshape(phi.shape)
        return phi, steps

    def critical_nucleus(self, R_guess):
        """Newton from the classical droplet of radius R_guess. (A fixed-volume constrained search was declared first and
        collapses to the homogeneous state in a large periodic box, the canonical evaporation branch; Section 84.1.)"""
        return self.newton(self.droplet(R_guess))

    def flow(self, phi, dt, t_end, stop):
        """Explicit Allen-Cahn flow phi_t = -dF/dphi / h^2; returns the final state, time, energies and whether F decreased."""
        E = [self.energy(phi)]; t = 0.; decreasing = True
        while t < t_end:
            phi = phi - dt*self.gradient(phi)/self.h**2; t += dt; E.append(self.energy(phi)); decreasing &= E[-1] <= E[-2] + 1e-12*abs(E[-2])
            if stop(phi): break
        return phi, t, E, bool(decreasing)

    def spectrum(self, phi, R):
        """The two smallest eigenvalues of -kappa Lap_h + f'' (all eigenvalues exceed min f'' >= -1, so a shift at -1.1
        isolates them), and the three nearest -2/R^2: the unstable mode and the two translation modes."""
        H = self.hessian(phi); low = np.sort(eigsh(H, k=2, sigma=-1.1, which='LM', return_eigenvectors=False))
        vals, vecs = eigsh(H, k=3, sigma=-2/R**2, which='LM'); order = np.argsort(vals)
        return low, vals[order], vecs[:, order]

def planar_interface_energy(h=.5, kappa=1., W=1., n=400):
    """Discrete planar interface energy at dg = 0: minimize a periodic 1D chain at mean 1/2 (two interfaces)."""
    x = (np.arange(n) + .5)*h; L = n*h; w = np.sqrt(kappa/(2*W))
    phi0 = .5*(np.tanh((x - L/4)/(2*w)) - np.tanh((x - 3*L/4)/(2*w)))
    def fun(y):
        phi = .5 + y - y.mean(); d = np.roll(phi, -1) - phi
        E = kappa/(2*h)*np.sum(d*d) + h*np.sum(W*phi**2*(1 - phi)**2)
        g = -kappa/h*(np.roll(phi, 1) + np.roll(phi, -1) - 2*phi) + h*2*W*phi*(1 - phi)*(1 - 2*phi); return E, g - g.mean()
    r = minimize(fun, phi0 - .5, jac=True, method='L-BFGS-B', options=dict(maxiter=100000, gtol=1e-13, ftol=1e-16))
    return float(r.fun/2)
