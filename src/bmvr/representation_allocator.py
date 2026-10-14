"""Cross-representation allocation for gliding edge-dislocation ensembles (theory Section 91).

Discrete particles (particle-in-cell) where blocks are sparse, transported signed densities where they are dense, one
shared filtered Fourier stress solve, and Burgers-exact conversions between the two."""
import numpy as np
from .extended_defects import wavevectors


class GlideEnsemble:
    def __init__(self, G, L=1., mu=1., nu=.3, b=1e-6, M=1., tau=0., width=1.5, block=16, n_c=64):
        self.G, self.L, self.h, self.b, self.M, self.tau, self.B, self.n_c = G, L, L/G, b, M, tau, block, n_c
        E = 2*mu/(1 - nu); KX, KY = wavevectors(G, L); k2 = KX**2 + KY**2; k4 = k2**2; k4[0, 0] = np.inf
        # sigma_xy = -dx dy chi, chi_hat = E' eta_hat / k^4, eta = -dy alpha  =>  sigma_hat = KX KY E' (-i KY alpha_hat) / k^4
        self.kernel = KX*KY*E*(-1j*KY)/k4*np.exp(-k2*(width*self.h)**2/2)

    # grid transfer (cell-centred cloud-in-cell)
    def _cic(self, x, y):
        fx, fy = x/self.h - .5, y/self.h - .5; i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int); tx, ty = fx - i0, fy - j0; G = self.G
        idx = [((i0 + a) % G)*G + (j0 + c) % G for a in (0, 1) for c in (0, 1)]; w = [(1 - tx if a == 0 else tx)*(1 - ty if c == 0 else ty) for a in (0, 1) for c in (0, 1)]
        return idx, w

    def deposit(self, x, y, q):
        idx, w = self._cic(x, y); G = self.G
        return sum(np.bincount(i, weights=q*wt, minlength=G*G) for i, wt in zip(idx, w)).reshape(G, G)/self.h**2

    def gather(self, field, x, y):
        idx, w = self._cic(x, y); f = field.ravel(); return sum(f[i]*wt for i, wt in zip(idx, w))

    def stress(self, alpha):
        return np.real(np.fft.ifftn(np.fft.fftn(alpha)*self.kernel))

    def transport(self, rho, v_face, dt):
        """MUSCL-minmod upwind flux along x (axis 0), periodic; returns the new density and the face fluxes."""
        d = np.roll(rho, -1, 0) - rho; slope = np.where(d*np.roll(d, 1, 0) > 0, np.sign(d)*np.minimum(abs(d), abs(np.roll(d, 1, 0))), 0.)
        left = rho + .5*slope; right = np.roll(rho - .5*slope, -1, 0); F = v_face*np.where(v_face > 0, left, right)
        return rho - dt/self.h*(F - np.roll(F, 1, 0)), F

    def blocks(self, x, y, rho):
        """Dislocations per block: particles plus integrated densities."""
        nb = self.G//self.B; bi = (x/self.h).astype(int)//self.B % nb; bj = (y/self.h).astype(int)//self.B % nb
        count = np.bincount(bi*nb + bj, minlength=nb*nb).reshape(nb, nb).astype(float)
        dens = sum(rho).reshape(nb, self.B, nb, self.B).sum(axis=(1, 3))*self.h**2
        return count + dens

    def cell_mask(self, block_mask):
        return np.kron(block_mask, np.ones((self.B, self.B), bool))


class Run:
    """One method: 'D' (all discrete), 'C' (all continuum) or 'H' (allocated)."""

    def __init__(self, ens, x, y, s, ids, method, realloc=20):
        self.e, self.method, self.realloc = ens, method, realloc; G = ens.G
        self.x, self.y, self.s, self.ids = x.copy(), y.copy(), s.copy(), ids.copy()
        self.rho = [np.zeros((G, G)), np.zeros((G, G))]  # +, -
        self.gamma = np.zeros((G, G)); nb = G//ens.B; self.cont = np.zeros((nb, nb), bool); self.ledger = []; self.steps = 0
        if method == 'C': self._absorb(np.ones(len(self.x), bool))
        if method == 'H': self.allocate()

    def totals(self):
        h2 = self.e.h**2; return len(self.x) + sum(r.sum() for r in self.rho)*h2, self.s.sum() + (self.rho[0].sum() - self.rho[1].sum())*h2

    def _absorb(self, mask):
        for k, sg in enumerate((1, -1)):
            m = mask & (self.s == sg)
            if m.any(): self.rho[k] += self.e.deposit(self.x[m], self.y[m], np.ones(m.sum()))
        keep = ~mask; self.x, self.y, self.s, self.ids = self.x[keep], self.y[keep], self.s[keep], self.ids[keep]

    def _emit(self, cells):
        """Replace floor(rho h^2) of each sign in the given cells by particles at stratified positions."""
        h = self.e.h; xs, ys, ss = [], [], []
        for k, sg in enumerate((1, -1)):
            m = np.floor(self.rho[k]*h*h + 1e-12).astype(int)*cells; I, J = np.nonzero(m)
            for i, j in zip(I, J):
                c = m[i, j]; r = int(np.ceil(np.sqrt(c))); a = (np.arange(c) % r + .5)/r; bb = (np.arange(c)//r + .5)/r
                xs.append((i + a)*h); ys.append((j + bb)*h); ss.append(np.full(c, sg))
            self.rho[k] -= m/h**2
        if xs:
            n0 = len(self.x); self.x = np.r_[self.x, np.concatenate(xs)]; self.y = np.r_[self.y, np.concatenate(ys)]
            self.s = np.r_[self.s, np.concatenate(ss)]; self.ids = np.r_[self.ids, -np.ones(len(self.x) - n0, int)]

    def allocate(self):
        before = self.totals(); n = self.e.blocks(self.x, self.y, self.rho)
        new = np.where(self.cont, n >= self.e.n_c/2, n >= self.e.n_c); cells_new = self.e.cell_mask(new)
        ci = (self.x/self.e.h).astype(int) % self.e.G; cj = (self.y/self.e.h).astype(int) % self.e.G
        self._absorb(cells_new[ci, cj]); self._emit(~cells_new); self.cont = new; after = self.totals()
        self.ledger.append(dict(step=self.steps, count_error=abs(after[0] - before[0]), burgers_error=abs(after[1] - before[1]), continuum_blocks=int(new.sum()), particles=len(self.x)))

    def step(self, dt):
        e = self.e; alpha = e.b*(e.deposit(self.x, self.y, self.s.astype(float)) + self.rho[0] - self.rho[1]); sig = e.stress(alpha)
        if len(self.x):
            v = e.M*self.s*e.b*(e.gather(sig, self.x, self.y) + e.tau); xm = self.x + .5*dt*v
            self.gamma += e.deposit(xm % e.L, self.y, self.s*e.b*v*dt); self.x = (self.x + dt*v) % e.L
        vf = e.M*e.b*(.5*(sig + np.roll(sig, -1, 0)) + e.tau)
        for k, sg in enumerate((1, -1)):
            if self.rho[k].any():
                self.rho[k], F = e.transport(self.rho[k], sg*vf, dt); self.gamma += sg*e.b*.5*(F + np.roll(F, 1, 0))*dt
        self.steps += 1
        if self.method == 'H':
            cells = e.cell_mask(self.cont); ci = (self.x/e.h).astype(int) % e.G; cj = (self.y/e.h).astype(int) % e.G
            inside = cells[ci, cj]
            if inside.any(): b0 = self.totals(); self._absorb(inside); b1 = self.totals(); self.ledger.append(dict(step=self.steps, count_error=abs(b1[0] - b0[0]), burgers_error=abs(b1[1] - b0[1]), absorbed=int(inside.sum())))
            if self.steps % self.realloc == 0: self.allocate()
