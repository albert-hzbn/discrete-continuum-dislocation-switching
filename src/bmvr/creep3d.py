"""Section 98: 3D creep cavitation on a grain-boundary plane coupled to hybrid AT2 fracture, on the GPU.

Periodic cube of side 1 (L = 10 um for copper), GB plane z = 1/2. Stresses in units of the shear modulus, time in
t_u = L^3 kT/(delta D_b Omega mu). Mechanics: trilinear hexahedra with one-point integration (rotated finite
differences) on voxel corners, conjugate gradients preconditioned by the exact FFT inverse of the homogeneous operator.
Damage: hybrid AT2 (isotropic degradation in the momentum balance, Amor positive energy with a history field).
GB plane: vacancy excess c on n^2 cells, semi-implicit FFT step for diffusion and the distributed source, explicit
absorption by discrete spherical cavities. Uses CuPy when available, NumPy otherwise.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

try:
    import cupy as xp
    GPU = True
except ImportError:                     # pragma: no cover
    xp, GPU = np, False


def to_host(a):
    return a.get() if GPU and isinstance(a, xp.ndarray) else np.asarray(a)


@dataclass(frozen=True)
class Copper773:
    """Copper at 773 K (Frost and Ashby 1982, Table 4.1; nu from Ledbetter 1974)."""
    T: float = 773.
    L: float = 10e-6
    omega: float = 1.18e-29
    mu0: float = 4.21e10
    dmu: float = -0.54
    Tm: float = 1356.
    dDb0: float = 5.0e-15
    Qb: float = 104e3
    nu: float = 0.324
    gamma_s: float = 1.7

    @property
    def mu(self):
        return self.mu0*(1 + (self.T - 300)/self.Tm*self.dmu)

    @property
    def dDb(self):
        return self.dDb0*math.exp(-self.Qb/(8.314462618*self.T))

    @property
    def t_unit(self):
        return self.L**3*1.380649e-23*self.T/(self.dDb*self.omega*self.mu)

    def stress(self, MPa):
        return MPa*1e6/self.mu


@dataclass(frozen=True)
class Params3D:
    n: int = 96
    ell: float = 0.025            # 0.25 um
    residual: float = 1e-3
    chi: float = 1.
    l_s: float = 0.1              # source collection length, 1 um
    k_ex: float = 3.5            # calibrated to Raj-Ashby on one case (Section 98 v2); v1 used 10
    w_ring: float = 0.015         # 0.15 um
    w_render: float = 0.015
    count: int = 12
    min_sep: float = 0.15
    r_range: tuple = (0.03, 0.05)
    seed: int = 98
    dt: float = 5e-4
    cg_tol: float = 1e-6


def nuclei(p):
    rng = np.random.default_rng(p.seed)
    centres = []
    while len(centres) < p.count:
        c = rng.random(2)
        if all(np.linalg.norm((c - o + .5) % 1 - .5) >= p.min_sep for o in centres):
            centres.append(c)
    return np.array(centres), rng.uniform(*p.r_range, p.count)


class Mechanics:
    """Periodic linear elasticity with stiffness g(x) C at voxel centres, displacements at voxel corners."""

    def __init__(self, n, lam, mu):
        self.n, self.lam, self.mu, self.h = n, lam, mu, 1/n
        k = 2*np.pi*np.fft.fftfreq(n); kz = 2*np.pi*np.fft.rfftfreq(n)
        KX, KY, KZ = np.meshgrid(k, k, kz, indexing='ij')
        ex, ey, ez = (np.exp(1j*K) for K in (KX, KY, KZ))
        avg = lambda e: (1 + e)/2
        D = np.stack([(ex - 1)*avg(ey)*avg(ez), (ey - 1)*avg(ex)*avg(ez), (ez - 1)*avg(ex)*avg(ey)])*n
        A = (lam*np.einsum('a...,b...->...ab', D.conj(), D) + mu*np.einsum('a...,b...->...ab', D, D.conj())
             + mu*np.einsum('a...,a...->...', D, D.conj())[..., None, None]*np.eye(3))
        norm = np.einsum('a...,a...->...', D, D.conj()).real
        null = norm < 1e-10*norm.max()
        A[null] = np.eye(3)
        Ainv = np.linalg.inv(A); Ainv[null] = 0
        self.D, self.Ainv = xp.asarray(D), xp.asarray(Ainv)

    def fft(self, f):
        return xp.fft.rfftn(f, axes=(-3, -2, -1))

    def ifft(self, f):
        return xp.fft.irfftn(f, s=(self.n,)*3, axes=(-3, -2, -1))

    def strain(self, u_hat):
        """Voxel-centre strain (6 comps xx,yy,zz,yz,xz,xy) of the periodic fluctuation."""
        G = self.ifft(self.D[:, None]*u_hat[None])          # G[j, a] = d u_a / d x_j
        return xp.stack([G[0, 0], G[1, 1], G[2, 2], .5*(G[1, 2] + G[2, 1]), .5*(G[0, 2] + G[2, 0]), .5*(G[0, 1] + G[1, 0])])

    def stress(self, eps, g):
        tr = eps[0] + eps[1] + eps[2]
        s = 2*self.mu*eps
        s[:3] += self.lam*tr
        return g*s

    def div(self, sig):
        """Adjoint of strain: nodal force f_a = sum_j conj(D_j) sigma_aj (Fourier)."""
        S = [[sig[0], sig[5], sig[4]], [sig[5], sig[1], sig[3]], [sig[4], sig[3], sig[2]]]
        Sh = [[self.fft(S[a][j]) for j in range(3)] for a in range(3)]
        return xp.stack([sum(self.D[j].conj()*Sh[a][j] for j in range(3)) for a in range(3)])

    def precondition(self, r_hat):
        return xp.einsum('...ab,b...->a...', self.Ainv, r_hat)

    def solve(self, g, E, u_hat=None, tol=1e-6, maxiter=2000):
        """Find the periodic fluctuation for mean strain E (6-vector) and stiffness field g."""
        Ev = xp.asarray(E, dtype=float).reshape(6, 1, 1, 1)
        b = -self.div(self.stress(xp.broadcast_to(Ev, (6,) + g.shape).copy(), g))
        K = lambda uh: self.div(self.stress(self.strain(uh), g))
        u = xp.zeros_like(b) if u_hat is None else u_hat
        r = b - K(u)
        z = self.precondition(r); p = z.copy()
        rz = xp.vdot(r, z).real
        bn = float(xp.sqrt(xp.vdot(b, self.precondition(b)).real)) or 1.
        it = 0
        for it in range(maxiter):
            if float(xp.sqrt(rz)) <= tol*bn:
                break
            Kp = K(p)
            alpha = rz/xp.vdot(p, Kp).real
            u = u + alpha*p; r = r - alpha*Kp
            z = self.precondition(r)
            rz_new = xp.vdot(r, z).real
            p = z + (rz_new/rz)*p; rz = rz_new
        eps = self.strain(u) + Ev
        return u, eps, it


class Creep3D:
    def __init__(self, material=Copper773(), stress_MPa=30., model='A', params=Params3D()):
        self.m, self.P, self.model = material, params, model
        self.S = material.stress(stress_MPa); self.S_MPa = stress_MPa
        n = params.n; self.n, self.h = n, 1/n
        nu = material.nu
        self.lam, self.mu = 2*nu/(1 - 2*nu), 1.
        self.Gc = 2*material.gamma_s/(material.mu*material.L)
        self.gamma = material.gamma_s/(material.mu*material.L)
        c = (np.arange(n) + .5)/n
        self.X2, self.Y2 = np.meshgrid(c, c, indexing='ij')
        if model == 'B':
            self.xc, self.R = np.zeros((0, 2)), np.zeros(0)
        else:
            self.xc, self.R = nuclei(params)
        self.c = np.zeros((n, n))
        self.sigma_n = np.full((n, n), self.S)
        self.t, self.created = 0., 0.
        k = 2*np.pi*np.fft.fftfreq(n, d=self.h)
        KX, KY = np.meshgrid(k, k, indexing='ij')
        self.k2 = (4*n**2)*(np.sin(KX*self.h/2)**2 + np.sin(KY*self.h/2)**2)        # 5-point Laplacian symbol
        self.k_s = 1/(params.chi*params.l_s**2)
        self.inventory0 = self.inventory()
        self.mech_on = model in ('A', 'A0', 'B')
        if self.mech_on:
            self.mech = Mechanics(n, self.lam, self.mu)
            g = (xp.arange(n) + .5)/n
            self.X3, self.Y3, self.Z3 = xp.meshgrid(g, g, g, indexing='ij')
            self.q = xp.zeros((n,)*3); self.H = xp.zeros((n,)*3)
            self.u_cases = [None]*3; self.E = None; self.E0 = None
            kk = 2*np.pi*np.fft.fftfreq(n, d=self.h); kz = 2*np.pi*np.fft.rfftfreq(n, d=self.h)
            a, b_, cz = np.meshgrid(kk, kk, kz, indexing='ij')
            self.lap3 = xp.asarray((4*n**2)*(np.sin(a*self.h/2)**2 + np.sin(b_*self.h/2)**2 + np.sin(cz*self.h/2)**2))
        self.history = []

    # -- geometry ------------------------------------------------------------------------------------------------
    def plane_distance(self, i):
        dx = (self.X2 - self.xc[i, 0] + .5) % 1 - .5; dy = (self.Y2 - self.xc[i, 1] + .5) % 1 - .5
        return np.hypot(dx, dy)

    def pore3d(self):
        p = xp.zeros((self.n,)*3)
        s = self.P.w_render/4.4
        for (x0, y0), r in zip(self.xc, self.R):
            dx = (self.X3 - x0 + .5) % 1 - .5; dy = (self.Y3 - y0 + .5) % 1 - .5; dz = self.Z3 - .5
            d = xp.sqrt(dx**2 + dy**2 + dz**2)
            p = xp.maximum(p, 1/(1 + xp.exp(xp.clip((d - r)/s, -50, 50))))
        return p

    def area_fraction(self):
        covered = np.zeros((self.n, self.n), bool)
        for i, r in enumerate(self.R):
            covered |= self.plane_distance(i) < r
        return float(covered.mean())

    def rings(self):
        w = []
        for i, r in enumerate(self.R):
            ring = np.exp(-((self.plane_distance(i) - r)/self.P.w_ring)**2)
            w.append(ring/(ring.sum()*self.h**2))
        return np.array(w) if w else np.zeros((0, self.n, self.n))

    def inventory(self):
        return float(self.c.sum()*self.h**2 + 4/3*np.pi*np.sum(self.R**3))

    def ledger_error(self):
        scale = self.inventory0 if self.inventory0 > 0 else 1.0     # model B has no cavities: absolute ledger
        return abs(self.inventory() - self.inventory0 - self.created)/scale

    # -- GB plane ------------------------------------------------------------------------------------------------
    def plane_steps(self, steps):
        P, dt, chi = self.P, self.P.dt, self.P.chi
        W = self.rings()
        sig_hat = np.fft.fft2(self.sigma_n)
        denom = 1 + dt*(chi*self.k2 + self.k_s*chi)
        for s in range(steps):
            if s and s % 20 == 0:
                W = self.rings()
            drive = chi*self.c[None] - 2*self.gamma/self.R[:, None, None]
            a = P.k_ex*W*drive*(2*np.pi*self.R[:, None, None])
            dV = a.sum(axis=(1, 2))*self.h**2*dt
            absorb = a.sum(axis=0)
            # c^{n+1}: (1 + dt(chi K2 + k_s chi)) c = c - dt absorb + dt(-K2 sigma... ) ; div grad mu = -K2 (chi c - sigma)
            rhs = np.fft.fft2(self.c - dt*absorb) + dt*(self.k2 + self.k_s)*sig_hat
            c_new = np.fft.ifft2(rhs/denom).real
            r_s = -self.k_s*(chi*c_new - self.sigma_n)
            self.created += float(r_s.sum()*self.h**2*dt)
            self.c = c_new
            self.R = np.cbrt(np.maximum(self.R**3 + dV*3/(4*np.pi), 0))
            self.t += dt

    # -- mechanics -----------------------------------------------------------------------------------------------
    def degradation(self, p):
        return (1 - self.q)**2*(1 - p)**2 + self.P.residual

    def equilibrate(self, p):
        g = self.degradation(p)
        stresses, epss, its = [], [], []
        for k in range(3):
            E = np.zeros(6); E[k] = 1.
            u, eps, it = self.mech.solve(g, E, self.u_cases[k], self.P.cg_tol)
            self.u_cases[k] = u; epss.append(eps); its.append(it)
            sig = self.mech.stress(eps, g)
            stresses.append(to_host(sig[:3].mean(axis=(1, 2, 3))))
        Mst = np.array(stresses).T                              # Mst[i, k] = <sigma_i> for unit E_k
        E = np.linalg.solve(Mst, np.array([0, 0, self.S]))
        eps = sum(E[k]*epss[k] for k in range(3))
        sig = self.mech.stress(eps, g)
        self.E = E
        if self.E0 is None:
            self.E0 = E.copy()
        self.eps, self.sig, self.cg_its = eps, sig, its
        z = self.n//2
        self.sigma_n = to_host(.5*(sig[2, :, :, z - 1] + sig[2, :, :, z]))
        return eps

    def damage(self, eps, p):
        K = self.lam + 2*self.mu/3
        tr = eps[0] + eps[1] + eps[2]
        dev = eps.copy(); dev[:3] -= tr/3
        psi = K/2*xp.maximum(tr, 0)**2 + self.mu*(dev[0]**2 + dev[1]**2 + dev[2]**2 + 2*(dev[3]**2 + dev[4]**2 + dev[5]**2))
        self.H = xp.maximum(self.H, psi)
        drive = 2*(1 - p)**2*self.H
        coef = self.Gc/self.P.ell + drive
        a0 = float(coef.mean()); gl = self.Gc*self.P.ell
        prec = lambda r: xp.fft.irfftn(xp.fft.rfftn(r)/(a0 + gl*self.lap3), s=r.shape, axes=tuple(range(r.ndim)))
        lap = lambda f: (xp.roll(f, 1, 0) + xp.roll(f, -1, 0) + xp.roll(f, 1, 1) + xp.roll(f, -1, 1)
                         + xp.roll(f, 1, 2) + xp.roll(f, -1, 2) - 6*f)*self.n**2
        A = lambda f: coef*f - gl*lap(f)
        q = self.q.copy(); r = drive - A(q); z = prec(r); d = z.copy(); rz = xp.vdot(r, z)
        bn = float(xp.linalg.norm(drive)) or 1.
        for _ in range(500):
            if float(xp.linalg.norm(r)) <= 1e-8*bn:
                break
            Ad = A(d); al = rz/xp.vdot(d, Ad)
            q += al*d; r -= al*Ad; z = prec(r); rz2 = xp.vdot(r, z); d = z + rz2/rz*d; rz = rz2
        q_new = xp.clip(q, 0, 1)
        change = float(xp.abs(q_new - self.q).max())
        self.q = q_new
        return change

    def mechanics(self):
        if not self.mech_on:
            self.sigma_n = np.full((self.n, self.n), self.S); return
        p = self.pore3d() if self.model != 'B' else xp.zeros((self.n,)*3)
        for _ in range(8):
            eps = self.equilibrate(p)
            if self.model == 'A0':
                break
            if self.damage(eps, p) < 1e-3:
                break
        self.p3 = p
        if self.model != 'A0':
            self.equilibrate(p)

    # -- failure and driver ---------------------------------------------------------------------------------------
    def severed_fraction(self):
        z = self.n//2
        d = 1 - (1 - self.q[:, :, z - 1:z + 1])*(1 - self.p3[:, :, z - 1:z + 1])
        return float((d.max(axis=2) > .9).mean())

    def failed(self):
        if self.model in ('A', 'B'):
            runaway = self.E is not None and self.E[2] > 20*self.E0[2]
            return self.severed_fraction() >= .98 or runaway
        return self.area_fraction() >= np.pi/4

    def snapshot(self):
        out = dict(t=self.t, t_hours=self.t*self.m.t_unit/3600, R_mean=float(self.R.mean()) if len(self.R) else 0.,
                   area_fraction=self.area_fraction(), ledger=self.ledger_error(),
                   sigma_n_max=float(self.sigma_n.max()), sigma_n_mean=float(self.sigma_n.mean()))
        if self.mech_on:
            out.update(max_q=float(self.q.max()), E_zz=float(self.E[2]), cg=self.cg_its,
                       severed=self.severed_fraction() if hasattr(self, 'p3') else 0.)
        return out

    # -- checkpoint / restart (state after a completed mechanics update; the solver object is rebuilt) -------------
    def checkpoint(self, path):
        import pickle
        state = {k: (('__gpu__', to_host(v)) if GPU and isinstance(v, xp.ndarray) else v)
                 for k, v in self.__dict__.items() if k != 'mech'}
        state['u_cases'] = [None if u is None else ('__gpu__', to_host(u)) for u in self.u_cases] if self.mech_on else None
        tmp = str(path) + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(state, f)
        import os
        os.replace(tmp, path)

    @classmethod
    def restore(cls, path):
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        back = lambda v: xp.asarray(v[1]) if isinstance(v, tuple) and len(v) == 2 and v[0] == '__gpu__' else v
        sim = cls.__new__(cls)
        sim.__dict__.update({k: back(v) for k, v in state.items() if k != 'u_cases'})
        if sim.mech_on:
            sim.u_cases = [None if u is None else back(u) for u in state['u_cases']]
            sim.mech = Mechanics(sim.n, sim.lam, sim.mu)
        return sim

    def run(self, t_max=25., dt_mech=.05, dt_mech_fine=.01, frames=(), callback=None, resume=False):
        if not resume:                  # a restored state already holds its mechanics (damage is irreversible)
            self.mechanics(); self.history.append(self.snapshot())
        frames = sorted(frames)
        while self.t < t_max - 1e-12:
            fine = self.mech_on and self.model != 'A0' and float(self.q.max()) >= .05
            interval = dt_mech_fine if fine else dt_mech
            if self.model != 'B':
                self.plane_steps(max(1, int(round(interval/self.P.dt))))
            else:
                self.t += interval
            self.mechanics()
            self.history.append(self.snapshot())
            while frames and self.t >= frames[0] - 1e-9:
                frames.pop(0)
                if callback:
                    callback(self)
            if self.failed():
                return dict(failed=True, t_f=self.t)
        return dict(failed=False, t_f=None)
