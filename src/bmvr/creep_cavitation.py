"""Section 97: grain-boundary creep cavitation coupled to phase-field fracture.

One energy carries vacancy transport with a grain-boundary (GB) source, cavity (pore) growth, elasticity degraded by
cavities and damage, and irreversible AT2 damage. Dimensionless units: domain side 1, shear modulus 1, Lame lambda 2.
Cell fields c (vacancy excess) and p (pore fraction) live on the Q1 elements of the Section 36 fracture solver.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .fracture2d import Fracture2D, FractureParameters, spectral_elasticity


@dataclass(frozen=True)
class CavitationParameters:
    chi: float = 1.0          # vacancy susceptibility (inverse thermodynamic factor)
    gamma: float = 1e-4       # cavity surface energy per unit length (2D)
    w_p: float = 1/64         # width used to render cavities into the stiffness
    w_gb: float = 1/64        # GB indicator width
    omega: float = 1.0        # vacancy volume (couples the normal stress to the vacancy potential)
    M_b: float = 1e-3         # bulk vacancy mobility
    M_gb: float = 1.0         # extra GB mobility
    k_ex: float = 10.0        # vacancy absorption rate at the cavity-GB junction
    k_s: float = 100.0        # GB source/sink rate
    nuclei: tuple = (0.125, 0.375, 0.625, 0.875)
    r0: float = 0.05          # initial cavity radius


class CoupledFracture(Fracture2D):
    """AT2 fracture whose stiffness is also degraded by the pore field: g = (1-q)^2 (1-p)^2 + residual."""

    def set_pores(self, p_cells):
        self.pore_gauss = np.repeat(p_cells.reshape(-1, 1), 4, axis=1)

    def evaluate(self, u, q, tangent=False):
        strain = np.einsum('gai,ei->ega', self.B, u[self.dofs]); damage = q[self.cells] @ self.N.T
        combined = 1 - (1 - damage)*(1 - self.pore_gauss)
        elastic = spectral_elasticity(strain, combined, self.p.lam, self.p.shear, self.p.residual, tangent)
        gradient = np.einsum('gai,ea->egi', self.gradient, q[self.cells])
        fracture = float(self.weight*np.sum(self.toughness*(damage**2/self.p.ell + self.p.ell*np.sum(gradient**2, axis=-1))/2))
        energy = float(self.weight*np.sum(elastic['density']))
        force_element = self.weight*np.einsum('gai,ega->ei', self.B, elastic['stress'])
        force = np.bincount(self.dofs.ravel(), weights=force_element.ravel(), minlength=2*self.node_count)
        result = {'energy': energy + fracture, 'elastic_energy': energy, 'fracture_energy': fracture, 'force': force,
                  'positive_energy': elastic['positive_energy'], 'damage_gauss': damage, 'strain': strain,
                  'stress': elastic['stress']}
        if tangent:
            element = self.weight*np.einsum('gai,egab,gbj->eij', self.B, elastic['tangent'], self.B)
            result['stiffness'] = self.matrix(element, self.dofs, 2*self.node_count)
        return result

    def damage_system(self, u):
        strain = np.einsum('gai,ei->ega', self.B, u[self.dofs])
        positive = spectral_elasticity(strain, np.zeros(strain.shape[:-1]), self.p.lam, self.p.shear, self.p.residual,
                                       False)['positive_energy']*(1 - self.pore_gauss)**2
        coefficient = 2*positive + self.toughness/self.p.ell
        element = self.weight*(np.einsum('eg,ga,gb->eab', coefficient, self.N, self.N)
                               + self.p.ell*np.einsum('eg,gai,gbi->eab', self.toughness, self.gradient, self.gradient))
        rhs_element = 2*self.weight*np.einsum('eg,ga->ea', positive, self.N)
        rhs = np.bincount(self.cells.ravel(), weights=rhs_element.ravel(), minlength=self.node_count)
        return self.matrix(element, self.cells, self.node_count), rhs


class CreepCavitation:
    """Cells (i along x, j along y); mode 'coupled', 'fracture' (no cavities) or 'uncoupled' (fixed stress)."""

    def __init__(self, n=64, mode='coupled', stress=0.016, cav=None, frac=None):
        self.n, self.mode, self.S = n, mode, stress
        self.cp = cav or CavitationParameters(); self.fp = frac or FractureParameters()
        self.h = 1/n
        xc = (np.arange(n) + .5)*self.h
        self.X, self.Y = np.meshgrid(xc, xc, indexing='ij')
        self.s = np.exp(-((self.Y - .5)/self.cp.w_gb)**2)
        self.M = self.cp.M_b + self.cp.M_gb*self.s
        self.Mx = .5*(self.M[1:, :] + self.M[:-1, :]); self.My = .5*(self.M[:, 1:] + self.M[:, :-1])
        self.c = np.zeros((n, n))
        self.xc = np.array(self.cp.nuclei if mode != 'fracture' else ())
        self.R = np.full(len(self.xc), self.cp.r0)
        self.render()
        self.sigma_n = np.full((n, n), stress)
        self.mech = None
        if mode != 'uncoupled':
            self.mech = CoupledFracture(n, parameters=self.fp)
            self.u = np.zeros(2*self.mech.node_count); self.q = np.zeros(self.mech.node_count); self.U = stress/2.67
        self.dt = .2*self.h**2/(self.M.max()*self.cp.chi)
        self.t, self.created = 0., 0.
        self.inventory0 = self.inventory()

    # -- chemistry -------------------------------------------------------------------------------------------
    def div_M_grad(self, mu):
        fx = self.Mx*(mu[1:, :] - mu[:-1, :])/self.h; fy = self.My*(mu[:, 1:] - mu[:, :-1])/self.h
        out = np.zeros_like(mu)
        out[:-1, :] += fx; out[1:, :] -= fx; out[:, :-1] += fy; out[:, 1:] -= fy
        return out/self.h

    def render(self):
        """Pore fraction seen by the mechanics: smooth discs of radius R_i centred on the GB."""
        self.p = np.zeros((self.n, self.n))
        for x0, r in zip(self.xc, self.R):
            d = np.hypot(self.X - x0, self.Y - .5)
            self.p = np.maximum(self.p, 1/(1 + np.exp(np.clip((d - r)/(self.cp.w_p/4.4), -50, 50))))

    def junctions(self):
        """Absorption weight of cavity i: its rendered rim times the GB indicator, normalised to unit integral."""
        out = []
        for x0, r in zip(self.xc, self.R):
            d = np.hypot(self.X - x0, self.Y - .5)
            rim = np.exp(-((d - r)/self.cp.w_p)**2)*self.s
            out.append(rim/(rim.sum()*self.h**2))
        return out

    def inventory(self):
        return float(self.c.sum()*self.h**2 + np.pi*np.sum(self.R**2))

    def chem_step(self):
        cp, dt = self.cp, self.dt
        mu_c = cp.chi*self.c - cp.omega*self.s*self.sigma_n
        r_s = -cp.k_s*self.s*mu_c
        rate = self.c*0
        dV = np.zeros(len(self.R))
        for i, w in enumerate(self.junctions()):
            # Hull-Rimmer: vacancies at the cavity-GB junction condense if chi*c exceeds the capillary potential gamma/R
            local = cp.k_ex*w*(cp.chi*self.c - cp.gamma/self.R[i])*(2*np.pi*self.R[i])
            rate -= local
            dV[i] = local.sum()*self.h**2*dt
        self.c = self.c + dt*(self.div_M_grad(mu_c) + r_s + rate)
        self.R = np.sqrt(np.maximum(self.R**2 + dV/np.pi, 0))
        self.created += float(r_s.sum()*self.h**2*dt)
        self.t += dt

    # -- mechanics ----------------------------------------------------------------------------------------
    def mechanics(self):
        if self.mech is None:
            self.sigma_n = np.full((self.n, self.n), self.S); return
        self.mech.set_pores(self.p if self.mode == 'coupled' else np.zeros_like(self.p))
        q_old = self.q.copy()
        for _ in range(4):                          # adjust the grip displacement to hold the mean stress S
            u, q, state = self.mech.solve_load(self.u, q_old, self.U, tolerance=1e-6)
            mean = state['reaction']/self.mech.width
            if mean <= 0:
                break
            ratio = self.S/mean
            self.u, self.q = u, q
            if abs(ratio - 1) < 1e-3:
                break
            self.U *= ratio
        self.state = state
        self.sigma_n = state['stress'][..., 1].mean(axis=1).reshape(self.n, self.n)

    def combined_damage_cells(self):
        qc = self.q[self.mech.cells].mean(axis=1).reshape(self.n, self.n) if self.mech is not None else 0*self.p
        return 1 - (1 - qc)*(1 - self.p)

    def failed(self):
        """True when cells with combined damage > 0.9 connect the left and right edges."""
        broken = self.combined_damage_cells() > .9
        labels, _ = ndimage.label(broken)
        left, right = set(labels[0, :][labels[0, :] > 0]), set(labels[-1, :][labels[-1, :] > 0])
        return bool(left & right)

    def ledger_error(self):
        return abs(self.inventory() - self.inventory0 - self.created)/max(self.inventory0, 1e-12)

    def run(self, t_max=20., dt_mech=0.05, frames=(), callback=None):
        """Mechanics and damage are re-equilibrated every dt_mech of diffusion time; frames are snapshot times."""
        self.mechanics()
        history, step, next_mech, frames = [self.snapshot()], 0, dt_mech, sorted(frames)
        while self.t < t_max:
            if self.mode != 'fracture':
                self.chem_step()
                if step % 50 == 0:
                    self.render()
            else:
                self.t += self.dt
            step += 1
            if self.t >= next_mech:
                next_mech += dt_mech
                self.render()
                self.mechanics()
                history.append(self.snapshot())
                while frames and self.t >= frames[0]:
                    frames.pop(0)
                    if callback: callback(self)
                if self.failed():
                    return dict(failed=True, t_f=self.t, history=history, steps=step)
        return dict(failed=False, t_f=None, history=history, steps=step)

    def snapshot(self):
        d = self.combined_damage_cells()
        return dict(t=float(self.t), pore_area=float(np.pi*np.sum(self.R**2)), radius=float(self.R.mean()) if len(self.R) else 0., max_damage=float(self.q.max()) if self.mech is not None else 0.,
                    max_sigma_n_gb=float((self.sigma_n*(self.s > .5)).max()), broken_fraction_gb=float((d[:, self.n//2 - 1:self.n//2 + 1] > .9).mean()),
                    ledger=self.ledger_error())
