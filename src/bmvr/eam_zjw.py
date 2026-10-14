"""Analytical Zhou-Johnson-Wadley embedded-atom potential (Phys. Rev. B 69, 144113, 2004) for copper, as an ASE calculator.

Energy E = sum_i F(rho_i) + 1/2 sum_{i != j} phi(r_ij), with rho_i = sum_j f(r_ij). Theory Section 82.2."""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

CU = dict(re=2.556162, fe=1.554485, rho_e=21.175871, rho_s=21.175395, alpha=8.127620, beta=4.334731, A=.396620, B=.548085,
          kappa=.308782, lam=.756515, Fn=(-2.170269, -.263788, 1.088878, -.817603), F=(-2.19, 0., .561830, -2.100595), eta=.310490, Fe=-2.186568)


def _term(r, amp, decay, shift, re):
    """amp exp(-decay (r/re - 1)) / (1 + (r/re - shift)^20) and its r-derivative."""
    x = r/re; e = amp*np.exp(-decay*(x - 1)); s = (x - shift)**19; d = 1 + s*(x - shift)
    return e/d, (-decay*e*d - e*20*s)/(d*d)/re


class ZhouEAM(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces', 'energies']

    def __init__(self, parameters=None, cutoff=6.0, skin=.5, **kwargs):
        super().__init__(**kwargs); self.p = dict(CU if parameters is None else parameters); self.cutoff = cutoff; self.skin = skin; self._list = None

    def density(self, r):
        p = self.p; return _term(r, p['fe'], p['beta'], p['lam'], p['re'])

    def pair(self, r):
        p = self.p; a, da = _term(r, p['A'], p['alpha'], p['kappa'], p['re']); b, db = _term(r, p['B'], p['beta'], p['lam'], p['re'])
        return a - b, da - db

    def embedding(self, rho):
        p = self.p; rn, re, r0 = .85*p['rho_e'], p['rho_e'], 1.15*p['rho_e']; F = np.empty_like(rho); dF = np.empty_like(rho)
        lo = rho < rn; hi = rho >= r0; mid = ~lo & ~hi
        for mask, coef, ref in ((lo, p['Fn'], rn), (mid, p['F'], re)):
            y = rho[mask]/ref - 1; F[mask] = sum(c*y**i for i, c in enumerate(coef)); dF[mask] = sum(i*c*y**(i - 1) for i, c in enumerate(coef) if i)/ref
        z = rho[hi]/p['rho_s']; eta = p['eta']; zeta = z**eta
        F[hi] = p['Fe']*(1 - eta*np.log(z))*zeta; dF[hi] = p['Fe']*(-eta**2*np.log(z))*zeta/rho[hi]
        return F, dF

    def _pairs(self, atoms):
        """Neighbour pairs within cutoff + skin, rebuilt only when an atom has moved more than skin/2."""
        pos = atoms.positions
        if (self._list is None or len(pos) != len(self._list[3]) or not np.array_equal(atoms.cell.array, self._list[4])
                or np.max(np.linalg.norm(pos - self._list[3], axis=1)) > self.skin/2):
            i, j, S = neighbor_list('ijS', atoms, self.cutoff + self.skin); self._list = (i, j, S@atoms.cell.array, pos.copy(), atoms.cell.array.copy())
        i, j, shift = self._list[:3]; D = pos[j] + shift - pos[i]; r = np.linalg.norm(D, axis=1); keep = r < self.cutoff
        return i[keep], j[keep], D[keep], r[keep]

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes); n = len(self.atoms)
        i, j, D, r = self._pairs(self.atoms); f, df = self.density(r); phi, dphi = self.pair(r)
        rho = np.bincount(i, weights=f, minlength=n); F, dF = self.embedding(rho)
        energies = F + .5*np.bincount(i, weights=phi, minlength=n)
        g = (.5*dphi + dF[i]*df)/r  # derivative of the terms owned by i along the ordered pair (i, j), divided by r
        forces = np.zeros((n, 3))
        for k in range(3):
            forces[:, k] = np.bincount(i, weights=g*D[:, k], minlength=n) - np.bincount(j, weights=g*D[:, k], minlength=n)
        self.results = dict(energy=float(energies.sum()), free_energy=float(energies.sum()), energies=energies, forces=forces)
