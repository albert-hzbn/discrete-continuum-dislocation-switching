"""Tabulated single-element embedded-atom potential in the LAMMPS setfl (eam.alloy) format, as an ASE calculator.

The tables F(rho), f(r) and r phi(r) are interpolated with cubic splines; energy and forces reuse the vectorized ZhouEAM
assembly. Theory Section 82.6 (Mishin et al. 2001 copper)."""
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from .eam_zjw import ZhouEAM


def read_setfl(path):
    lines = Path(path).read_text().splitlines()
    nrho, drho, nr, dr, cutoff = lines[4].split(); nrho, nr = int(nrho), int(nr); drho, dr, cutoff = float(drho), float(dr), float(cutoff)
    head = lines[5].split(); values = np.array(' '.join(lines[6:]).split(), float)
    assert len(values) == nrho + 2*nr, 'single-element setfl expected'
    return dict(F=values[:nrho], f=values[nrho:nrho + nr], rphi=values[nrho + nr:], drho=drho, dr=dr, cutoff=cutoff,
                number=int(head[0]), mass=float(head[1]), lattice=float(head[2]), structure=head[3] if len(head) > 3 else '')


class SetflEAM(ZhouEAM):
    def __init__(self, path, skin=.5, **kwargs):
        t = read_setfl(path); super().__init__(cutoff=t['cutoff'], skin=skin, **kwargs); self.table = t
        rho = np.arange(len(t['F']))*t['drho']; r = np.arange(len(t['f']))*t['dr']
        self._F = CubicSpline(rho, t['F']); self._dF = self._F.derivative()
        self._f = CubicSpline(r, t['f']); self._df = self._f.derivative()
        self._rphi = CubicSpline(r, t['rphi']); self._drphi = self._rphi.derivative()

    def density(self, r):
        return self._f(r), self._df(r)

    def pair(self, r):
        rp = self._rphi(r); phi = rp/r; return phi, (self._drphi(r) - phi)/r

    def embedding(self, rho):
        return self._F(rho), self._dF(rho)
