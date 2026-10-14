import numpy as np
import pytest

pytest.importorskip("ase")
from ase.build import bulk
from bmvr.eam_zjw import ZhouEAM


def test_zjw_forces_are_the_exact_energy_gradient():
    at = bulk('Cu', 'fcc', a=3.615, cubic=True).repeat((2, 2, 2)); at.rattle(.05, seed=3); at.calc = ZhouEAM(); F = at.get_forces(); h = 1e-5
    for k, c in ((0, 0), (5, 1), (19, 2)):
        a = at.copy(); a.calc = ZhouEAM(); a.positions[k, c] += h; ep = a.get_potential_energy(); a.positions[k, c] -= 2*h; em = a.get_potential_energy()
        assert abs(-(ep - em)/(2*h) - F[k, c]) < 1e-7
    assert abs(F.sum(axis=0)).max() < 1e-10


def test_zjw_copper_reproduces_its_fitted_bulk_state():
    energy = lambda a: (lambda b: (setattr(b, 'calc', ZhouEAM()), b.get_potential_energy()/len(b))[1])(bulk('Cu', 'fcc', a=a, cubic=True))
    e = [energy(a) for a in (3.605, 3.615, 3.625)]
    assert e[1] < e[0] and e[1] < e[2] and abs(e[1] + 3.54) < 1e-3
    assert abs(ZhouEAM().embedding(np.array([1e-12]))[0][0]) < 1e-5  # an isolated atom has zero energy


def test_setfl_mishin_reproduces_its_fit_and_exact_forces():
    import pytest
    from pathlib import Path
    from bmvr.eam_setfl import SetflEAM, read_setfl
    path = Path(__file__).resolve().parents[1]/'data'/'potentials'/'Cu01.eam.alloy'
    if not path.exists(): pytest.skip('Mishin potential file not present')
    t = read_setfl(path); assert t['number'] == 29 and abs(t['lattice'] - 3.615) < 1e-9 and abs(t['cutoff'] - 5.506786) < 1e-6
    b = bulk('Cu', 'fcc', a=3.615, cubic=True); b.calc = SetflEAM(path); assert abs(b.get_potential_energy()/len(b) + 3.54) < 1e-4
    at = bulk('Cu', 'fcc', a=3.615, cubic=True).repeat((2, 2, 2)); at.rattle(.05, seed=5); at.calc = SetflEAM(path); F = at.get_forces(); h = 1e-5
    for k, c in ((0, 0), (9, 2)):
        a = at.copy(); a.calc = SetflEAM(path); a.positions[k, c] += h; ep = a.get_potential_energy(); a.positions[k, c] -= 2*h
        assert abs(-(ep - a.get_potential_energy())/(2*h) - F[k, c]) < 1e-7
