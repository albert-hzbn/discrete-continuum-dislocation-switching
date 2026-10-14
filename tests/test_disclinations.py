import numpy as np
from bmvr.disclinations import Disclinations, applied_shear


def test_disclination_forces_are_the_exact_energy_gradient():
    d = Disclinations(96, 96.); pos = np.array([[40., 45.], [55., 47.], [60., 70.]]); om = [.2, -.3, .1]; F = d.forces(pos, om); h = 1e-4
    for i, k in ((0, 0), (1, 1), (2, 0)):
        p = pos.copy(); p[i, k] += h; ep = d.energy(p, om); p[i, k] -= 2*h
        assert abs(-(ep - d.energy(p, om))/(2*h) - F[i, k]) < 1e-8*abs(F).max()


def test_disclination_dipole_feels_the_peach_koehler_force_and_newton_pairs_cancel():
    d = Disclinations(96, 96.); a = 2.; w = .25; pos = np.array([[48., 48 + a], [48., 48 - a]]); om = [w, -w]
    F = d.forces(pos, om, applied_shear(.01)); assert abs(F[:, 0].sum() - .01*2*a*w) < 1e-12
    Fi = d.forces(pos, om); assert np.allclose(Fi.sum(axis=0), 0, atol=1e-12)
