import numpy as np
from bmvr.critical_nucleus import NucleationEnergy, planar_interface_energy


def test_nucleation_energy_gradient_and_hessian_are_exact():
    e = NucleationEnergy(16, dg=.05); rng = np.random.default_rng(4); phi = rng.uniform(0, 1, (16, 16)); g = e.gradient(phi); h = 1e-6
    for k in ((0, 0), (5, 9), (15, 3)):
        a = phi.copy(); a[k] += h; b = phi.copy(); b[k] -= h
        assert abs((e.energy(a) - e.energy(b))/(2*h) - g[k]) < 1e-6
    v = rng.standard_normal((16, 16)); Hv = (e.hessian(phi)@v.ravel()).reshape(16, 16)*e.h**2
    assert np.allclose(Hv, (e.gradient(phi + 1e-6*v) - e.gradient(phi - 1e-6*v))/2e-6, atol=1e-6)


def test_critical_nucleus_is_an_index_one_saddle_near_classical_theory():
    gamma = planar_interface_energy(); dg = .08; e = NucleationEnergy(64, dg=dg); phi, it = e.critical_nucleus(gamma/dg)
    assert np.abs(e.gradient(phi)).max()/e.h**2 < 1e-10
    low, near, _ = e.spectrum(phi, gamma/dg)
    assert low[0] < -1e-2 and low[1] > -1e-6 and np.all(abs(near[1:]) < 1e-6)
    barrier = e.energy(phi) - e.energy(0*phi); assert abs(barrier/(np.pi*gamma**2/dg) - 1) < .07
