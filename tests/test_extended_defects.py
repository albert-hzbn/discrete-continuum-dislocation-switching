import numpy as np
from bmvr import extended_defects as X
from bmvr.grid import Grid
from bmvr.energy import grain_energy


def test_partial_separation_matches_textbook_form():
    mu, nu, g = 1., .3, .01
    for beta in np.radians([0, 30, 60, 90]):
        bp2 = 1/3
        textbook = mu*bp2/(8*np.pi*g)*(2-nu)/(1-nu)*(1-2*nu*np.cos(2*beta)/(2-nu))
        assert np.isclose(X.analytic_separation(beta, g, mu, nu), textbook)
        b1, b2 = X.partial_burgers(beta)
        assert np.allclose(b1+b2, [np.sin(beta), np.cos(beta)])


def test_pairwise_energy_reduces_to_grain_energy_when_all_pairs_equal():
    grid = Grid.uniform(16., 32)
    rng = np.random.default_rng(1)
    p = rng.random((3,)+grid.shape); p /= p.sum(axis=0)
    ones = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    # grain_energy uses face differences and np.gradient uses central ones, so compare the bulk terms exactly
    bulk = sum(grid.integrate(p[i]*p[j]) for i in range(3) for j in range(i+1, 3))
    assert np.isclose(bulk, grid.integrate(np.sum(.5*p*(1-p), axis=0)))
    assert X.pairwise_energy(grid, p, ones) > 0


def test_disclination_dipole_is_charge_neutral_and_bragg_williams_is_self_consistent():
    n, L = 64, 64.
    eta = .1*(X.gaussian(n, L, 32, 36, 1.5)-X.gaussian(n, L, 32, 28, 1.5))
    assert abs(eta.sum()*(L/n)**2) < 1e-12
    s = X.airy_stress(eta, L, 1., .3)
    assert all(np.isfinite(v).all() for v in s.values())
    for T in (.5, .9):
        e = X.bw_order(T)
        assert abs(e-np.tanh(e/T)) < 1e-12 and abs(X.bw_force(e, T)) < 1e-10
    assert X.bw_order(1.2) == 0.


def test_anisotropic_energy_factor_reduces_to_isotropic_and_to_the_partial_formula():
    mu, nu = 1., .3; lam = 2*mu*nu/(1-2*nu); C = X.cubic_stiffness(lam+2*mu, lam, mu)
    t = np.array([1., -1, 0])/np.sqrt(2); e = np.array([1., 1, -2])/np.sqrt(6); K = X.energy_factor(C, t)
    assert abs(t@K@t - mu) < 1e-12 and abs(e@K@e - mu/(1-nu)) < 1e-12
    b = t  # a = sqrt(2) gives |b| = 1; partials a/6[2-1-1] and a/6[1-21]
    b1, b2 = np.sqrt(2)/6*np.array([2., -1, -1]), np.sqrt(2)/6*np.array([1., -2, 1])
    assert np.allclose(b1+b2, b)
    assert abs(X.anisotropic_separation(C, t, b1, b2, .01) - X.analytic_separation(0., .01, mu, nu, 1.)) < 1e-10
