import numpy as np
import pytest
from scipy.linalg import expm
from bmvr.finite_elasticity import FiniteParameters, FiniteElasticity2D, constitutive


def state():
    F = np.array([[1.12, .07, 0], [0, .96, .02], [.01, 0, 1.03]])
    Fp = np.eye(3); Fp[0, 1] = .08
    return F, Fp, .3, .4, 1.1


def test_constitutive_energy_and_tangent_derivatives():
    F, *other = state(); base = constitutive(F, *other); step = 1e-6
    for k in range(3):
        for l in range(3):
            d = np.zeros((3, 3)); d[k, l] = step
            plus = constitutive(F+d, *other); minus = constitutive(F-d, *other)
            assert abs((plus['energy']-minus['energy'])/(2*step)-base['P'][k, l]) < 2e-9
            np.testing.assert_allclose((plus['P']-minus['P'])/(2*step), base['tangent'][:, :, k, l], atol=2e-9)
    np.testing.assert_allclose(base['tangent'], base['tangent'].transpose(2, 3, 0, 1), atol=1e-14)


def test_reciprocal_scalar_derivatives():
    args = list(state()); base = constitutive(*args); step = 1e-6
    for index, key, sign in ((2, 'gc', 1), (3, 'geta', 1), (4, 'entropy', -1)):
        plus = args.copy(); minus = args.copy(); plus[index] += step; minus[index] -= step
        a = constitutive(*plus); b = constitutive(*minus)
        assert abs((a['energy']-b['energy'])/(2*step)-sign*base[key]) < 2e-9
    assert abs(args[4]*(a['entropy']-b['entropy'])/(2*step)-base['heat_capacity']) < 3e-9


def test_objectivity_cauchy_symmetry_and_stress_free_growth():
    F, Fp, c, eta, T = state(); base = constitutive(F, Fp, c, eta, T)
    Q = expm(np.array([[0, -.3, .1], [.3, 0, -.2], [-.1, .2, 0]]))
    rotated = constitutive(Q@F, Fp, c, eta, T)
    assert abs(rotated['energy']-base['energy']) < 1e-14
    np.testing.assert_allclose(rotated['P'], Q@base['P'], atol=1e-14)
    np.testing.assert_allclose(base['sigma'], base['sigma'].T, atol=1e-14)
    a = np.exp(.02*(c-.2)+.015*eta**2*(3-2*eta)+.005*(T-1))
    np.testing.assert_allclose(constitutive(a*Fp, Fp, c, eta, T)['P'], 0, atol=1e-14)


def test_small_strain_limit():
    A = np.array([[.3, .2, 0], [-.1, -.2, .1], [0, 0, .05]])
    expected = A+A.T+2*np.trace(A)*np.eye(3)
    errors = [np.linalg.norm(constitutive(np.eye(3)+e*A)['P']/e-expected) for e in (.01, .005, .0025)]
    assert errors[0]/errors[1] > 1.99 and errors[1]/errors[2] > 1.99
    assert errors[-1]/np.linalg.norm(expected) < .001


@pytest.mark.parametrize('kwargs', [dict(F=np.diag([-1., 1, 1])), dict(Fp=2*np.eye(3)),
    dict(T=0), dict(c=1.1), dict(eta=-.1), dict(T=np.nan),
    dict(parameters=FiniteParameters(alpha_T=1.))])
def test_invalid_states_rejected(kwargs):
    with pytest.raises(ValueError):
        constitutive(**{'F': np.eye(3), **kwargs})


def test_assembled_potential_and_stiffness():
    model = FiniteElasticity2D(3, c=.3, eta=.4, T=1.1, body=[.2, -.1])
    rng = np.random.default_rng(713); u = rng.normal(size=2*model.mesh.node_count)*.003
    v = rng.normal(size=len(u)); v /= np.linalg.norm(v); eps = 1e-6
    base = model.evaluate(u, True); a = model.evaluate(u+eps*v); b = model.evaluate(u-eps*v)
    assert abs((a['potential']-b['potential'])/(2*eps)-base['force']@v) < 1e-9
    np.testing.assert_allclose((a['force']-b['force'])/(2*eps), base['stiffness']@v, atol=2e-9)


def test_finite_affine_patch():
    model = FiniteElasticity2D(6); X = model.mesh.coordinates
    exact = (X@np.array([[.2, .15], [.05, -.1]]).T).ravel(); fixed = model.boundary()
    initial = exact.copy(); rng = np.random.default_rng(41)
    free = np.ones(len(exact), dtype=bool); free[fixed] = False
    initial[free] += rng.normal(size=free.sum())*.002
    u, _, status = model.equilibrate(initial, fixed, exact[fixed])
    np.testing.assert_allclose(u, exact, atol=1e-10)
    assert status['relative_residual'] < 1e-9
    assert np.max(np.diff(status['potential'])) < 1e-13


def test_corner_inversion_caught_despite_positive_gauss_points():
    model = FiniteElasticity2D(2); x, y = model.mesh.coordinates.T
    u = np.column_stack((-1.05*x*y, np.zeros_like(x))).ravel()
    F2 = np.eye(2)+np.einsum('gijd,ed->egij', model.G, u[model.mesh.dofs])
    assert np.min(np.linalg.det(F2)) > 0
    with pytest.raises(ValueError, match='corner inversion'):
        model.evaluate(u)


def test_checkpoint_preserves_prescribed_fields_and_continuation(tmp_path):
    model = FiniteElasticity2D(4, c=lambda x, y: .2+.1*x, T=1.1, body=[.001, 0])
    x, y = model.mesh.coordinates.T
    lifting = np.column_stack((y*np.sin(np.pi*x), np.zeros_like(x))).ravel()
    fixed = model.boundary(); u, _, _ = model.equilibrate(.1*lifting, fixed, .1*lifting[fixed])
    path = tmp_path/'checkpoint.npz'; model.save(path, u, .1, 4)
    restored, v, clock = FiniteElasticity2D.load(path)
    assert clock == dict(load=.1, step=4)
    for name in ('c', 'eta', 'T', 'Fp', 'body', 'external'):
        np.testing.assert_array_equal(getattr(model, name), getattr(restored, name))
    a, _, _ = model.equilibrate(u+.02*lifting, fixed, .12*lifting[fixed])
    b, _, _ = restored.equilibrate(v+.02*lifting, fixed, .12*lifting[fixed])
    np.testing.assert_array_equal(a, b)


def test_unconverged_solve_is_failure():
    model = FiniteElasticity2D(3, body=[.01, 0]); fixed = model.boundary()
    with pytest.raises(RuntimeError, match='failed'):
        model.equilibrate(np.zeros(2*model.mesh.node_count), fixed, np.zeros(len(fixed)), max_iterations=0)
