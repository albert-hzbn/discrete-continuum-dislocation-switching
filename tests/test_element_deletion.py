import numpy as np
import pytest
from bmvr.finite_elasticity import constitutive, FiniteParameters
from bmvr.nodal_damage import NodalThermalFracture2D
from test_nodal_damage import example


def test_masked_constitutive_keeps_only_chemical_and_thermal_terms():
    p = FiniteParameters()
    F = np.array([[1.1, .2, 0], [.05, .9, 0], [0, 0, 1.]])
    live = constitutive(F, c=.3, eta=.4, T=1.2, parameters=p, elastic_split='volumetric-deviatoric')
    dead = constitutive(F, c=.3, eta=.4, T=1.2, parameters=p, elastic_split='volumetric-deviatoric', mechanical=0.)
    assert np.all(dead['P'] == 0) and np.all(dead['tangent'] == 0) and dead['elastic'] == 0 and dead['damage_elastic'] == 0
    assert np.isclose(dead['gc'], .3 - p.c0) and np.isclose(dead['heat_capacity'], p.cv)
    assert np.isclose(dead['entropy'], p.cv * np.log(1.2 / p.T0))
    assert np.isclose(dead['energy'], live['energy'] - live['elastic'])
    same = constitutive(F, c=.3, eta=.4, T=1.2, parameters=p, elastic_split='volumetric-deviatoric', mechanical=1.)
    assert all(np.array_equal(live[k], same[k]) for k in ('P', 'energy', 'entropy', 'heat_capacity', 'gc'))


def saturated_edge(model, state):
    d = state.damage.copy()
    cell = model.damage_cells[4]
    d[cell[:2]] = 1.0  # one edge fully broken
    state.damage = d
    return cell


@pytest.mark.parametrize("degree", [1, 2])
def test_deletion_conserves_total_energy_and_removes_mechanics(degree, tmp_path):
    model, s = example(degree, 'AT2')
    saturated_edge(model, s)
    before = model.rate(s)['diagnostics']['total_energy']
    s2, record = model.delete_broken(s)
    assert record is not None and 4 in record['cells']
    assert record['ledger_error'] < 1e-13 and abs(model.rate(s2)['diagnostics']['total_energy'] - before) < 1e-13 * abs(before)
    assert not model.mechanics.active[4]
    if degree == 2:
        # Folding a deleted cell through its own centre node (shared with no other cell) is not refused;
        # the same fold in an active cell is.
        m = model.mechanics.mesh
        centre = [i for i in m.cells[4] if np.allclose(m.coordinates[i], m.coordinates[m.cells[4]].mean(0))][0]
        folded = s2.u.copy()
        folded[2 * centre] += 10 * (m.coordinates[:, 0].max() / m.n)
        model.mechanics.evaluate(folded)
        model.mechanics.active[4] = True
        with pytest.raises(ValueError):
            model.mechanics.evaluate(folded)
        model.mechanics.active[4] = False
    model.mechanics.subdivide = 4
    model.save(tmp_path / 'deleted.npz', s2)
    loaded, state, _ = NodalThermalFracture2D.load(tmp_path / 'deleted.npz')
    assert np.array_equal(loaded.mechanics.active, model.mechanics.active)
    assert loaded.mechanics.subdivide == 4
    assert np.isclose(loaded.rate(state)['diagnostics']['total_energy'], model.rate(s2)['diagnostics']['total_energy'], rtol=1e-13)
    assert model.delete_broken(s2)[1] is None  # irreversible and idempotent


def test_no_deletion_without_a_saturated_edge():
    model, s = example(2, 'AT2')
    assert model.delete_broken(s)[1] is None
    assert getattr(model.mechanics, 'active', None) is None


def test_subdivided_certificate_is_a_rigorous_and_sharper_lower_bound():
    from bmvr.quadratic_mesh import subdivided_bound, _SPLIT
    rng = np.random.default_rng(7)
    basis = lambda t: np.stack(((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3), -1)
    t = np.linspace(0, 1, 101)
    sharper = 0
    for _ in range(100):
        C = rng.normal(size=(4, 4)) + 2.0
        true = np.einsum('ia,ab,jb->ij', basis(t), C, basis(t)).min()
        bound = subdivided_bound(C, 6)
        assert bound <= true + 1e-12
        sharper += C.min() <= 0 < bound
    assert sharper > 0  # some patches refused by the plain bound are certified after subdivision
    A = _SPLIT[0]
    C = rng.normal(size=(4, 4))
    assert np.allclose(np.einsum('ia,ab,jb->ij', basis(t), A @ C @ A.T, basis(t)),
                       np.einsum('ia,ab,jb->ij', basis(t / 2), C, basis(t / 2)), atol=1e-14)


def test_c2_split_derivatives_are_exact_inside_the_band_and_heat_capacity_is_continuous():
    p = FiniteParameters(split_smoothing=1e-3)
    base = np.diag([np.exp(-2.5e-4), np.exp(-2.5e-4), 1.0])  # ln J = -5e-4, inside the band
    base[0, 1] = 2e-4
    kw = dict(c=p.c0, eta=0., T=p.T0, parameters=p, elastic_scale=np.array(.3), elastic_split='volumetric-deviatoric')
    r = constitutive(base, **kw)
    h = 1e-7
    for i, j in ((0, 0), (1, 1), (0, 1)):
        plus, minus = base.copy(), base.copy()
        plus[i, j] += h
        minus[i, j] -= h
        assert np.isclose((constitutive(plus, **kw)['energy'] - constitutive(minus, **kw)['energy']) / (2 * h), r['P'][i, j], rtol=1e-6, atol=1e-10)
        dP = (constitutive(plus, **kw)['P'] - constitutive(minus, **kw)['P']) / (2 * h)
        assert np.allclose(dP, r['tangent'][:, :, i, j], rtol=1e-5, atol=1e-8)
    capacity = lambda l: constitutive(np.diag([np.exp(l / 2), np.exp(l / 2), 1.]), **kw)['heat_capacity']
    assert abs(capacity(-1e-9) - capacity(1e-9)) < 1e-8
    assert abs(capacity(-1e-3 - 1e-9) - capacity(-1e-3 + 1e-9)) < 1e-6
    sharp = dict(kw, parameters=FiniteParameters())
    jump = lambda l: constitutive(np.diag([np.exp(l / 2), np.exp(l / 2), 1.]), **sharp)['heat_capacity']
    assert abs(jump(-1e-9) - jump(1e-9)) > 1e-4  # the C1 split does jump
