import numpy as np
from bmvr.nodal_damage import NodalThermalFracture2D
from bmvr.thermofracture import ThermalFractureParameters
from bmvr.dynamic_fracture import DynamicFracture2D
from bmvr.fracture2d import FractureParameters


def half_broken(tmp_path=None):
    model = NodalThermalFracture2D(4, degree=2, fracture=ThermalFractureParameters(elastic_split='volumetric-deviatoric'))
    m = model.mechanics.mesh
    x, y = m.coordinates.T
    u = np.zeros(2 * m.node_count)
    u[1::2] = .01 * (y - .5)
    model.values = u[model.fixed].copy()
    s = model.initial(u=u, T=1.0, c=.3, eta=.2, damage=.2)
    d = s.damage.reshape(5, 5).copy()
    d[:3, 2] = 1.0  # damage nodes x = 0, 0.25, 0.5 on y = 0.5: edges 0 and 1 broken
    s.damage = d.ravel()
    return model, s


def test_node_release_splits_behind_the_tip_and_conserves_energy(tmp_path):
    model, s = half_broken()
    m = model.mechanics.mesh
    before_nodes = m.node_count
    lower_cells = m.cells[np.arange(4) * 4 + 1].copy()
    s2, record = model.split_broken_line(s)
    assert record['nodes'] == 4 and record['ledger_error'] < 1e-13
    rows = 9
    expected = {1 * rows + 4, 3 * rows + 4, 0 * rows + 4, 2 * rows + 4}  # two midpoints, x=0 and x=0.25 corners
    assert set(model.split) == expected and 4 * rows + 4 not in model.split  # the tip corner (x=0.5) stays shared
    assert m.node_count == before_nodes + 4 and len(s2.u) == 2 * m.node_count
    assert np.array_equal(m.cells[np.arange(4) * 4 + 1], lower_cells)  # cells below the line keep the originals
    # The faces are decoupled: moving an upper duplicate leaves the force on its original unchanged.
    a = min(model.split)
    b = model.split[a]
    f0 = model.mechanics.evaluate(s2.u)['force'][2 * a:2 * a + 2]
    moved = s2.u.copy()
    moved[2 * b + 1] += 1e-3
    assert np.allclose(model.mechanics.evaluate(moved)['force'][2 * a:2 * a + 2], f0, atol=1e-14)
    model.save(tmp_path / 'split.npz', s2)
    loaded, state, _ = NodalThermalFracture2D.load(tmp_path / 'split.npz')
    assert loaded.split == model.split and np.array_equal(loaded.mechanics.mesh.cells, m.cells)
    assert np.isclose(loaded.rate(state)['diagnostics']['total_energy'], model.rate(s2)['diagnostics']['total_energy'], rtol=1e-13)
    assert model.split_broken_line(s2)[1] is None


def test_dynamic_fracture_mass_load_conservation_and_irreversibility():
    model = DynamicFracture2D(24, FractureParameters(ell=.1), traction=.002)
    assert np.isclose(model.mass.sum() / 2, 1.0) and abs(model.external.sum()) < 1e-15
    s = model.initial()
    s['q'][:] = 0.0  # undamaged: an elastic wave problem
    e = model.energies(s)
    H0 = e['kinetic'] + e['elastic'] + e['fracture'] - e['work_potential']
    peak = 0.0
    for _ in range(60):
        previous = s['q']
        s, _ = model.step(s, .01)
        assert np.all(s['q'] >= previous)
        e = model.energies(s)
        peak = max(peak, e['kinetic'] + e['elastic'])
    H = e['kinetic'] + e['elastic'] + e['fracture'] - e['work_potential']
    assert abs(H - H0) < .02 * peak


def test_rectangular_dynamic_domain_keeps_mass_load_balance_and_band_counting():
    model = DynamicFracture2D(10, FractureParameters(ell=.2), traction=.002, nx=20)
    b = model.base
    assert b.width == 2.0 and b.node_count == 21 * 11 and len(b.cells) == 200
    assert np.isclose(model.mass.sum() / 2, 2.0) and abs(model.external.sum()) < 1e-15
    assert np.isclose(model.external[1::2].clip(min=0).sum(), .002 * 2.0)  # traction times the top-edge length
    q = np.zeros(b.node_count)
    grid = q.reshape(21, 11)
    grid[15, 2] = grid[15, 8] = 1.0  # two separate bands crossing x = 1.5
    assert model.branches(q, 1.5) == 2 and model.centerline_tip(q) == 1.5


def test_reactive_erosion_removes_only_refusing_broken_cells_and_conserves_energy():
    model, s = half_broken()
    s, _ = model.split_broken_line(s)
    candidates = model.broken_edge_cells(s)
    broken_cell = int(np.flatnonzero(candidates)[0])
    intact_cell = int(np.flatnonzero(~candidates)[0])
    model.mechanics.refused_cells = np.array([broken_cell, intact_cell])
    same, event = model.erode_refused(s)
    assert event is None and same is s  # an intact refusing cell means a genuine failure: nothing is removed
    model.mechanics.refused_cells = np.array([broken_cell])
    E0 = model.rate(s)['diagnostics']['total_energy']
    s2, event = model.erode_refused(s)
    assert event['cells'] == [broken_cell] and event['ledger_error'] < 1e-12
    assert not model.mechanics.active[broken_cell] and model.mechanics.active.sum() == len(candidates) - 1
    assert np.isclose(model.rate(s2)['diagnostics']['total_energy'], E0, rtol=1e-12, atol=0)
    assert model.mechanics.refused_cells is None


def test_smeared_to_discrete_transition_is_stress_free_energy_exact_and_consistent(tmp_path):
    model, s = half_broken()
    s, _ = model.split_broken_line(s)
    E0 = model.rate(s)['diagnostics']['total_energy']
    s2, event = model.discretize_released(s)
    assert sorted(event['cells']) == [1, 2, 5, 6]  # cells below and above the two released line edges (n = 4, row j0 = 2)
    assert event['ledger_error'] < 1e-12
    assert np.isclose(model.rate(s2)['diagnostics']['total_energy'], E0, rtol=1e-12, atol=0)
    mech = model.mechanics
    model.install(s)  # at the pre-transition temperature the new cells are intact and stress-free in their current shape
    assert np.abs(mech.evaluate(s.u)['local']['P'][event['cells']]).max() < 1e-12
    assert np.all(mech.elastic_scale[event['cells']] == 1.)
    model.install(s2)  # the released energy heats them slightly, so a small isotropic thermal stress remains
    ev = mech.evaluate(s2.u, tangent=True)
    # The transformed tangent is the derivative of the transformed force.
    rng = np.random.default_rng(3)
    du = 1e-7 * rng.standard_normal(len(s2.u))
    df = mech.evaluate(s2.u + du)['force'] - ev['force']
    assert np.linalg.norm(df - ev['stiffness'] @ du) <= 1e-4 * np.linalg.norm(df)
    assert model.discretize_released(s2)[1] is None
    model.save(tmp_path / 'discrete.npz', s2)
    loaded, state, _ = NodalThermalFracture2D.load(tmp_path / 'discrete.npz')
    assert np.array_equal(loaded.discrete, model.discrete) and np.array_equal(loaded.mechanics.reset, mech.reset)
    assert np.isclose(loaded.rate(state)['diagnostics']['total_energy'], model.rate(s2)['diagnostics']['total_energy'], rtol=1e-13)


def test_band_transition_adds_fully_formed_cells_only_in_released_columns():
    model, s = half_broken()
    d = s.damage.reshape(5, 5).copy()
    d[:, [0, 1, 3, 4]] = 0.95  # the band is fully formed everywhere off the line, behind and ahead of the tip
    d[3:, 2] = 0.95  # ahead of the tip the line itself is not broken, so those columns are not released
    s.damage = d.ravel()
    s, _ = model.split_broken_line(s)
    face = model.released_face_cells()
    band = model.released_face_cells(s, band=.9)
    assert np.array_equal(np.flatnonzero(face), [1, 2, 5, 6])
    assert set(np.flatnonzero(band)) == {0, 1, 2, 3, 4, 5, 6, 7}  # all four rows of released columns 0 and 1, nothing ahead
    s2, event = model.discretize_released(s, band=.9)
    assert sorted(event['cells']) == [0, 1, 2, 3, 4, 5, 6, 7] and event['ledger_error'] < 1e-12


def test_elastic_flank_freezes_plastic_flow_only_in_discrete_cells(tmp_path):
    model, s = half_broken()
    s, _ = model.split_broken_line(s)
    s, _ = model.discretize_released(s)
    base = model.rate(s)
    model.elastic_discrete = True
    frozen = model.rate(s)
    assert np.all(frozen['gamma'][model.discrete] == 0)
    assert np.array_equal(frozen['gamma'][~model.discrete], base['gamma'][~model.discrete])
    model.save(tmp_path / 'elastic.npz', s)
    loaded, state, _ = NodalThermalFracture2D.load(tmp_path / 'elastic.npz')
    assert loaded.elastic_discrete and np.all(loaded.rate(state)['gamma'][loaded.discrete] == 0)
