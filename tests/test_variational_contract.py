import numpy as np
import pytest
from bmvr.grid import Grid, circle
from bmvr.energy import DoubleWell, pore_energy, pore_forces, grain_energy, grain_force
from bmvr.adaptivity import focused_grid, transfer, adapt
from bmvr.solvers import ac_step, ch_step, project_simplex
from bmvr.elasticity import coherent_elasticity
from bmvr.defects import Screws, binding, coupled_energy_forces, core_kernel
from bmvr.reactions import annihilate


def directional_difference(fn, value, direction, eps=1e-6):
    return (fn(value + eps * direction) - fn(value - eps * direction)) / (2 * eps)


@pytest.mark.parametrize("periodic", [False, True])
def test_finite_volume_mass_and_summation_by_parts(periodic):
    g = Grid(np.array([0, .3, 1, 2]), np.array([0, .2, .8, 2]), periodic)
    rng = np.random.default_rng(41)
    v, w = rng.normal(size=g.shape), rng.normal(size=g.shape)
    assert abs(g.integrate(g.lap(v))) < 1e-12
    assert g.integrate(v * g.lap(w)) == pytest.approx(g.integrate(w * g.lap(v)), abs=1e-12)
    assert g.integrate(v * g.lap(v)) == pytest.approx(-2 * g.gradient_energy(v), abs=1e-12)


def test_nonuniform_anisotropic_energy_derivative():
    g = focused_grid(16, .5, ((-3, 3), (-2, 4)))
    rng = np.random.default_rng(5)
    v, d = rng.normal(size=g.shape), rng.normal(size=g.shape)
    model = DoubleWell(k=(.7, 1.4))
    measured = directional_difference(lambda a: model.value(g, a), v, d)
    assert measured == pytest.approx(g.integrate(model.force(g, v) * d), rel=1e-7)


def test_ac_energy_decays_with_resolved_profile():
    g = Grid.uniform(32, 64)
    v, m = circle(g, 7), DoubleWell()
    old = m.value(g, v)
    for _ in range(30):
        v = ac_step(g, v, m, .01)
        new = m.value(g, v)
        assert new <= old + 1e-12
        old = new


def test_ch_mass_energy_and_linear_mode():
    g = Grid.uniform(32, 64, periodic=True)
    mode = np.cos(2 * np.pi * g.X / 32)
    c = 1e-5 * mode
    m = DoubleWell()
    old = m.value(g, c)
    result = ch_step(g, c, .01)
    lam = 4 * np.sin(np.pi / 64) ** 2 / .5 ** 2
    expected = 1 + .01 * lam * (1 - lam)
    measured = np.sum(result * mode) / np.sum(c * mode)
    assert measured == pytest.approx(expected, rel=1e-5)
    assert abs(g.integrate(result - c)) < 1e-12
    assert m.value(g, result) <= old + 1e-12


def test_overlap_transfer_conserves_and_preserves_bounds():
    old, new = Grid.uniform(16, 32), focused_grid(16, .5, ((-3, 3), (-2, 4)))
    c = np.random.default_rng(8).uniform(.1, .9, old.shape)
    mapped = transfer(old, new, c)
    assert old.integrate(c) == pytest.approx(new.integrate(mapped), abs=1e-12)
    assert c.min() <= mapped.min() <= mapped.max() <= c.max()


def test_adaptation_accepts_constant_state_without_artificial_energy():
    old = Grid.uniform(32, 64)
    v = -np.ones(old.shape)
    m = DoubleWell()
    new, fields, record = adapt(old, [v], .5, 32, lambda g, f: m.value(g, f[0]))
    assert record["changed"]
    assert new.size < old.size
    assert np.all(fields[0] == -1)
    assert record["common_grid_energy_jump"] == 0


def test_simplex_and_grain_energy_derivative():
    g = Grid.uniform(8, 16)
    rng = np.random.default_rng(11)
    p = project_simplex(rng.normal(size=(4, *g.shape)))
    assert p.min() >= 0
    assert np.max(np.abs(p.sum(axis=0) - 1)) < 1e-14
    direction = rng.normal(size=p.shape)
    measured = directional_difference(lambda a: grain_energy(g, a), p, direction)
    expected = g.integrate(np.sum(grain_force(g, p) * direction, axis=0))
    assert measured == pytest.approx(expected, rel=1e-7)


def test_condensed_elastic_force_and_equilibrium():
    g = Grid.uniform(16, 33, periodic=True)
    v = circle(g, 4)
    d = np.random.default_rng(12).normal(size=g.shape)
    result = coherent_elasticity(g, v)
    measured = directional_difference(lambda a: coherent_elasticity(g, a)["energy"], v, d)
    assert measured == pytest.approx(g.integrate(result["force"] * d), rel=1e-6, abs=1e-9)
    assert result["equilibrium_residual"] < 1e-12
    assert coherent_elasticity(g, v, misfit=0)["energy"] == 0


def test_pair_force_action_reaction_and_energy_derivative():
    s = Screws(np.array([[-2., 0], [2., 0]]), np.array([1., -1.]))
    _, f = s.pair()
    assert np.linalg.norm(f.sum(axis=0)) < 1e-14
    direction = np.array([[1., 0], [0, 0]])
    measured = directional_difference(lambda x: s.moved(x).pair()[0], s.positions, direction)
    assert -measured == pytest.approx(f[0, 0], rel=1e-7)


def test_reciprocal_binding_including_kernel_normalization():
    g = focused_grid(16, .5, ((-3, 3), (-3, 3)))
    v, c = circle(g, 3), .2 + .03 * np.sin(g.X)
    s = Screws(np.array([[1.3, .4]]), np.array([1.]))
    e, gv, gc, f = binding(g, v, c, s)
    direction = np.random.default_rng(13).normal(size=g.shape)
    assert directional_difference(lambda a: binding(g, a, c, s)[0], v, direction) == pytest.approx(g.integrate(gv * direction), rel=1e-6)
    assert directional_difference(lambda a: binding(g, v, a, s)[0], c, direction) == pytest.approx(g.integrate(gc * direction), rel=1e-6)
    move = np.array([[1., 0]])
    assert -directional_difference(lambda x: binding(g, v, c, s.moved(x))[0], s.positions, move) == pytest.approx(f[0, 0], rel=1e-6)
    rho, rx, ry = core_kernel(g, [1.3, .4], s.core)
    assert g.integrate(rho) == pytest.approx(1)
    assert abs(g.integrate(rx)) + abs(g.integrate(ry)) < 1e-12


def test_annihilation_balances_charge_and_energy():
    s = Screws(np.array([[-.2, 0], [.2, 0]]), np.array([1., -1.]))
    result, event = annihilate(s)
    assert event["occurred"] and len(result.burgers) == 0
    assert event["released_energy"] > 0 and event["burgers_error"] == 0
    equal = Screws(s.positions, np.array([1., 1.]))
    assert not annihilate(equal)[1]["occurred"]


def test_pore_conversion_mass_and_dissipation():
    g = Grid.uniform(16, 32)
    p = (circle(g, 3) + 1) / 2
    n = np.full(g.shape, .12)
    un, up = pore_forces(g, n, p)
    reaction = p ** 2 * (1 - p) ** 2 * (up - un)
    dn, dp = g.lap(un) + reaction, -reaction
    assert abs(g.integrate(dn + dp)) < 1e-12
    predicted = g.integrate(un * dn + up * dp)
    assert predicted < 0
    eps = 1e-5
    observed = (pore_energy(g, n + eps * dn, p + eps * dp) - pore_energy(g, n, p)) / eps
    assert observed == pytest.approx(predicted, rel=1e-3)


def test_full_coupled_glide_force_includes_plastic_work():
    g = Grid.uniform(16, 32)
    s = Screws(np.array([[-3., 0], [3., 0]]), np.array([1., -1.]))
    origins = s.positions - np.array([[.2, 0], [-.2, 0]])
    v, c = circle(g, 4), np.full(g.shape, .2)
    result = coupled_energy_forces(g, v, c, s, origins)
    direction = np.array([[1., 0], [0, 0]])
    measured = directional_difference(lambda x: coupled_energy_forces(g, v, c, s.moved(x), origins)["energy"], s.positions, direction)
    assert -measured == pytest.approx(result["force"][0, 0], rel=1e-5)


def test_checkpoint_continuation_is_identical(tmp_path):
    from bmvr.checkpoint import save, load
    grid, model = Grid.uniform(16, 32), DoubleWell()
    state = circle(grid, 4)
    for _ in range(8):
        state = ac_step(grid, state, model, .01)
    save(tmp_path / "restart.npz", grid, {"phase": state}, .08, .01, {"model": "AC", "W": 1, "kappa": 1})
    restored_grid, fields, meta = load(tmp_path / "restart.npz")
    direct = ac_step(grid, state, model, .01)
    resumed = ac_step(restored_grid, fields["phase"], model, meta["dt"])
    assert np.array_equal(direct, resumed)
    assert meta["time"] == .08
    assert model.value(grid, state) == model.value(restored_grid, fields["phase"])


def test_fourier_solver_rejects_nonuniform_grid():
    g = focused_grid(16, .5, ((-3, 3), (-2, 2)))
    with pytest.raises(ValueError):
        ch_step(g, np.zeros(g.shape), .1)


def test_absent_grain_stays_absent_at_a_three_phase_junction():
    from bmvr.solvers import grain_step
    g = Grid.uniform(8, 16)
    phases = np.zeros((4, *g.shape))
    phases[:3] = 1 / 3
    next_phases, _ = grain_step(g, phases, .01)
    assert np.all(next_phases[3] == 0)
    assert np.max(np.abs(next_phases.sum(axis=0) - 1)) < 1e-14
