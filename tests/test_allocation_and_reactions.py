from dataclasses import replace
import numpy as np
import pytest
from bmvr.grid import Grid
from bmvr.energy import DoubleWell
from bmvr.allocation import inverse_diffusion_map, TransferCache, two_grid_indicator, plan_grid, split_candidates, allocate
from bmvr.interface_reactions import InterfaceParameters, InterfaceReactionModel, approach_exposure


def simple_fields(grid, fields):
    v, c = fields
    model = DoubleWell()
    return {"energy": model.value(grid, v) + grid.integrate(c * c / 2),
            "rates": [-model.force(grid, v), grid.lap(c)]}


def test_inverse_diffusion_metric_matches_neumann_operator():
    g = Grid.uniform(16, 32)
    potential = np.cos(np.pi * (g.X + 8) / 16) * np.cos(2 * np.pi * (g.Y + 8) / 16)
    rate = -g.lap(potential)
    cell, recovered, removed = inverse_diffusion_map(g, rate)
    assert abs(removed) < 1e-12
    assert np.max(np.abs(recovered - potential)) < 1e-12
    assert cell.min() >= 0
    assert cell.sum() == pytest.approx(g.integrate(rate * potential), rel=1e-12)
    assert cell.sum() == pytest.approx(2 * g.gradient_energy(potential), rel=1e-12)


def test_nonconservative_rate_is_rejected():
    g = Grid.uniform(8, 16)
    with pytest.raises(ValueError, match="nonzero inventory"):
        inverse_diffusion_map(g, np.ones(g.shape))


def test_sparse_overlap_transfer_preserves_inventory():
    fine, coarse = Grid.uniform(16, 32), Grid.uniform(16, 8)
    c = np.random.default_rng(7).normal(size=fine.shape)
    cache = TransferCache()
    projected = cache.project(fine, coarse, c)
    assert coarse.integrate(projected) == pytest.approx(fine.integrate(c), abs=1e-12)
    assert np.array_equal(cache.project(fine, fine, c), c)


def test_full_grid_discrepancy_is_zero():
    g = Grid.uniform(16, 32)
    fields = [-np.ones(g.shape), .2 + .01 * np.cos(np.pi * g.X / 8)]
    report = two_grid_indicator(g, fields, g, simple_fields)
    assert report["met"]
    assert max(report["squared"]) < 1e-25


def test_diffusion_feature_refines_without_a_phase_interface():
    g = Grid.uniform(24, 48)
    fields = [-np.ones(g.shape), .2 + .1 * np.exp(-((g.X - 5) ** 2 + (g.Y - 5) ** 2) / 2)]
    coarse, _, limited = plan_grid(g, fields, simple_fields, .5, max_splits=0)
    refined, _, report = plan_grid(g, fields, simple_fields, .5)
    assert not limited["met"]
    assert report["met"]
    assert refined.size > coarse.size
    assert report["selected"]
    assert any(c["low"] < 5 < c["high"] for c in report["selected"])


@pytest.mark.parametrize("angle", [0, 15, 45, 80])
def test_every_reaction_state_balances_burgers_and_slip_geometry(angle):
    m = InterfaceReactionModel(InterfaceParameters(angle_degrees=angle))
    assert abs(m.outgoing_burgers @ m.outgoing_plane) < 1e-14
    expected = 2 * (1 - np.cos(np.deg2rad(angle)))
    assert np.sum(m.residual[1] ** 2) == pytest.approx(expected, abs=1e-14)
    for state in range(4):
        assert np.linalg.norm(m.charge(state) - m.incident_burgers) < 1e-14
    reflected = m.event(0, 3)
    assert reflected["normal_travel_direction"] == -1
    assert np.array_equal(m.mobile[3], m.incident_burgers)
    assert m.event(0, 2)["mobile_count"] == 0


def test_transition_rates_satisfy_detailed_balance():
    m = InterfaceReactionModel()
    flux = m.stationary[:, None] * m.rates
    assert np.max(np.abs(flux - flux.T)) < 1e-14
    assert np.max(np.abs(m.generator @ m.stationary)) < 1e-14
    assert np.max(np.abs(m.generator.sum(axis=0))) < 1e-14


def test_thermal_evolution_decreases_ensemble_free_energy():
    m = InterfaceReactionModel()
    p = m.propagate(np.linspace(0, 80, 401))
    energy = m.ensemble_free_energy(p)
    assert p.min() >= -1e-14
    assert np.max(np.abs(p.sum(axis=1) - 1)) < 1e-12
    assert np.max(np.diff(energy)) < 1e-12
    state = p[30]
    derivative = (m.energies + m.parameters.thermal_energy * (np.log(state) + 1)) @ (m.generator @ state)
    assert derivative == pytest.approx(-m.parameters.thermal_energy * m.entropy_production(state), rel=1e-12)
    assert m.event(0, 3)["configuration_energy_change"] > 0  # A thermal uphill channel remains valid.


def test_first_event_sampling_matches_exact_rates():
    m = InterfaceReactionModel()
    waits, states = m.sample_first_events(20000, seed=19)
    expected = m.first_event_probabilities()
    observed = np.bincount(states, minlength=4) / len(states)
    se = np.sqrt(expected * (1 - expected) / len(states))
    assert np.all(np.abs(observed - expected) <= 5 * se + 1 / len(states))
    assert waits.min() >= 0
    for state in np.unique(states):
        assert m.event(0, int(state))["burgers_error"] < 1e-12


def test_finite_width_gate_integrates_to_closed_form():
    times = np.linspace(0, 24, 2401)
    distance, exposure = approach_exposure(times)
    derivative = np.gradient(exposure, times)
    exact = 1 / np.cosh(distance / np.sqrt(2)) ** 2
    assert np.max(np.abs(derivative[1:-1] - exact[1:-1])) < 2e-5
    _, narrow = approach_exposure(times, width=.7)
    assert exposure[1000] > narrow[1000]


def test_cost_and_residual_rankings_differ_when_strip_costs_differ():
    g = Grid(np.arange(0., 9., 2.), np.arange(0., 9.))
    fine = Grid(np.arange(0., 8.5, .5), np.arange(0., 8.5, .5))
    indicator = {"ranking_map": np.ones(fine.shape)}
    raw = split_candidates(g, fine, indicator, mode="residual")[0]
    budgeted = split_candidates(g, fine, indicator, mode="cost")[0]
    assert raw["axis"] == "x"
    assert budgeted["axis"] == "y"
    assert budgeted["benefit"] < raw["benefit"]
    assert budgeted["added_dofs"] < raw["added_dofs"]


def test_allocation_limit_does_not_claim_budget_satisfaction():
    g = Grid.uniform(24, 48)
    fields = [-np.ones(g.shape), .2 + .1 * np.exp(-((g.X - 5) ** 2 + (g.Y - 5) ** 2) / 2)]
    returned, _, report = allocate(g, fields, simple_fields, .5, max_splits=0)
    assert not report["met"]
    assert not report["accepted"]
    assert returned is g
    with pytest.raises(ValueError, match="exceeds the cell budget"):
        plan_grid(g, fields, simple_fields, .5, max_cells=1)


def test_refinement_reuses_existing_edges():
    g = Grid(np.r_[np.arange(-8, -4, 2), np.arange(-4, 4.5, .5), [6, 8]],
             np.arange(-8, 8.5, .5))
    fields = [-np.ones(g.shape), np.full(g.shape, .2)]
    candidate, _, report = plan_grid(g, fields, simple_fields, .5, refine_only=True)
    assert np.isin(g.xe, candidate.xe).all()
    assert np.isin(g.ye, candidate.ye).all()
    assert report["met"]


def test_zero_flux_channel_does_not_hide_entropy_production():
    m = InterfaceReactionModel()
    assert np.isinf(m.entropy_production(np.array([0., 0., 1., 0.])))
