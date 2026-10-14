import numpy as np
from bmvr.pore_topology import PoreModel, GAMMA


def test_availability_limiter_is_inactive_away_from_the_bound():
    model = PoreModel(length=32.0)
    n, p = np.full(model.grid.shape, .02), model.pore((0.0, 0.0), 6.0)
    energy = model.energy(n, p)
    limited, free = model.step(n, p, energy, limiter=True), model.step(n, p, energy, limiter=False)
    assert np.array_equal(limited[0], free[0]) and np.array_equal(limited[1], free[1])


def test_limited_steps_conserve_inventory_keep_bounds_and_descend():
    model = PoreModel(length=32.0)
    n = np.full(model.grid.shape, 1e-4)
    p = np.maximum(model.pore((-5.0, 0.0), 4.5), model.pore((5.0, 0.0), 4.5))
    energy, inventory = model.energy(n, p), model.inventory(n, p)
    active = 0
    for _ in range(200):
        n, p, new, _ = model.step(n, p, energy)
        assert new <= energy + 1e-11 and n.min() >= 0 and 0 <= p.min() and p.max() <= 1
        energy, active = new, active + model.limited
    assert active > 0
    assert abs(model.inventory(n, p) - inventory) / inventory < 1e-12


def test_periodic_component_count_and_conserving_insertion():
    model = PoreModel(length=64.0)
    assert model.components(model.pore((31.0, 0.0), 4.0))[1] == 1
    assert model.components(np.maximum(model.pore((-12.0, 0.0), 5.0), model.pore((12.0, 0.0), 5.0)))[1] == 2
    n, p = np.full(model.grid.shape, .08), np.zeros(model.grid.shape)
    new_n, new_p, record = model.insert(n, p, (3.0, -2.0))
    assert abs(model.inventory(new_n, new_p) - model.inventory(n, p)) / model.inventory(n, p) < 1e-13
    assert new_n.min() >= 0 and record["radius"] > GAMMA / record["depleted_mean"]
    assert model.insert(np.zeros_like(n), p, (0.0, 0.0))[2]["refused"] == "undersaturated"
    assert model.insert(new_n, new_p, (4.0, -2.0), exclusion=4 / np.sqrt(2))[2]["refused"] == "overlaps_existing_pore"
