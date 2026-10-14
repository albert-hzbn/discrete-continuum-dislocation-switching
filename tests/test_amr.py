import numpy as np
from bmvr.amr import QuadtreeAMR
from bmvr.grid import Grid
from bmvr.energy import DoubleWell
from bmvr.solvers import ac_step
from bmvr import decomposition as D
from bmvr.grid import circle


def test_regrid_is_balanced_covers_domain_and_refines_the_interface():
    amr = QuadtreeAMR(length=64.0, base_blocks=2, block=8, levels=3)
    X, Y = amr.finest_centers()
    fine = np.tanh((10.0 - np.hypot(X, Y)) / np.sqrt(2))
    data = amr.regrid(fine=fine)
    levelmap = amr.level_map(amr.leaves)
    assert sum(4 ** (amr.L - l) for l in amr.levels) == amr.nf ** 2  # leaves tile the domain exactly
    assert np.abs(np.diff(levelmap, axis=0)).max() <= 1 and np.abs(np.diff(levelmap, axis=1)).max() <= 1
    interface = (np.abs(fine) < .995).reshape(amr.nf, amr.B, amr.nf, amr.B).any(axis=(1, 3))
    assert (levelmap[interface] == amr.L).all()
    # Conservative transfer: block means reproduce the finest-level integral.
    area = sum(data[k].sum() * amr.h(l) ** 2 for k, l in enumerate(amr.levels))
    assert np.isclose(area, fine.sum() * amr.h_fine ** 2, rtol=1e-12)


def test_uniform_tree_step_equals_the_uniform_finite_volume_step():
    amr = QuadtreeAMR(length=32.0, base_blocks=2, block=8, levels=1)
    X, Y = amr.finest_centers()
    fine = np.tanh((6.0 - np.hypot(X, Y)) / np.sqrt(2))
    data = amr.regrid(fine=fine)
    assert (amr.levels == 1).all()
    grid = Grid.uniform(32.0, amr.n)
    reference = ac_step(grid, fine, DoubleWell(), .01)
    assert np.max(np.abs(amr.prolong(amr.step(data, .01)) - reference)) < 1e-12


def test_message_passing_strips_equal_the_serial_step_bitwise():
    grid = Grid.uniform(64.0, 128)
    v = circle(grid, 8.0)
    reference = v.copy()
    for _ in range(4):
        reference = ac_step(grid, reference, DoubleWell(), .02)
    _, field = D.run(grid, v, 3, 2, 2, .02)
    assert np.array_equal(field, reference)


def test_tcp_strips_match_serial_bitwise():
    from bmvr import decomposition as D
    from bmvr.energy import DoubleWell
    from bmvr.solvers import ac_step
    g = Grid.uniform(32., 64)
    v0 = circle(g, 8.)
    ref = v0.copy()
    for _ in range(6):
        ref = ac_step(g, ref, DoubleWell(), .02)
    _, field = D.run(g, v0, 3, 4, 2, .02, transport="tcp", base_port=47250)
    assert np.array_equal(field, ref)
