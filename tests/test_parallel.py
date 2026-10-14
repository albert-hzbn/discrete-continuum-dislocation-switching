import numpy as np
import pytest
from bmvr.grid import Grid, circle
from bmvr.energy import DoubleWell
from bmvr.adaptivity import focused_grid
from bmvr.solvers import ac_step, ch_step
from bmvr import parallel


@pytest.mark.parametrize("grid", [Grid.uniform(32.0, 64), focused_grid(64.0, .5, ((-8, 8), (-6, 10)))])
def test_parallel_ac_step_equals_numpy_step_for_every_thread_count(grid):
    model = DoubleWell(W=1.3, k=(1.0, .7))
    v = circle(grid, 5.0, center=(1.0, -.5))
    reference = ac_step(grid, v, model, .02, mobility=.8)
    results = []
    for threads in (1, 2, 4):
        parallel.set_threads(threads)
        results.append(parallel.ac_step(grid, v, model, .02, mobility=.8))
    assert np.max(np.abs(results[0] - reference)) < 1e-14
    assert all(np.array_equal(results[0], r) for r in results)


def test_parallel_ac_rejects_periodic_grids():
    grid = Grid.uniform(8.0, 16, periodic=True)
    with pytest.raises(ValueError):
        parallel.ac_step(grid, circle(grid, 2.0), DoubleWell(), .01)


def test_threaded_fft_ch_step_equals_numpy_step():
    grid = Grid.uniform(32.0, 64, periodic=True)
    c = .1 * np.random.default_rng(3).standard_normal(grid.shape)
    assert np.max(np.abs(parallel.ch_step(grid, c, .5, workers=4) - ch_step(grid, c, .5))) < 1e-14
