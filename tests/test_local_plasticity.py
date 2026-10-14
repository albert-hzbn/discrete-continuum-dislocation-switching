import numpy as np
import pytest
from bmvr.grid import Grid, circle
from bmvr.adaptivity import focused_grid
from bmvr.defects import Screws
from bmvr.local_plasticity import AntiplaneElasticity, energy_forces

GRIDS = [Grid.uniform(32.0, 32, periodic=True), focused_grid(32.0, .5, ((-6, 6), (-6, 6)), periodic=True)]


@pytest.mark.parametrize("grid", GRIDS)
def test_slip_and_glide_forces_are_exact_energy_derivatives(grid):
    elastic = AntiplaneElasticity(grid)
    rng = np.random.default_rng(1)
    v, c, g = circle(grid, 4.0), .2 + .01 * rng.standard_normal(grid.shape), .02 * rng.standard_normal(grid.shape)
    screws = Screws(np.array([[-5.0, .3], [5.0, -.2]]), np.array([1.0, -1.0]), core=1.0)
    origins = screws.positions - np.array([[.3, 0], [.1, 0]])
    state = energy_forces(grid, elastic, v, c, g, screws, origins)
    for idx in [(3, 4), (grid.shape[0] // 2, grid.shape[1] // 2), (grid.shape[0] - 2, 1)]:
        e = np.zeros(grid.shape)
        e[idx] = 1e-6
        fd = (energy_forces(grid, elastic, v, c, g + e, screws, origins)["energy"] - energy_forces(grid, elastic, v, c, g - e, screws, origins)["energy"]) / 2e-6 / grid.vol[idx]
        assert np.isclose(fd, state["g_gamma"][idx], rtol=1e-6, atol=1e-9)
    for k in range(2):
        d = np.zeros((2, 2))
        d[k, 0] = 1e-6
        fd = -(energy_forces(grid, elastic, v, c, g, screws.moved(screws.positions + d), origins)["energy"]
               - energy_forces(grid, elastic, v, c, g, screws.moved(screws.positions - d), origins)["energy"]) / 2e-6
        assert np.isclose(fd, state["force"][k, 0], rtol=1e-6)


@pytest.mark.parametrize("grid", GRIDS)
def test_uniform_slip_reduces_to_the_mean_load_energy_and_face_shares_equal_volumes(grid):
    elastic = AntiplaneElasticity(grid)
    result = elastic.solve(np.full(grid.shape, .03), .15, .01)
    assert np.isclose(result["energy"], grid.area * (.15 - .01 - .03) ** 2 / 2, rtol=1e-12)
    share = np.asarray(abs(elastic.Ay).T @ elastic.Vy).reshape(grid.shape)
    assert np.allclose(share, grid.vol, rtol=1e-12)


def test_volume_weights_equal_the_declared_weights_on_uniform_grids():
    grid = GRIDS[0]
    assert (AntiplaneElasticity(grid).Ay != AntiplaneElasticity(grid, weighted=False).Ay).nnz == 0
