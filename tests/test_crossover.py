import numpy as np
from bmvr.adaptivity import focused_grid
from bmvr.grid import Grid


def test_focused_grid_cell_ratio_matches_tensor_product_prediction():
    length, h = 128., .5
    uniform = Grid.uniform(length, int(length / h))
    for b in (16., 28., 42., 64.):  # band edges on the coarse lattice (multiples of 4h), as in the benchmark
        g = 2 * b / length
        adaptive = focused_grid(length, h, ((-b, b), (-b, b)))
        assert np.isclose(uniform.size / adaptive.size, 1 / (g + (1 - g) / 4) ** 2)
