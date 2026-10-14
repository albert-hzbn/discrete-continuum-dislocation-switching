import numpy as np
import pytest
from bmvr.grid import Grid, circle
from bmvr.energy import DoubleWell
from bmvr.diagnostics import interface_radii
from bmvr.solvers import ac_step, anisotropic_mobility
from bmvr.continuum3d import Cube
from bmvr.loops3d import LoopMechanics, circular_loop, velocities


def test_anisotropic_mobility_factor_follows_the_normal_angle():
    grid = Grid.uniform(32.0, 64)
    band = np.abs(grid.X) < 8
    assert np.allclose(anisotropic_mobility(grid, np.tanh(grid.X / 2), .3)[band], 1.3)
    diagonal = (np.abs(grid.X + grid.Y) < 8)[1:-1, 1:-1]
    assert np.allclose(anisotropic_mobility(grid, np.tanh((grid.X + grid.Y) / 2), .3)[1:-1, 1:-1][diagonal], .7)
    assert np.array_equal(anisotropic_mobility(grid, np.ones(grid.shape), .3), np.ones(grid.shape))
    with pytest.raises(ValueError):
        anisotropic_mobility(grid, np.tanh(grid.X), 1.0)


def test_anisotropic_step_reduces_isotropic_step_and_dissipates_energy():
    grid, model = Grid.uniform(48.0, 96), DoubleWell()
    v = circle(grid, 10.0)
    assert np.array_equal(ac_step(grid, v, model, .02, mobility=anisotropic_mobility(grid, v, 0.0)), ac_step(grid, v, model, .02))
    energies = [model.value(grid, v)]
    for _ in range(50):
        v = ac_step(grid, v, model, .02, mobility=anisotropic_mobility(grid, v, .3))
        energies.append(model.value(grid, v))
    assert np.all(np.diff(energies) <= 1e-12)
    assert np.allclose(v, v.T, atol=1e-12) and np.allclose(v, v[::-1], atol=1e-12)


def test_interface_radii_recover_a_tanh_circle():
    grid = Grid.uniform(64.0, 256)
    radii = interface_radii(grid, circle(grid, 10.0), np.linspace(0, 2 * np.pi, 24, endpoint=False))
    assert np.max(np.abs(radii - 10.0)) < 2e-3


def test_loop_velocity_is_linear_in_applied_shear_with_the_polygon_slope():
    mech = LoopMechanics(Cube(12.0, 9))
    loop = circular_loop(radius=2.5, count=12, burgers=(.2, 0., 0.), core=.8, glide_normal=np.array([0., 0., 1.]))
    radial = loop.nodes[:, :2] / np.linalg.norm(loop.nodes[:, :2], axis=1)[:, None]
    speeds = []
    for tau in (-.01, 0., .01):
        strain = np.zeros((3, 3))
        strain[0, 2] = strain[2, 0] = tau / 2
        rates, _ = velocities([loop], mech.evaluate([loop], mean_strain=strain)["forces"])
        speeds.append(np.mean(np.sum(rates[0][:, :2] * radial, axis=1)))
    slope = (speeds[2] - speeds[0]) / .02
    assert abs(slope / (-.2 * np.cos(np.pi / 12)) - 1) < 1e-10
    assert abs(speeds[1] - (speeds[0] + speeds[2]) / 2) < 1e-14
