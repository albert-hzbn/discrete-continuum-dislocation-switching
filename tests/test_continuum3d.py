"""Independent manufactured solutions and variational contracts for 3D physics."""

from dataclasses import replace
import numpy as np
import pytest
from bmvr.continuum3d import Cube, PeriodicElasticity, CoupledCrystal, Parameters3D, fcc_slip_systems


def test_3d_periodic_flux_conservation_and_adjoint():
    grid = Cube(9, 9)
    a, b = np.random.default_rng(51).normal(size=(2,) + grid.shape)
    assert abs(grid.integrate(grid.lap(a))) < 1e-11
    assert grid.integrate(a * grid.lap(b)) == pytest.approx(grid.integrate(b * grid.lap(a)), abs=1e-10)
    assert grid.integrate(a * grid.lap(a)) == pytest.approx(-2 * grid.gradient_energy(a), rel=1e-12)


def test_full_3d_homogeneous_loading_solution():
    grid, lam, shear = Cube(4, 7), 2., 3.
    mean = np.array([[.03, .02, -.01], [.02, -.015, .006], [-.01, .006, .025]])
    eigen = np.eye(3)[..., None, None, None] * np.full(grid.shape, .012)
    result = PeriodicElasticity(grid, lam, shear).solve(eigen, mean)
    strain = mean - .012 * np.eye(3)
    expected = 2 * shear * strain + lam * np.trace(strain) * np.eye(3)
    assert np.max(np.abs(result['stress'] - expected[..., None, None, None])) < 1e-12
    assert result['energy'] == pytest.approx(grid.volume * .5 * np.sum(expected * strain), rel=1e-12)
    assert result['equilibrium_rms'] < 1e-12


def test_3d_compatible_manufactured_eigenstrain_has_zero_stress():
    grid = Cube(2 * np.pi, 11)
    elastic = PeriodicElasticity(grid)
    x, y, z = grid.coordinates()
    displacement = np.array([.02 * np.sin(x + y), .03 * np.cos(y + z), .01 * np.sin(x - z)])
    grad = np.array([[elastic.derivative(displacement[i], j) for j in range(3)] for i in range(3)])
    eigen = (grad + grad.swapaxes(0, 1)) / 2
    result = elastic.solve(eigen)
    assert np.max(np.abs(result['stress'])) < 1e-12
    assert np.max(np.abs(result['displacement'] - displacement)) < 1e-12


def test_hydrostatic_sinusoidal_eigenstrain_analytical_solution():
    grid = Cube(2 * np.pi, 13)
    x, _, _ = grid.coordinates()
    amplitude = .02 * np.cos(2 * x)
    eigen = np.eye(3)[..., None, None, None] * amplitude
    result = PeriodicElasticity(grid, lam=2, shear=1).solve(eigen)
    assert np.max(np.abs(result['strain'][0, 0] - 2 * amplitude)) < 1e-12
    assert np.max(np.abs(result['strain'][1:, 1:])) < 1e-12
    assert result['equilibrium_rms'] < 1e-12


def test_arbitrary_eigenstrain_envelope_derivative():
    grid, rng = Cube(6, 7), np.random.default_rng(16)
    eigen, perturbation = rng.normal(scale=.02, size=(2, 3, 3) + grid.shape)
    eigen = (eigen + eigen.swapaxes(0, 1)) / 2
    perturbation = (perturbation + perturbation.swapaxes(0, 1)) / 2
    elastic = PeriodicElasticity(grid)
    result = elastic.solve(eigen)
    eps = 1e-5
    fd = (elastic.solve(eigen + eps * perturbation)['energy'] - elastic.solve(eigen - eps * perturbation)['energy']) / (2 * eps)
    exact = -grid.integrate(result['stress'] * perturbation)
    assert fd == pytest.approx(exact, rel=1e-7, abs=1e-10)
    assert result['equilibrium_rms'] < 1e-12


def test_fcc_systems_are_twelve_unique_isochoric_slips():
    s, m, schmid = fcc_slip_systems()
    assert len(s) == 12
    assert np.allclose(np.sum(s * m, axis=1), 0)
    assert np.allclose(np.linalg.norm(s, axis=1), 1)
    assert np.allclose(np.linalg.norm(m, axis=1), 1)
    assert np.max(np.abs(np.trace(schmid, axis1=1, axis2=2))) < 1e-12
    assert len({tuple(np.round(a.ravel(), 12)) for a in schmid}) == 12


@pytest.mark.parametrize('block', [0, 1, 2])
def test_each_coupled_3d_force_is_its_energy_derivative(block):
    grid, rng = Cube(6, 7), np.random.default_rng(29)
    model = CoupledCrystal(grid)
    fields = [rng.uniform(-.8, .8, grid.shape), rng.uniform(.15, .3, grid.shape),
              rng.normal(scale=.01, size=(12,) + grid.shape)]
    direction = rng.normal(size=fields[block].shape)
    eps = 1e-6
    plus, minus = list(fields), list(fields)
    plus[block] = plus[block] + eps * direction
    minus[block] = minus[block] - eps * direction
    fd = (model.evaluate(*plus)['energy'] - model.evaluate(*minus)['energy']) / (2 * eps)
    exact = grid.integrate(model.evaluate(*fields)['forces'][block] * direction)
    assert fd == pytest.approx(exact, rel=1e-6, abs=1e-7)


def test_homogeneous_single_slip_relaxation_and_yield_threshold():
    grid = Cube(3, 3)
    _, _, systems = fcc_slip_systems()
    S = systems[:1]
    imposed, shear, H, drag, Y = .12, 1., .2, 1., .015
    params = Parameters3D(phase_misfit=0, solute_expansion=0, binding=0,
                          hardening=H, yield_stress=Y, slip_drag=drag)
    model = CoupledCrystal(grid, params, mean_strain=imposed * S[0], schmid=S, shear=shear)
    fields = (np.full(grid.shape, -1.), np.full(grid.shape, .2), np.zeros((1,) + grid.shape))
    t, dt = 0., .01
    current = model.evaluate(*fields)
    for _ in range(100):
        fields, current, used, _ = model.step(fields, dt, current)
        t += used
    # sigma:S = shear*(imposed-gamma), because S:S=1/2.
    equilibrium = (shear * imposed - Y) / (shear + H)
    expected = equilibrium * (1 - np.exp(-(shear + H) * t / drag))
    assert fields[2].mean() == pytest.approx(expected, rel=.01)
    inactive = CoupledCrystal(grid, replace(params, yield_stress=1.), mean_strain=imposed * S[0], schmid=S)
    assert np.max(np.abs(inactive.evaluate(*fields)['rates'][2])) == 0


def test_coupled_3d_rates_dissipation_and_population_balance():
    grid = Cube(12, 13)
    x, y, z = grid.coordinates()
    model = CoupledCrystal(grid, mean_strain=np.array([[0, .04, 0], [.04, 0, 0], [0, 0, 0]]))
    fields = (np.tanh((3 - np.sqrt(x*x+y*y+z*z))/np.sqrt(2)),
              .2 + .03 * np.cos(2*np.pi*x/12), np.zeros((12,) + grid.shape))
    initial = model.evaluate(*fields)
    rate_energy = sum(grid.integrate(f * r) for f, r in zip(initial['forces'], initial['rates']))
    assert rate_energy == pytest.approx(-initial['dissipation'], rel=1e-12)
    inventory = grid.integrate(fields[1])
    current = initial
    for _ in range(10):
        before = current['energy']
        fields, current, used, trials = model.step(fields, .04, current)
        assert current['energy'] <= before + 1e-10
        assert abs(grid.integrate(fields[1]) - inventory) < 1e-10
        assert current['equilibrium_rms'] < 1e-10
    assert np.max(np.abs(fields[2])) > 0


def test_even_grid_is_rejected_instead_of_creating_nyquist_null_modes():
    with pytest.raises(ValueError):
        Cube(4, 8)
