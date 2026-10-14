import numpy as np
from bmvr.grid import Grid
from bmvr.solvers import ch_step, ch_sbdf2
from bmvr.coarsening import initial_coarsening_field
from bmvr.dynamic_fracture import ThermalDynamicFracture2D
from bmvr.fracture2d import FractureParameters


def test_sbdf2_is_second_order_and_conserves_mass():
    g = Grid.uniform(16., 32, periodic=True); sym = g.spectrum(); c0 = initial_coarsening_field(16., 32, 3)
    def run(dt, T=2.):
        c, prev = c0.copy(), None
        for _ in range(round(T/dt)):c, prev = ch_sbdf2(g, c, prev, dt, symbol=sym), c
        return c
    ref = run(1/320); e1 = np.sqrt(np.mean((run(.1)-ref)**2)); e2 = np.sqrt(np.mean((run(.05)-ref)**2))
    assert np.log2(e1/e2) > 1.8
    assert abs(run(.1).mean()-c0.mean()) < 1e-12


def test_thermal_dynamic_fracture_heat_is_nonnegative_and_energy_is_booked():
    m = ThermalDynamicFracture2D(24, FractureParameters(ell=.1), traction=.05, nx=48)
    s = m.initial(); e = m.energies(s); H0 = e['kinetic']+e['elastic']+e['heat']-e['work_potential']; peak = 0.
    for _ in range(60):
        s, _ = m.step(s, .005); e = m.energies(s); peak = max(peak, e['kinetic']+e['elastic'])
    assert s['T'].min() >= m.T0-1e-12 and e['heat'] >= 0
    assert abs(e['kinetic']+e['elastic']+e['heat']-e['work_potential']-H0) < .02*peak
