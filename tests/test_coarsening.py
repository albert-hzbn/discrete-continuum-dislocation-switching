import numpy as np
from bmvr.grid import Grid
from bmvr.coarsening import periodic_cell_resample,initial_coarsening_field,coarsening_windows,radial_structure


def test_cell_center_phase_shift_preserves_known_continuum_mode():
    coarse=Grid.uniform(32,33,periodic=True);fine=Grid.uniform(32,96,periodic=True)
    def mode(g):return np.sin(2*np.pi*(3*g.X+2*g.Y)/32)+.3*np.cos(2*np.pi*g.X/32)
    np.testing.assert_allclose(periodic_cell_resample(mode(coarse),32,96),mode(fine),atol=1e-13)
    a=initial_coarsening_field(128,128,1701);b=initial_coarsening_field(128,256,1701)
    np.testing.assert_allclose(periodic_cell_resample(a,128,256),b,atol=1e-13)
    np.testing.assert_allclose(np.sqrt(np.mean(a*a)),.04,atol=1e-14)


def test_physical_structure_factor_obeys_parseval():
    grid=Grid.uniform(32,64,periodic=True);field=np.cos(2*np.pi*4*grid.X/32)
    k,power,counts,spectrum=radial_structure(grid,field)
    np.testing.assert_allclose(spectrum.sum()/grid.area,np.var(field),atol=1e-13)
    assert np.argmax(power)==4
    np.testing.assert_allclose(np.sum(power*counts),spectrum.sum(),atol=1e-13)


def test_window_fits_recover_power_and_keep_box_failure_visible():
    time=np.arange(256,4097,16);length=4*time**(1/3)
    rows=coarsening_windows(time,length,.2*length,128)
    for row in rows:
        np.testing.assert_allclose(row['spectral_exponent'],1/3,atol=1e-13)
        np.testing.assert_allclose(row['energy_exponent'],1/3,atol=1e-13)
    assert not rows[-1]['scale_separated']
