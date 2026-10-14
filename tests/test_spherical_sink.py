import numpy as np
from scipy.integrate import quad
from bmvr.spherical_sink import SphericalSink,reaction_profile,collocation_reference,exact_discrete_evolution


def test_curved_layer_has_declared_integrated_reactivity():
    for radius,width,reactivity in ((2.,1.,3.),(4.,.4,.2)):
        integral=quad(lambda r:4*np.pi*r*r*reaction_profile(r,radius,width,reactivity),
                      radius-width/2,radius+width/2,epsabs=1e-11)[0]
        assert np.isclose(integral,4*np.pi*radius**2*reactivity,rtol=1e-12)


def test_spherical_diffusion_recovers_laplacian_of_radius_squared():
    model=SphericalSink(cells=37)
    lap=-model.stiffness@(model.r**2)/model.vol
    np.testing.assert_allclose(lap[:-1],6.,rtol=1e-12,atol=1e-12)


def test_reciprocal_storage_conserves_population_and_discrete_energy_identity():
    model=SphericalSink(cells=32);capacity=100.;dt=.7
    mass,stiffness=model.closed_system(capacity)
    state=np.r_[.2+.1*np.sin(model.r),1.]
    updated=model.stepper(capacity,dt)(state)
    assert updated.min()>=0
    assert abs(mass@(updated-state))<1e-10
    delta_energy=.5*mass@(updated**2-state**2)
    dissipation=-dt*updated@(stiffness@updated)-.5*mass@((updated-state)**2)
    assert delta_energy<0
    np.testing.assert_allclose(delta_energy,dissipation,rtol=1e-11,atol=1e-10)
    # A loaded surface releases particles; exchange has not been hard-coded as absorption.
    assert updated[-1]<state[-1] and model.vol@(updated[:-1]-state[:-1])>0


def test_stationary_reference_and_uniform_closed_equilibrium():
    model=SphericalSink(cells=256)
    c,capture,incoming=model.steady()
    exact,rate,_=collocation_reference(model.r)
    assert np.sqrt(model.vol@((c-exact)**2)/(model.vol@(exact**2)))<.002
    assert abs(capture-rate)/rate<.002
    np.testing.assert_allclose(capture,incoming,rtol=1e-10)
    state=np.full(257,.4)
    np.testing.assert_allclose(exact_discrete_evolution(model,100.,state,2.),state,atol=1e-10)
