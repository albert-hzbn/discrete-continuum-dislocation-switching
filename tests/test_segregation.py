import numpy as np
from bmvr.segregation import Segregation1D


def test_segregation_forces_are_exact_energy_derivatives():
    model = Segregation1D(n=40, length=10.0)
    eta, c = model.initial(.05)
    eta = eta + .02 * np.sin(model.x)
    c = c * (1 + .1 * np.cos(model.x))
    for i in (3, 20, 37):
        e = np.zeros(model.n)
        e[i] = 1e-6
        assert np.isclose((model.energy(eta + e, c) - model.energy(eta - e, c)) / 2e-6 / model.h, model.phase_force(eta, c)[i], rtol=1e-6)
        assert np.isclose((model.energy(eta, c + e) - model.energy(eta, c - e)) / 2e-6 / model.h, model.potential(eta, c)[i], rtol=1e-6)


def test_equilibrium_conserves_solute_follows_isotherm_and_gibbs_equation():
    model = Segregation1D(n=80, length=20.0)
    states = []
    for mean in (.0199, .02, .0201):
        eta, c = model.initial(mean)
        eta, c, info = model.equilibrate(eta, c, .02)
        assert info["mass_drift"] < 1e-12 and info["max_energy_increase"] < 1e-12
        ex = model.excess(eta, c)
        assert np.max(np.abs(c - model.isotherm(eta, ex["mu"]))) < 1e-8
        states.append(ex)
    slope = (states[2]["gamma"] - states[0]["gamma"]) / (states[2]["mu"] - states[0]["mu"])
    assert abs(slope / -states[1]["adsorption"] - 1) < 1e-4
    assert states[1]["adsorption"] > 0
