import numpy as np
import pytest
from scipy.linalg import expm
from bmvr.finite_elasticity import constitutive, FiniteParameters
from bmvr.thermoplastic import (ThermalPlastic2D, ThermalPlasticParameters,
    ThermalPlasticState, conductive_power, exponential_slip)


def example():
    n = 3; x, y = np.meshgrid(np.linspace(0,1,n+1),np.linspace(0,1,n+1),indexing='ij')
    u = np.stack((.15*y, np.zeros_like(x)), axis=-1).ravel()
    model = ThermalPlastic2D(n)
    model.values = u[model.fixed].copy()
    state = model.initial(T=lambda x,y:1+.1*np.cos(np.pi*x)*np.cos(np.pi*y),u=u,equilibrate=True)
    rng=np.random.default_rng(835)
    state.v=rng.normal(size=len(u))*.03;state.v[model.fixed]=0
    return model,state


def test_mixed_thermal_derivatives():
    model,s=example();rate=model.rate(s); m=model.mechanics;p=m.p;F=m.evaluate(s.u)['F'];eps=1e-5
    args=(s.Fp,m.c,m.eta)
    plus=constitutive(F,*args,m.T+eps,p);minus=constitutive(F,*args,m.T-eps,p)
    np.testing.assert_allclose((plus['P']-minus['P'])/(2*eps),rate['PT'],atol=1e-10)
    plus=constitutive(F,exponential_slip(rate['Lp'],eps)@s.Fp,m.c,m.eta,m.T,p)
    minus=constitutive(F,exponential_slip(rate['Lp'],-eps)@s.Fp,m.c,m.eta,m.T,p)
    np.testing.assert_allclose((plus['entropy']-minus['entropy'])/(2*eps),-rate['chi'],atol=1e-10)


def test_semidiscrete_energy_and_entropy_chain_rule():
    model,s=example();r=model.rate(s);eps=1e-5
    def perturb(sign):
        dt=sign*eps
        return ThermalPlasticState(s.u+dt*r['u'],s.v+dt*r['v'],s.T+dt*r['T'],
            exponential_slip(r['Lp'],dt)@s.Fp,s.gamma+dt*r['gamma'])
    a=model.rate(perturb(1))['diagnostics'];b=model.rate(perturb(-1))['diagnostics']
    assert abs((a['total_energy']-b['total_energy'])/(2*eps))<2e-9
    assert abs((a['entropy']-b['entropy'])/(2*eps)-r['diagnostics']['entropy_production'])<2e-9
    assert r['diagnostics']['plastic_power']>0


def test_conduction_conservation_and_entropy():
    rng=np.random.default_rng(736);T=1+rng.random((8,8));power,entropy=conductive_power(T,.4)
    assert abs(power.sum())<1e-14
    assert abs(np.sum(power/T)-entropy)<1e-14
    assert entropy>0
    power,entropy=conductive_power(np.ones((3,3)),.4)
    assert not np.any(power) and entropy==0


def test_exact_plastic_exponentials():
    L=np.zeros((4,3,3));L[:,0,1]=[.3,.3,0,.7];L[:,1,0]=[.2,-.2,.4,0]
    actual=exponential_slip(L,.31)
    np.testing.assert_allclose(actual,np.stack([expm(.31*x) for x in L]),atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(actual),1,atol=1e-14)


def test_isolated_step_refinement_and_volume():
    model,initial=example();errors=[]
    e0=model.rate(initial)['diagnostics']['total_energy']
    for dt in (.004,.002,.001):
        state=initial
        for _ in range(round(.02/dt)):state,d=model.step(state,dt)
        errors.append(abs(d['total_energy']-e0))
        assert d['plastic_volume_error']<1e-12
        np.testing.assert_array_equal(state.v[model.fixed],0)
        np.testing.assert_array_equal(state.u[model.fixed],model.values)
    assert errors[0]>3.5*errors[1] and errors[1]>3.5*errors[2]


def test_heat_mode_implicit_midpoint():
    n=4;model=ThermalPlastic2D(n,mechanical=FiniteParameters(alpha_T=0),
        transport=ThermalPlasticParameters(yield_stress=(100,100)),fixed=np.arange(2*(n+1)**2))
    s=model.initial(T=lambda x,y:1+.1*np.cos(np.pi*x)*np.cos(np.pi*y))
    dt=.01;new,_=model.step(s,dt)
    rate=8*.4/(10*model.mechanics.mesh.h**2)*np.sin(np.pi/(2*n))**2
    expected=1+(s.T-1)*(1-dt*rate/2)/(1+dt*rate/2)
    np.testing.assert_allclose(new.T,expected,atol=1e-13)


def test_restart_continuation(tmp_path):
    model,s=example();s,_=model.step(s,.001);path=tmp_path/'thermal.npz'
    model.save(path,s,.001,1);restored,t,clock=ThermalPlastic2D.load(path)
    assert clock=={'time':.001,'step':1}
    a,ad=model.step(s,.001);b,bd=restored.step(t,.001)
    for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert ad==bd


def test_nonconverged_midpoint_rejected():
    model,s=example()
    with pytest.raises(RuntimeError,match='did not converge'):model.step(s,.01,max_iterations=1)


def test_invalid_stage_rejected():
    model,s=example();s.T[0]=-1
    with pytest.raises(ValueError):model.step(s,.001)
    model=ThermalPlastic2D(2,mechanical=FiniteParameters(alpha_T=0),
        transport=ThermalPlasticParameters(yield_stress=(100,100)),fixed=np.arange(18))
    s=model.initial(T=np.array([2.,.1,.1,.1]));before=s.T.copy()
    with pytest.raises(ValueError,match='temperature'):model.step(s,10.)
    np.testing.assert_array_equal(s.T,before)


def test_invalid_material_rejected():
    with pytest.raises(ValueError):ThermalPlasticParameters(drag=(0,1))
    with pytest.raises(ValueError):ThermalPlasticParameters(conductivity=-1)
