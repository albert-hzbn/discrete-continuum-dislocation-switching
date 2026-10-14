"""Independent derivatives, continuum transport and refusal of lossy chart changes."""
from dataclasses import replace
import numpy as np
import pytest
from bmvr.continuum3d import Cube
from bmvr.loops3d import circular_loop
from bmvr.representation3d import NyeTransfer


def example(n=9):
    grid = Cube(12, n)
    x,y,z = grid.coordinates()
    background = np.eye(3)[...,None,None,None]*(.03*np.cos(2*np.pi*x/12)*np.sin(2*np.pi*y/12))
    model = NyeTransfer(grid, background=background, glide_normal=np.array([0.,0.,1.]))
    loop = circular_loop(radius=2., count=6, core_tension=0., glide_normal=np.array([0.,0.,1.]))
    return model, loop


def test_promotion_preserves_source_mean_and_shared_energy():
    model,loop=example()
    source=model.mechanics.source([loop])
    correction=np.diag([.01,-.01,0.])
    beta=model.promote([loop],correction)
    state=model.evaluate(beta)
    assert np.max(np.abs(np.fft.fftn(state['alpha'],axes=(-3,-2,-1))-source['alpha_hat']))<1e-12
    expected=-np.outer(loop.burgers,loop.area)/model.grid.volume+correction
    assert np.max(np.abs(state['mean_beta']-expected))<1e-12
    assert model.line_state([loop],correction)['energy']==pytest.approx(state['energy'],abs=1e-12)
    with pytest.raises(ValueError,match='Separate line tension'):
        model.promote([replace(loop,core_tension=.01)])


def test_shared_energy_directional_derivative_and_gauge():
    model,loop=example()
    beta=model.promote([loop]);state=model.evaluate(beta)
    direction=np.random.default_rng(30).normal(size=beta.shape)
    eps=1e-6
    numerical=(model.evaluate(beta+eps*direction)['energy']-model.evaluate(beta-eps*direction)['energy'])/(2*eps)
    assert numerical==pytest.approx(-model.grid.integrate(state['H']*direction),rel=1e-7,abs=1e-8)
    x,y,z=model.grid.coordinates()
    w=np.array([np.sin(2*np.pi*x/12),np.cos(2*np.pi*z/12),np.sin(2*np.pi*y/12)])*.01
    gradient=np.array([[model.mechanics.elastic.derivative(w[i],j) for j in range(3)] for i in range(3)])
    shifted=model.evaluate(beta+gradient)
    assert shifted['energy']==pytest.approx(state['energy'],abs=1e-12)
    assert np.max(np.abs(shifted['stress']-state['stress']))<1e-12


def test_density_core_pk_force_is_same_energy_derivative():
    model,loop=example(11)
    state=model.line_state([loop])
    direction=np.random.default_rng(31).normal(size=loop.nodes.shape);direction[:,2]=0
    eps=1e-5
    plus=model.line_state([loop.moved(loop.nodes+eps*direction)])['energy']
    minus=model.line_state([loop.moved(loop.nodes-eps*direction)])['energy']
    assert (plus-minus)/(2*eps)==pytest.approx(-np.sum(state['line_forces'][0]*direction),rel=1e-5,abs=1e-8)


def test_continuum_dissipation_and_burgers_transport_identity():
    model,loop=example()
    beta=model.promote([loop]);state=model.evaluate(beta)
    assert model.grid.integrate(state['H']*state['rate'])==pytest.approx(state['dissipation'],rel=1e-12)
    assert state['dissipation']>0
    first=beta.copy()
    for _ in range(4):
        old=model.evaluate(beta)['energy']
        beta,state,used,trials=model.field_step(beta,.02)
        assert state['energy']<=old+1e-12
        assert state['divergence_relative']<1e-12
        assert np.max(np.abs(state['alpha'].mean(axis=(2,3,4))))<1e-12
        assert abs(np.trace(state['mean_beta']))<1e-12
    assert np.linalg.norm(beta-first)>1e-6
    zero=model.evaluate(np.zeros_like(beta))
    assert np.max(np.abs(zero['rate']))==0


def test_inverse_recovers_changed_nodes_without_current_coordinates():
    model,hint=example()
    nodes=hint.nodes.copy()
    nodes[:,0]+=.13;nodes[:,1]*=1.04
    actual=hint.moved(nodes)
    correction=np.array([[0,0,.002],[0,0,0],[0,0,0]])
    beta=model.promote([actual],correction)
    fit=model.recover(beta,[hint],source_tolerance=1e-8,stress_tolerance=1e-8,force_tolerance=1e-8,energy_tolerance=1e-8)
    assert fit['accepted'],fit['errors']
    assert np.max(np.abs(fit['loops'][0].nodes-actual.nodes))<1e-6
    assert fit['errors']['mean_beta_absolute']<1e-12
    assert np.max(np.abs(fit['mean_correction']-correction))<1e-10


def test_incompatible_burgers_chart_refuses_and_preserves_field():
    model,hint=example()
    other=replace(hint,burgers=np.array([0.,.2,0.]))
    beta=model.promote([hint,other])
    fit=model.recover(beta,[hint],max_evaluations=10)
    assert not fit['accepted']
    assert fit['errors']['alpha_relative']>.1
    assert np.array_equal(fit['beta'],beta)
    assert fit['loops'] is None


def test_field_restart_preserves_independent_continuum_evolution(tmp_path):
    model,loop=example()
    beta,_,used,_=model.field_step(model.promote([loop]),.02)
    path=tmp_path/'density.npz'
    model.save(path,beta,[loop],used,1)
    restored,field,hints,clock=NyeTransfer.load(path)
    assert clock=={'time':used,'step':1}
    assert np.array_equal(hints[0].nodes,loop.nodes)
    a=model.field_step(beta,.02);b=restored.field_step(field,.02)
    assert np.array_equal(a[0],b[0])
    assert a[1]['energy']==b[1]['energy']
