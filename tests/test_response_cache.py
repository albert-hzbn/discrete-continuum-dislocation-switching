"""Exact scalar reduction and affine caching against independent tensor mechanics."""
import numpy as np
import pytest
from bmvr.continuum3d import Cube
from bmvr.transport_metric import PlanarTransport
from bmvr.response_cache import ScalarTransportOperator,AffineTransportCache,small_projection
from bmvr.transport_metric import weighted_projection
from bmvr.loops3d import circular_loop


def example():
    g=Cube(12,9);x,y,z=g.coordinates()
    model=PlanarTransport(g,background=np.eye(3)[...,None,None,None]*.03*np.cos(2*np.pi*x/12)*np.sin(2*np.pi*y/12))
    gamma=.02+.1*np.sin(2*np.pi*(x+y)/12)+.04*np.cos(4*np.pi*z/12)
    return model,gamma


def test_scalar_multiplier_recovers_tensor_energy_force_and_transport():
    model,gamma=example();op=ScalarTransportOperator(model)
    direct=model.evaluate(gamma);fast=op.evaluate(gamma)
    assert fast['energy']==pytest.approx(direct['energy'],abs=1e-12)
    assert np.max(np.abs(fast['conjugate']-direct['conjugate']))<1e-12
    assert np.max(np.abs(fast['full_rate']-direct['full_rate']))<1e-12
    assert op.symbol[0,0,0]==pytest.approx(.04)
    assert np.max(np.abs(op.symbol[0,0,1:]))==0


def test_optimized_full_step_matches_original_tensor_transport():
    model,gamma=example();op=ScalarTransportOperator(model);current=op.evaluate(gamma)
    fast,dt,trials=op.step(current,.02)
    slow,state,used,_=model.field_step(gamma,.02)
    assert dt==used
    assert np.max(np.abs(fast['gamma']-slow))<1e-12
    assert fast['energy']==pytest.approx(state['energy'],abs=1e-12)
    assert np.max(np.abs(fast['conjugate']-state['conjugate']))<1e-12


def test_affine_responses_and_energy_gradient_equal_fresh_solves():
    model,gamma=example();op=ScalarTransportOperator(model);cache=AffineTransportCache(op,gamma)
    x,y,z=model.grid.coordinates()
    for direction in (np.cos(2*np.pi*x/12),np.sin(2*np.pi*(x-z)/12)):cache.add(direction)
    c=np.array([.03,-.02]);state=cache.state(c);fresh=op.evaluate(state['gamma'])
    for key in ('conjugate','a','full_rate'):assert np.max(np.abs(state[key]-fresh[key]))<1e-12
    assert state['energy']==pytest.approx(fresh['energy'],abs=1e-12)
    derivative=-cache.linear+cache.stiffness@c
    for i in range(2):
        change=np.eye(2)[i]*1e-6
        numerical=(cache.state(c+change)['energy']-cache.state(c-change)['energy'])/2e-6
        assert numerical==pytest.approx(derivative[i],rel=1e-7,abs=1e-9)


def test_enrichment_and_advance_need_no_repeated_mechanical_solve():
    model,_=example();loop=circular_loop(radius=2,count=6,center=(.2,.1,0),core_tension=0)
    gamma=model.gamma_from_loops([loop]);op=ScalarTransportOperator(model);cache=AffineTransportCache(op,gamma)
    c,state,event=cache.enrich(np.array([]),budget=.05,max_modes=3)
    assert event['added_modes']==1 and event['energy_jump']==0 and not event['fallback_required']
    assert state['projection']['relative_residual']<1e-10
    calls=op.response_calls
    c,following,dt,_=cache.step(c,state,.02)
    assert op.response_calls==calls
    assert following['energy']<=state['energy']
    direct=model.evaluate(following['gamma'])
    assert following['energy']==pytest.approx(direct['energy'],abs=1e-12)


def test_zero_mode_cap_falls_back_without_state_change():
    model,gamma=example();op=ScalarTransportOperator(model);cache=AffineTransportCache(op,gamma)
    c,state,event=cache.enrich(np.array([]),max_modes=0)
    assert event['fallback_required']
    assert np.array_equal(state['gamma'],gamma)
    a,_,_=op.step(state,.02);b,_,_=op.step(op.evaluate(gamma),.02)
    assert np.array_equal(a['gamma'],b['gamma'])


def test_small_gram_projection_matches_weighted_svd_with_null_site_fallback():
    rng=np.random.default_rng(32);modes=rng.normal(size=(3,21));K=np.exp(rng.uniform(-5,5,21));h=rng.normal(size=21)
    a=small_projection(modes,K*h,K,.2,h);b=weighted_projection(modes,K*h,K,.2)
    assert np.allclose(a['coordinates'],b['coordinates'],rtol=1e-10,atol=1e-12)
    assert a['relative_residual']==pytest.approx(b['relative_residual'],rel=1e-12)
    K[0]=0
    a=small_projection(modes,K*h,K,.2,h);b=weighted_projection(modes,K*h,K,.2)
    assert np.array_equal(a['rate'],b['rate'])


@pytest.mark.parametrize('mode',['affine','field'])
def test_cached_restart_preserves_next_evolution(tmp_path,mode):
    model,gamma=example();op=ScalarTransportOperator(model);cache=AffineTransportCache(op,gamma)
    c,state,_=cache.enrich(np.array([]))
    c,state,used,_=cache.step(c,state,.02)
    if mode=='field':
        for _ in range(3):state,_,_=op.step(state,.02)
    path=tmp_path/'cache.npz';cache.save(path,c,state,mode,.02,1)
    restored,rc,rs,clock=AffineTransportCache.load(path)
    assert clock['mode']==mode
    if mode=='field':
        a,_,_=op.step(state,.02);b,_,_=restored.operator.step(rs,.02)
    else:
        state=cache.state(c);rs=restored.state(rc)
        _,a,_,_=cache.step(c,state,.02);_,b,_,_=restored.step(rc,rs,.02)
    assert np.array_equal(a['gamma'],b['gamma'])
    assert a['energy']==b['energy']
