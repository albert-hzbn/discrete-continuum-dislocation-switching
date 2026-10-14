"""Independent energy, reciprocal coupling, conservation and restart checks."""
import numpy as np
import pytest
from bmvr.continuum3d import Cube,CoupledCrystal
from bmvr.coupled_transport import CoupledTransport,CoupledGammaCache
from bmvr.transport_metric import PlanarTransport


def example():
    g=Cube(12,9);x,y,z=g.coordinates();q=2*np.pi/12
    mean=np.array([[.01,0,.06],[0,-.005,0],[.06,0,0]])
    model=CoupledTransport(g,burgers=(.2,.1,0),mean_strain=mean)
    v=.5*np.cos(q*x)*np.sin(q*y)
    c=.2+.03*np.cos(q*(x+z))
    gamma=.02+.1*np.sin(q*(x+y))+.04*np.cos(2*q*z)
    return model,(v,c,gamma)


def test_block_kernel_matches_independent_tensor_energy_and_reciprocal_forces():
    model,fields=example();v,c,gamma=fields;g=model.grid
    normal=np.array([0.,0.,1.]);beta=np.outer(model.burgers,normal)[...,None,None,None]*gamma
    crystal=CoupledCrystal(g,parameters=model.p,mean_strain=model.mean_strain)
    ref=crystal.evaluate(v,c,np.zeros((12,)+g.shape),(beta+beta.swapaxes(0,1))/2)
    actual=model.evaluate(*fields);tensor=model.tensor_readback(*fields)
    assert actual['energy']==pytest.approx(ref['energy']+tensor['core_energy'],abs=1e-11)
    for a,b in zip(actual['forces'][:2],ref['forces'][:2]):assert np.max(abs(a-b))<1e-12
    assert np.max(abs(actual['trace_stress']-np.einsum('ii...->...',ref['stress'])))<1e-12
    assert np.max(abs(model.kernel[0,1]-model.kernel[1,0]))<1e-14
    assert tensor['divergence_relative']<1e-12
    assert tensor['plastic_trace']==0
    p=model.p;d=p.phase_misfit*(v+1)/2+p.solute_expansion*(c-p.reference_population)
    transport=PlanarTransport(g,burgers=model.burgers,mean_strain=model.mean_strain,
        background=np.eye(3)[...,None,None,None]*d)
    independent=transport.evaluate(gamma)
    assert np.max(abs(actual['gamma_conjugate']-independent['conjugate']))<1e-12
    assert np.max(abs(actual['rates'][2]-independent['full_rate']))<1e-12


@pytest.mark.parametrize('index',range(3))
def test_each_field_force_is_the_directional_energy_derivative(index):
    model,fields=example();g=model.grid;x,y,z=g.coordinates()
    direction=np.random.default_rng(33+index).normal(size=g.shape)
    delta=1e-6;plus=list(fields);minus=list(fields)
    plus[index]=fields[index]+delta*direction;minus[index]=fields[index]-delta*direction
    derivative=(model.evaluate(*plus)['energy']-model.evaluate(*minus)['energy'])/(2*delta)
    exact=g.integrate(model.evaluate(*fields)['forces'][index]*direction)
    assert derivative==pytest.approx(exact,rel=2e-7,abs=2e-7)


def test_rates_satisfy_dissipation_identity_and_inventory():
    model,fields=example();state=model.evaluate(*fields);g=model.grid
    derivative=sum(g.integrate(f*r) for f,r in zip(state['forces'],state['rates']))
    assert derivative==pytest.approx(-state['dissipation'],rel=1e-12)
    assert abs(g.integrate(state['rates'][1]))<1e-12
    for _ in range(4):
        new,next_state,_,_=model.step(fields,.01,current=state)
        assert next_state['energy']<state['energy']
        assert g.integrate(new[1])==pytest.approx(g.integrate(fields[1]),abs=1e-12)
        fields,state=new,next_state


def test_cache_refreshes_background_and_matches_full_at_changed_coupled_state():
    model,(v,c,gamma)=example();cache=CoupledGammaCache(model,gamma)
    a,state,event=cache.enrich(v,c,np.array([]))
    assert event['energy_jump']==0 and not event['fallback_required']
    assert state['projection']['relative_residual']<1e-10
    a[0]=.02;v=v+.03;c=c+.001*np.sin(v)
    state=cache.state(v,c,a);direct=model.evaluate(v,c,state['gamma'])
    assert state['energy']==pytest.approx(direct['energy'],abs=1e-11)
    for key in ('trace_stress','gamma_conjugate','gamma_mobility'):
        assert np.max(abs(state[key]-direct[key]))<1e-12
    stale=model.evaluate(v-.03,c,gamma)
    assert np.linalg.norm(state['trace_stress']-stale['trace_stress'])>1e-3
    nv,nc,a,following,_,_=cache.step(v,c,a,.01,current=state)
    assert following['energy']<state['energy']
    assert model.grid.integrate(nc)==pytest.approx(model.grid.integrate(c),abs=1e-12)


def test_exhausted_cap_requests_exact_field_fallback():
    model,(v,c,gamma)=example();cache=CoupledGammaCache(model,gamma)
    a,state,event=cache.enrich(v,c,np.array([]),max_modes=0)
    assert event['fallback_required'] and np.array_equal(state['gamma'],gamma)
    result,_,_,_=model.step((v,c,state['gamma']),.01,current=state)
    reference,_,_,_=model.step((v,c,gamma),.01)
    assert all(np.array_equal(x,y) for x,y in zip(result,reference))


@pytest.mark.parametrize('mode',['affine','field'])
def test_checkpoint_preserves_next_coupled_step_and_controller(tmp_path,mode):
    model,(v,c,gamma)=example();cache=CoupledGammaCache(model,gamma)
    a,state,_=cache.enrich(v,c,np.array([]))
    v,c,a,state,used,_=cache.step(v,c,a,.01,current=state);gamma=state['gamma']
    if mode=='field':(v,c,gamma),state,_,_=model.step((v,c,gamma),.01)
    path=tmp_path/'coupled.npz'
    cache.save(path,v,c,a,gamma,mode,used,1,budget=.04,max_modes=3)
    restored,rv,rc,ra,rg,clock=CoupledGammaCache.load(path)
    assert clock=={'mode':mode,'time':used,'step':1,'budget':.04,'max_modes':3}
    if mode=='field':
        expected,es,_,_=model.step((v,c,gamma),.01)
        actual,rs,_,_=restored.model.step((rv,rc,rg),.01)
    else:
        v,c,a,es,_,_=cache.step(v,c,a,.01)
        rv,rc,ra,rs,_,_=restored.step(rv,rc,ra,.01)
        expected=(v,c,a);actual=(rv,rc,ra)
    assert all(np.array_equal(x,y) for x,y in zip(expected,actual))
    assert es['energy']==rs['energy']
