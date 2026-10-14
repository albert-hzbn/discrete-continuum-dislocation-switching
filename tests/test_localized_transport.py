"""Conservative localized trial spaces, exact metrics and coupled continuation."""
import numpy as np
import pytest
from bmvr.continuum3d import Cube
from bmvr.coupled_transport import CoupledTransport
from bmvr.localized_transport import DiffusionMetric,LocalizedTransport


def example(caps=(32,32,32)):
    grid=Cube(12,7);x,y,z=grid.coordinates();q=2*np.pi/12
    fields=(.3*np.cos(q*x)*np.sin(q*y),.2+.02*np.cos(q*(y+z)),.03*np.sin(q*(x+y))+.04*np.cos(q*z))
    return LocalizedTransport(CoupledTransport(grid,burgers=(.2,.1,0)),fields,caps=caps)


def test_diffusion_inverse_uses_discrete_symbol_and_rejects_mass_change():
    grid=Cube(12,9);metric=DiffusionMetric(grid,.7)
    potential=np.random.default_rng(34).normal(size=grid.shape)
    rate=-.7*grid.lap(potential)
    assert np.max(abs(metric.inverse(rate)-(potential-potential.mean())))<1e-13
    assert grid.integrate(rate*metric.inverse(rate))==pytest.approx(1.4*grid.gradient_energy(potential),rel=1e-12)
    with pytest.raises(ValueError,match='zero-mean'):metric.inverse(rate+.01)


def test_population_candidates_are_local_conservative_and_sum_to_residual():
    chart=example();state=chart.state();candidates=chart.candidates(1,state);g=chart.model.grid
    assert np.max(abs(candidates.sum(axis=0)-state['rates'][1]))<1e-14
    assert all(abs(g.integrate(c))<1e-12 for c in candidates)
    for patch,c in enumerate(candidates):
        support=chart.masks[patch].copy()
        for axis in range(3):support|=np.roll(chart.masks[patch],1,axis)
        assert np.count_nonzero(c[~support])==0
    masked=chart.masks*state['rates'][1]
    assert max(abs(g.integrate(c)) for c in masked)>1e-3


def test_activation_is_state_preserving_and_all_metrics_dissipate():
    chart=example();initial=chart.state();state,events=chart.activate();g=chart.model.grid
    assert len(events)>0 and not any(chart.full)
    assert all(np.array_equal(a,b) for a,b in zip(initial['fields'],state['fields']))
    assert abs(state['energy']-initial['energy'])<1e-12
    for force,p,budget in zip(state['forces'],state['projections'],chart.budgets):
        assert p['relative_residual']<=budget
        assert g.integrate(force*p['rate'])==pytest.approx(-p['projected_norm_squared'],rel=1e-10,abs=1e-12)
        assert abs(p['orthogonality'])<1e-10
        assert p['pythagorean_error']<1e-10


def test_local_modes_and_responses_recover_direct_coupled_energy_and_forces():
    chart=example();state,_=chart.activate();state,_,_=chart.step(.01,current=state)
    direct=chart.model.evaluate(*state['fields'])
    assert state['energy']==pytest.approx(direct['energy'],abs=1e-11)
    for a,b in zip(state['forces'],direct['forces']):assert np.max(abs(a-b))<1e-12
    for block in (0,2):
        for mode,patch in zip(chart.modes[block],chart.patch_history[block]):assert np.count_nonzero(mode[~chart.masks[patch]])==0


def test_full_cap_fallback_preserves_reference_evolution():
    chart=example(caps=(0,0,0));before=chart.state();state,events=chart.activate()
    assert chart.full==[True,True,True] and len(events)==3
    actual,used,_=chart.step(.01,current=state)
    expected,reference,dt,_=chart.model.step(before['fields'],.01)
    assert used==dt
    for a,b in zip(actual['fields'],expected):assert np.max(abs(a-b))<1e-14
    assert actual['energy']==pytest.approx(reference['energy'],abs=1e-12)


def test_evolution_preserves_population_energy_and_requires_rate_audit():
    chart=example();initial=chart.state();g=chart.model.grid
    with pytest.raises(ValueError,match='Activate'):chart.step(.01)
    for _ in range(3):
        state,_=chart.activate();following,_,_=chart.step(.01,current=state)
        assert following['energy']<state['energy']
        assert g.integrate(following['fields'][1])==pytest.approx(g.integrate(initial['fields'][1]),abs=1e-12)


def test_reset_rebases_without_losing_physical_fields():
    chart=example();state,_=chart.activate();before,_,_=chart.step(.01,current=state)
    for b in range(3):chart.reset(b)
    after=chart.state()
    assert all(len(m)==0 for m in chart.modes)
    for a,b in zip(before['fields'],after['fields']):assert np.array_equal(a,b)
    assert after['energy']==pytest.approx(before['energy'],abs=1e-12)
    _,events=chart.activate();assert events


def test_reset_at_cap_preserves_state_and_has_bounded_fallback(tmp_path):
    chart=example(caps=(8,8,8));chart.reset_at_cap=True;events=[]
    for _ in range(12):
        state,added=chart.activate();events.extend(added);chart.step(.05,current=state)
    resets=[e for e in events if e['action']=='reset']
    assert resets and all(e['state_jump_max']==0 for e in resets)
    assert chart.full==[False,True,False]
    assert any(e['action']=='promote' and e['block']=='population' for e in events)
    path=tmp_path/'reset.npz';chart.save(path);restored=LocalizedTransport.load(path)
    assert restored.reset_at_cap
    a,ea=chart.activate();b,eb=restored.activate();assert ea==eb
    a,_,_=chart.step(.05,current=a);b,_,_=restored.step(.05,current=b)
    for x,y in zip(a['fields'],b['fields']):assert np.array_equal(x,y)


@pytest.mark.parametrize('caps',[(32,32,32),(0,0,0),(0,32,0)])
def test_restart_preserves_activation_and_evolution(tmp_path,caps):
    chart=example(caps=caps);state,_=chart.activate();chart.step(.01,current=state)
    path=tmp_path/'localized.npz';chart.save(path);restored=LocalizedTransport.load(path)
    assert restored.budgets==chart.budgets and restored.caps==chart.caps
    assert restored.time==chart.time and restored.steps==chart.steps
    a,ea=chart.activate();b,eb=restored.activate();assert ea==eb
    a,dt,_=chart.step(.01,current=a);b,rdt,_=restored.step(.01,current=b)
    assert dt==rdt
    for x,y in zip(a['fields'],b['fields']):assert np.array_equal(x,y)
    assert a['energy']==b['energy']
