"""Common-mobility identities and adaptive kinetic residual control."""
from dataclasses import replace
import numpy as np
import pytest
from bmvr.continuum3d import Cube
from bmvr.loops3d import circular_loop
from bmvr.transport_metric import PlanarTransport,weighted_projection
from bmvr.transport_checkpoint import save_transport,load_transport


def example():
    grid=Cube(12,9);x,y,z=grid.coordinates()
    background=np.eye(3)[...,None,None,None]*(.03*np.cos(2*np.pi*x/12)*np.sin(2*np.pi*y/12))
    model=PlanarTransport(grid,background=background)
    loop=circular_loop(radius=2,count=6,center=(.2,.1,0),core_tension=0,glide_normal=np.array([0.,0.,1.]))
    return model,loop


def test_scalar_sheet_gauge_preserves_source_stress_and_mean():
    m,loop=example();gamma=m.gamma_from_loops([loop])
    scalar=m.evaluate(gamma);coulomb=m.base.evaluate(m.base.promote([loop]))
    for key in ('alpha','stress','mean_beta'):
        assert np.max(np.abs(scalar[key]-coulomb[key]))<1e-12
    assert scalar['energy']==pytest.approx(coulomb['energy'],abs=1e-12)
    assert np.max(np.abs(m.embed(scalar['full_rate'])-scalar['rate']))<1e-12


def test_analytic_chart_columns_match_independent_geometry_differences():
    m,loop=example();B=m.tangent([loop]);eps=1e-5
    for node,axis in ((0,0),(1,1),(4,0)):
        d=np.zeros_like(loop.nodes);d[node,axis]=eps
        fd=(m.gamma_from_loops([loop.moved(loop.nodes+d)])-m.gamma_from_loops([loop.moved(loop.nodes-d)]))/(2*eps)
        assert np.linalg.norm(fd-B[2*node+axis])/np.linalg.norm(fd)<1e-7


def test_common_metric_projection_energy_and_residual_identities():
    m,loop=example();state=m.chart_state([loop]);p=state['projection']
    assert abs(p['orthogonality'])<1e-11*p['full_norm_squared']
    assert p['pythagorean_error']<1e-11*p['full_norm_squared']
    assert state['projected_energy_release']==pytest.approx(p['projected_norm_squared'],rel=1e-10)
    assert p['projected_norm_squared']>=0
    assert p['discarded_norm_squared']>0


def test_pinned_sites_are_constraints_not_free_pseudoinverse_directions():
    B=np.array([[1.,1.,0.],[0.,1.,1.]])
    full=np.array([0.,2.,3.]);K=np.array([0.,1.,1.])
    p=weighted_projection(B,full,K,1.)
    assert p['pinned_rate_max']<1e-12
    assert p['constraint_rank']==1
    assert np.allclose(p['rate'],[0.,2.5,2.5])
    with pytest.raises(ValueError,match='zero mobility'):
        weighted_projection(B,np.array([1.,2.,3.]),K,1.)


def test_residual_enrichment_changes_no_state_and_recovers_full_instantaneous_rate():
    m,loop=example();before=m.chart_state([loop])
    details,amps,after,event=m.enrich([loop],[],np.array([]),budget=1e-7,max_details=1)
    assert event['added_details']==1
    assert not event['fallback_required']
    assert event['energy_jump']==0
    assert np.array_equal(before['gamma'],after['gamma'])
    assert after['projection']['relative_residual']<1e-7
    assert np.max(np.abs(after['projection']['rate']-after['full_rate']))<1e-10


def test_nodal_drag_is_unused_and_full_mobility_sets_clock():
    m,loop=example();a=m.chart_state([loop])['projection']['rate']
    b=m.chart_state([replace(loop,mobility=17.)])['projection']['rate']
    assert np.array_equal(a,b)
    m.base.field_mobility*=2
    c=m.chart_state([loop])['projection']['rate']
    assert np.allclose(c,2*a,rtol=1e-10,atol=1e-12)


def test_exhausted_detail_budget_requests_full_field_and_enriched_step_dissipates():
    m,loop=example()
    _,_,_,refused=m.enrich([loop],[],np.array([]),budget=1e-7,max_details=0)
    assert refused['fallback_required']
    details,amps,state,_=m.enrich([loop],[],np.array([]),budget=.05,max_details=1)
    loops,amps,following,used,trials=m.chart_step([loop],details,amps,.01,current=state)
    assert following['energy']<state['energy']
    assert abs(np.trace(following['mean_beta']))<1e-12
    assert following['divergence_relative']<1e-12
    assert np.linalg.norm(following['gamma']-state['gamma'])>0


@pytest.mark.parametrize('mode',['chart','field'])
def test_controller_restart_preserves_details_history_and_next_step(tmp_path,mode):
    m,loop=example();loops=[loop]
    details,amps,state,_=m.enrich(loops,[],np.array([]))
    loops,amps,state,used,_=m.chart_step(loops,details,amps,.01,current=state)
    gamma=state['gamma']
    path=tmp_path/'transport.npz'
    save_transport(path,m,loops,details,amps,gamma,mode,used,1,budget=.05,max_details=2)
    restored,rl,rd,ra,rg,clock=load_transport(path)
    assert clock=={'mode':mode,'time':used,'step':1,'history':0.,'budget':.05,'max_details':2}
    assert np.array_equal(gamma,rg)
    if mode=='field':
        a=m.field_step(gamma,.01);b=restored.field_step(rg,.01)
        assert np.array_equal(a[0],b[0])
    else:
        a=m.chart_step(loops,details,amps,.01);b=restored.chart_step(rl,rd,ra,.01)
        assert np.array_equal(a[1],b[1])
        assert np.array_equal(a[2]['gamma'],b[2]['gamma'])
        assert a[2]['energy']==b[2]['energy']
