"""Independent graph charge, energy derivative, interface reaction and kinetic tests."""

from dataclasses import replace
import numpy as np
import pytest
from bmvr.continuum3d import Cube,CoupledCrystal,Parameters3D
from bmvr.loops3d import LoopMechanics,circular_loop
from bmvr.graphs3d import DislocationGraph,GraphMechanics,encounter_graphs,graph_state,graph_step,frozen_reaction_family,ensemble_trajectory,tangent_projector,save_graph_checkpoint,load_graph_checkpoint,cross_reflection_graph


def test_unbalanced_junction_is_rejected():
    graph=encounter_graphs()['transmitted']
    burgers=graph.burgers.copy();burgers[-1,0]+=.001
    with pytest.raises(ValueError,match='balance'):
        replace(graph,burgers=burgers)


def test_simple_cycle_graph_matches_loop_source_energy_and_forces():
    grid=Cube(12,13);loop=circular_loop(radius=2,count=8)
    edges=np.array([[i,(i+1)%8] for i in range(8)])
    graph=DislocationGraph(loop.nodes,edges,np.tile(loop.burgers,(8,1)),core=loop.core)
    a=LoopMechanics(grid).evaluate([loop]);b=GraphMechanics(grid).evaluate(graph)
    assert a['energy']==pytest.approx(b['energy'],rel=1e-12)
    assert np.max(np.abs(a['forces'][0]-b['forces']))<1e-12
    assert b['divergence_relative']<1e-12 and b['curl_relative']<1e-12


@pytest.mark.parametrize('branch',['incident','transmitted','absorbed','reflected'])
def test_each_spatial_branch_has_balanced_charge_and_valid_curl(branch):
    graph=encounter_graphs()[branch]
    source=GraphMechanics(Cube(12,13)).source(graph)
    assert graph.balance_error<1e-12
    assert source['divergence_relative']<1e-12 and source['curl_relative']<1e-12


def test_transmission_leaves_real_interface_residual_and_outgoing_plane():
    theta=35; family=encounter_graphs(theta);graph=family['transmitted']
    residual=graph.burgers[graph.kinds==1]
    assert residual.shape==(1,3)
    assert np.max(np.abs(residual[0]-(family['incident'].burgers[0]-graph.burgers[0])))<1e-12
    assert np.allclose(graph.nodes[graph.edges[graph.kinds==1],0],0)
    normal=np.array([np.sin(np.deg2rad(theta)),0,np.cos(np.deg2rad(theta))])
    assert abs(graph.nodes[1]@normal)<1e-12 and abs(graph.burgers[0]@normal)<1e-12
    mechanics=GraphMechanics(Cube(12,13))
    assert np.linalg.norm(mechanics.source(graph)['alpha_hat'])>0
    # A residual cannot simply be omitted: its endpoints would lose Burgers balance.
    with pytest.raises(ValueError,match='balance'):
        DislocationGraph(graph.nodes,graph.edges[:-1],graph.burgers[:-1],core=graph.core)


def test_absorption_stores_charge_and_reflection_keeps_burgers_sign():
    family=encounter_graphs()
    absorbed=family['absorbed']
    assert len(absorbed.nodes)==3 and absorbed.kinds.sum()==1
    assert np.array_equal(absorbed.burgers[0],family['incident'].burgers[0])
    assert np.array_equal(family['reflected'].burgers,family['incident'].burgers)
    assert family['reflected'].nodes[1,0]<family['incident'].nodes[1,0]


def test_zero_misorientation_needs_no_residual_edge():
    graph=encounter_graphs(0)['transmitted']
    assert len(graph.edges)==4 and not graph.kinds.any() and graph.balance_error==0


def test_spatial_glide_branches_preserve_zero_mean_plastic_volume():
    model=GraphMechanics(Cube(12,13))
    for graph in encounter_graphs().values():
        beta=model.source(graph)['distortion']
        assert abs(np.mean(np.einsum('ii...->...',beta)))<1e-12


def test_junction_mobility_is_intersection_of_planes():
    normals=np.array([[1,0,0],[0,0,1],[1,0,1]])
    p=tangent_projector(normals)
    assert np.allclose(p,np.diag([0,1,0]))
    assert np.allclose(p@p,p)


def test_graph_with_interface_energy_has_correct_nodal_derivative():
    grid=Cube(12,13);model=GraphMechanics(grid,quadrature_order=10)
    graph=encounter_graphs()['transmitted']
    x,y,z=grid.coordinates();phase=.7*np.sin(2*np.pi*x/12)+.1*np.cos(2*np.pi*(y+z)/12)
    direction=np.random.default_rng(56).normal(size=graph.nodes.shape)
    mean=np.array([[0,0,-.07],[0,0,0],[-.07,0,0]])
    result=model.evaluate(graph,mean,phase,.4,.7)
    eps=1e-5
    a=model.evaluate(graph.moved(graph.nodes+eps*direction),mean,phase,.4,.7)['energy']
    b=model.evaluate(graph.moved(graph.nodes-eps*direction),mean,phase,.4,.7)['energy']
    assert -np.sum(result['forces']*direction)==pytest.approx((a-b)/(2*eps),rel=1e-5,abs=1e-8)


def test_interface_field_adjoint_matches_energy_derivative():
    grid=Cube(12,13);model=GraphMechanics(grid)
    graph=encounter_graphs()['transmitted'];rng=np.random.default_rng(99)
    phase=rng.normal(size=grid.shape);direction=rng.normal(size=grid.shape)
    eps=1e-5
    result=model.evaluate(graph,phase=phase,mobile_barrier=.4,residual_binding=.7)
    a=model.evaluate(graph,phase=phase+eps*direction,mobile_barrier=.4,residual_binding=.7)['energy']
    b=model.evaluate(graph,phase=phase-eps*direction,mobile_barrier=.4,residual_binding=.7)['energy']
    assert grid.integrate(result['field_force']*direction)==pytest.approx((a-b)/(2*eps),rel=1e-5,abs=1e-8)


def test_coupled_graph_interface_dissipation_and_node_constraints():
    grid=Cube(12,13);mechanics=GraphMechanics(grid)
    crystal=CoupledCrystal(grid,Parameters3D(yield_stress=10))
    x,_,_=grid.coordinates();v=np.tanh(np.sin(2*np.pi*x/12)*12/(2*np.pi*np.sqrt(2)))
    fields=(v,np.full(grid.shape,.2),np.zeros((12,)+grid.shape))
    graph=encounter_graphs()['transmitted'];initial=graph.nodes.copy();inventory=grid.integrate(fields[1])
    state=graph_state(crystal,fields,graph,mechanics)
    energy_rate=sum(grid.integrate(f*r) for f,r in zip(state['forces'],state['rates']))-np.sum(state['node_forces']*state['node_rates'])
    assert energy_rate==pytest.approx(-state['dissipation'],rel=1e-11)
    for _ in range(4):
        old=state['energy']
        fields,graph,state,_,_=graph_step(crystal,fields,graph,mechanics,.01,state)
        assert state['energy']<=old+1e-10
        assert abs(grid.integrate(fields[1])-inventory)<1e-10
    assert np.max(np.abs(np.einsum('nij,nj->ni',np.eye(3)-graph.projectors,graph.nodes-initial)))<1e-12


def test_actual_graph_energy_family_has_detailed_balance_and_ensemble_decay():
    grid=Cube(12,13);mechanics=GraphMechanics(grid)
    x,_,_=grid.coordinates();v=np.tanh(np.sin(2*np.pi*x/12)*12/(2*np.pi*np.sqrt(2)))
    energies=np.array([mechanics.evaluate(g,phase=v,mobile_barrier=.2,residual_binding=.4)['energy'] for g in encounter_graphs().values()])
    theta=.05;family=frozen_reaction_family(energies,theta)
    flux=family['stationary'][:,None]*family['rates']
    assert np.max(np.abs(flux-flux.T))<1e-12
    probability,free=ensemble_trajectory(energies,family,np.linspace(0,20,61),theta)
    assert np.max(np.abs(probability.sum(axis=1)-1))<1e-12
    assert probability.min()>=-1e-12 and np.max(np.diff(free))<=1e-10


def test_graph_restart_preserves_random_state_and_continuous_evolution(tmp_path):
    grid=Cube(12,9);mechanics=GraphMechanics(grid,quadrature_order=6)
    crystal=CoupledCrystal(grid,Parameters3D(yield_stress=10))
    x,_,_=grid.coordinates();fields=(.8*np.sin(2*np.pi*x/12),np.full(grid.shape,.2),np.zeros((12,)+grid.shape))
    graph=encounter_graphs()['transmitted'];rng=np.random.default_rng(123)
    _=rng.random(5)
    path=tmp_path/'graph.npz'
    save_graph_checkpoint(path,crystal,fields,graph,mechanics,.1,.01,10,rng,mobile_barrier=.3,residual_binding=.6,
                          reaction_context={'event_done':True,'branch':'transmitted'})
    rc,rf,rg,rm,rr,clock=load_graph_checkpoint(path)
    assert rng.random()==rr.random()
    assert clock['reaction_context']=={'event_done':True,'branch':'transmitted'}
    a=graph_step(crystal,fields,graph,mechanics,.01,mobile_barrier=.3,residual_binding=.6)
    b=graph_step(rc,rf,rg,rm,clock['dt'],mobile_barrier=clock['mobile_barrier'],residual_binding=clock['residual_binding'])
    assert all(np.array_equal(x,y) for x,y in zip(a[0],b[0]))
    assert np.array_equal(a[1].nodes,b[1].nodes) and a[2]['energy']==b[2]['energy']


def test_cross_system_reflection_reduces_to_recoil_and_balances_charge():
    family=encounter_graphs();incident=family['incident']
    same=cross_reflection_graph(incident,0.)
    assert np.array_equal(same.nodes,family['reflected'].nodes) and np.array_equal(same.burgers,family['reflected'].burgers)
    assert np.array_equal(same.projectors,family['reflected'].projectors)
    for theta in (25.,60.,-40.):
        graph=cross_reflection_graph(incident,theta);a=np.deg2rad(theta)
        normal=np.array([np.sin(a),0,np.cos(a)]);midpoint=(graph.nodes[0]+graph.nodes[2])/2
        residual=graph.burgers[graph.kinds==1]
        assert graph.balance_error<1e-12 and residual.shape==(1,3)
        assert np.max(np.abs(residual[0]-(incident.burgers[0]-graph.burgers[0])))<1e-12
        assert abs((graph.nodes[1]-midpoint)@normal)<1e-12 and abs(graph.burgers[0]@normal)<1e-12
        assert graph.nodes[1,0]<0
        source=GraphMechanics(Cube(12,13)).source(graph)
        assert source['divergence_relative']<1e-10 and source['curl_relative']<1e-10


def test_applied_glide_force_on_reflected_system_follows_cos_two_theta():
    """Section 49.2: applied part of the implemented apex force equals tau_in cos(2 theta) along the new glide direction."""
    grid=Cube(12,13);mechanics=GraphMechanics(grid)
    p=Parameters3D(W=1.,gradient=(1.,1.,1.4),phase_misfit=.03,binding=.1,yield_stress=10.,phase_mobility=1.)
    loaded=CoupledCrystal(grid,p,mean_strain=np.array([[0,0,-.08],[0,0,0],[-.08,0,0]]));free=CoupledCrystal(grid,p)
    x,_,_=grid.coordinates();v=np.tanh(np.sin(2*np.pi*x/12)*12/(2*np.pi*np.sqrt(2)))
    fields=(v,np.full(grid.shape,.2),np.zeros((12,)+grid.shape));incident=encounter_graphs()['incident']
    def applied(graph):return graph_state(loaded,fields,graph,mechanics)['node_forces'][1]-graph_state(free,fields,graph,mechanics)['node_forces'][1]
    reference=applied(incident)[0]
    assert reference>0
    for theta in (0.,25.,40.,50.,60.,75.):
        a=np.deg2rad(theta);graph=cross_reflection_graph(incident,theta)
        along=(graph.projectors[1]@applied(graph))@np.array([np.cos(a),0,-np.sin(a)])
        assert along==pytest.approx(reference*np.cos(2*a),rel=1e-9,abs=1e-12)
