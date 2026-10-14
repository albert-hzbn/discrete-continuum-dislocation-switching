"""Conservation, uniform-stress work and independent energy/PK force tests."""

from dataclasses import replace
import json
import numpy as np
import pytest
from bmvr.continuum3d import Cube, CoupledCrystal, fcc_slip_systems
from bmvr.loops3d import Loop, LoopMechanics, circular_loop, annihilate_coincident, velocities, coupled_state, coupled_step, save_checkpoint, load_checkpoint


def test_closed_loop_source_divergence_curl_and_mean_slip():
    grid = Cube(16,17)
    loop = circular_loop(count=11,center=(.2,.3,.4))
    source = LoopMechanics(grid).source([loop])
    assert source['divergence_relative'] < 1e-12
    assert source['curl_relative'] < 1e-12
    expected = -np.outer(loop.burgers,loop.area)/grid.volume
    assert np.max(np.abs(source['distortion'].mean(axis=(2,3,4))-expected)) < 1e-12


def test_exact_collinear_subdivision_preserves_source_and_energy():
    model = LoopMechanics(Cube(16,17))
    loop = circular_loop(count=9)
    fine = loop.subdivide(.6)
    a,b = model.source([loop]), model.source([fine])
    assert len(fine.nodes)>len(loop.nodes)
    assert np.max(np.abs(a['alpha_hat']-b['alpha_hat'])) < 1e-12
    assert np.max(np.abs(loop.area-fine.area)) < 1e-12
    assert model.evaluate([loop],need_forces=False)['energy'] == pytest.approx(model.evaluate([fine],need_forces=False)['energy'],rel=1e-12)
    coarsened = fine.coarsen_collinear()
    assert len(coarsened.nodes)==len(loop.nodes)
    assert np.max(np.abs(model.source([coarsened])['alpha_hat']-a['alpha_hat']))<1e-12


@pytest.mark.parametrize('background',[False,True])
def test_loop_pk_force_matches_energy_derivative(background):
    grid = Cube(12,15)
    model = LoopMechanics(grid,quadrature_order=10)
    loop = circular_loop(radius=2,count=7,center=(.2,.1,.3),burgers=(.2,.05,.08),core=.8)
    loop.nodes[:,2] += .15*np.sin(np.arange(7)*4*np.pi/7)
    rng = np.random.default_rng(231)
    direction = rng.normal(size=loop.nodes.shape)
    mean = np.array([[0,.025,-.02],[.025,.01,0],[-.02,0,0]])
    extra = None
    if background:
        x,y,z = grid.coordinates()
        extra = np.eye(3)[...,None,None,None]*(.04*np.cos(2*np.pi*x/12)*np.sin(2*np.pi*(z+y)/12))
    result = model.evaluate([loop],mean,extra)
    eps = 1e-5
    plus=model.evaluate([loop.moved(loop.nodes+eps*direction)],mean,extra,need_forces=False)['energy']
    minus=model.evaluate([loop.moved(loop.nodes-eps*direction)],mean,extra,need_forces=False)['energy']
    derivative=(plus-minus)/(2*eps)
    assert -np.sum(result['forces'][0]*direction) == pytest.approx(derivative,rel=1e-5,abs=1e-8)


def test_uniform_stress_force_agrees_with_oriented_area_work():
    grid=Cube(12,13)
    model=LoopMechanics(grid)
    loop=circular_loop(radius=2,count=8,burgers=(.2,0,0),core_tension=0)
    stress=np.array([[0,0,.12],[0,0,0],[.12,0,0]])
    field=np.broadcast_to(stress[...,None,None,None],(3,3)+grid.shape)
    forces=model.forces([loop],field)[0]
    direction=loop.nodes.copy()
    direction[:,2]=0
    eps=1e-5
    area_derivative=(loop.moved(loop.nodes+eps*direction).area-loop.moved(loop.nodes-eps*direction).area)/(2*eps)
    energy_derivative=np.sum(stress*np.outer(loop.burgers,area_derivative))
    assert -np.sum(forces*direction)==pytest.approx(energy_derivative,rel=1e-10)
    assert np.max(np.abs(forces.sum(axis=0)))<1e-12


def test_glide_projection_has_nonnegative_dissipation_and_stays_in_plane():
    loop=circular_loop(glide_normal=np.array([0.,0.,1.]))
    force=np.random.default_rng(2).normal(size=loop.nodes.shape)
    rates,dissipation=velocities([loop],[force])
    assert np.max(np.abs(rates[0][:,2]))==0
    assert dissipation>0
    loop.moved(loop.nodes+.01*rates[0])


def test_lab_xy_loop_is_an_allowed_rotated_fcc_slip_system():
    s=np.array([1.,-1.,0.])/np.sqrt(2)
    m=np.ones(3)/np.sqrt(3)
    rotation=np.array([s,np.cross(m,s),m])
    directions,normals,_=fcc_slip_systems(rotation)
    matching=(np.abs(directions @ np.array([1.,0.,0.]))>1-1e-12)&(np.abs(normals @ np.array([0.,0.,1.]))>1-1e-12)
    assert np.count_nonzero(matching)==1


def test_exact_opposite_loop_annihilation_cancels_complete_elastic_source():
    model=LoopMechanics(Cube(12,13))
    first=circular_loop(radius=2,count=8)
    second=replace(first,burgers=-first.burgers)
    result=model.evaluate([first,second],need_forces=False)
    assert result['elastic_energy']<1e-25
    remaining,event=annihilate_coincident(first,second)
    assert remaining==[] and event['accepted']
    assert event['released_core_energy']==pytest.approx(result['energy'])
    near=second.moved(second.nodes+[1e-4,0,0])
    assert not annihilate_coincident(first,near)[1]['accepted']


def test_loop_serialization_preserves_geometry_charge_and_constitutive_state():
    loop=circular_loop(glide_normal=np.array([0.,0.,1.]))
    restored=Loop(**json.loads(json.dumps(loop.to_dict())))
    assert np.array_equal(restored.nodes,loop.nodes)
    assert np.array_equal(restored.burgers,loop.burgers)
    assert restored.to_dict()==loop.to_dict()


def test_moving_loop_and_fields_share_one_decreasing_energy():
    grid=Cube(12,13)
    crystal=CoupledCrystal(grid,mean_strain=np.array([[0,.03,-.04],[.03,0,0],[-.04,0,0]]))
    mechanics=LoopMechanics(grid,quadrature_order=6)
    x,y,z=grid.coordinates()
    fields=(np.tanh((3-np.sqrt(x*x+y*y+z*z))/np.sqrt(2)),np.full(grid.shape,.2),np.zeros((12,)+grid.shape))
    loops=[circular_loop(radius=2,count=8,center=(1,0,0),glide_normal=np.array([0.,0.,1.]))]
    nodes=loops[0].nodes.copy()
    inventory=grid.integrate(fields[1])
    current=coupled_state(crystal,fields,loops,mechanics)
    for _ in range(4):
        old=current['energy']
        fields,loops,current,dt,trials=coupled_step(crystal,fields,loops,mechanics,.02,current)
        assert current['energy']<=old+1e-10
        assert abs(grid.integrate(fields[1])-inventory)<1e-10
        assert current['source_divergence_relative']<1e-12
        assert current['equilibrium_rms']<1e-10
    assert np.max(np.abs(nodes-loops[0].nodes))>1e-5
    assert np.max(np.abs(loops[0].nodes[:,2]))==0


def test_polygon_core_tension_recovers_circular_curvature_velocity():
    radius, tension, mobility = 3., .07, 1.2
    model = LoopMechanics(Cube(12,7))
    loop = circular_loop(radius=radius, count=11, core_tension=tension, mobility=mobility)
    force = model.forces([loop], np.zeros((3,3)+model.grid.shape))
    rate,_ = velocities([loop],force)
    expected = -mobility*tension/radius*(loop.nodes/radius)
    assert np.max(np.abs(rate[0]-expected)) < 1e-12


def test_two_loop_interaction_force_is_a_shared_energy_derivative():
    model = LoopMechanics(Cube(12,13),quadrature_order=10)
    first = circular_loop(radius=2,count=8,center=(-1,0,0))
    second = circular_loop(radius=1.4,count=7,center=(2,.2,.4),burgers=(-.15,.04,0))
    value = model.evaluate([first,second])
    direction = np.tile([1.,.3,-.2],(len(second.nodes),1))
    eps=1e-5
    plus = model.evaluate([first,second.moved(second.nodes+eps*direction)],need_forces=False)['energy']
    minus = model.evaluate([first,second.moved(second.nodes-eps*direction)],need_forces=False)['energy']
    assert -np.sum(value['forces'][1]*direction)==pytest.approx((plus-minus)/(2*eps),rel=1e-5,abs=1e-8)


@pytest.mark.parametrize('pinned',[False,True])
def test_coupled_loop_checkpoint_continues_identically(tmp_path,pinned):
    grid=Cube(12,9)
    crystal=CoupledCrystal(grid)
    mechanics=LoopMechanics(grid,quadrature_order=6)
    x,y,z=grid.coordinates()
    fields=(np.tanh((3-np.sqrt(x*x+y*y+z*z))/np.sqrt(2)),np.full(grid.shape,.2),np.zeros((12,)+grid.shape))
    loops=[circular_loop(radius=2,count=7,glide_normal=np.array([0.,0.,1.]))]
    fields,loops,state,used,_=coupled_step(crystal,fields,loops,mechanics,.01,pinned=pinned)
    path=tmp_path/'checkpoint.npz'
    save_checkpoint(path,crystal,fields,loops,mechanics,time=used,dt=.01,step=1,pinned=pinned)
    restored,rf,rl,rm,clock=load_checkpoint(path)
    a=coupled_step(crystal,fields,loops,mechanics,.01,pinned=pinned)
    b=coupled_step(restored,rf,rl,rm,clock['dt'],pinned=clock['pinned'])
    assert clock=={'time':used,'dt':.01,'step':1,'pinned':pinned}
    assert all(np.array_equal(x,y) for x,y in zip(a[0],b[0]))
    assert np.array_equal(a[1][0].nodes,b[1][0].nodes)
    assert a[2]['energy']==b[2]['energy']
