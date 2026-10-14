import numpy as np
import pytest
from bmvr.nodal_damage import NodalThermalFracture2D
from bmvr.thermochemical import ThermalChemical2D
from bmvr.thermofracture import ThermalFracture2D,ThermalFractureParameters
from bmvr.finite_elasticity import FiniteParameters
from bmvr.thermoplastic import ThermalPlastic2D,ThermalPlasticParameters
from bmvr.thermochemical import ChemicalParameters
from test_thermofracture import copy,perturb


@pytest.mark.parametrize('degree',[1,2])
def test_partition_affine_gradient_and_positive_mass(degree):
    model=NodalThermalFracture2D(3,degree=degree);mesh=model.mechanics.mesh
    np.testing.assert_allclose(model.damage_N.sum(axis=1),1,atol=2e-16)
    np.testing.assert_allclose(model.damage_gradient.sum(axis=1),0,atol=5e-16)
    d=.3+.07*model.damage_coordinates[:,0]-.04*model.damage_coordinates[:,1]
    q=model.damage_quadrature(d);points=mesh.quadrature_points
    np.testing.assert_allclose(q,.3+.07*points[...,0]-.04*points[...,1],atol=2e-16)
    grad=np.einsum('gaj,ea->egj',model.damage_gradient,d[model.damage_cells])
    np.testing.assert_allclose(grad,np.broadcast_to([.07,-.04],grad.shape),atol=4e-16)
    assert np.all(model.damage_mass>0) and abs(model.damage_mass.sum()-1)<8*np.finfo(float).eps
    assert abs(model.damage_mass@d-.315)<2e-16


def example(degree,kind):
    model=NodalThermalFracture2D(3,degree=degree,fracture=ThermalFractureParameters(crack_density=kind,elastic_split='volumetric-deviatoric'))
    m=model.mechanics.mesh;x,y=m.coordinates.T;u=np.zeros(2*m.node_count)
    u[0::2]=-.05*x+.3*y;u[1::2]=-.05*y;model.values=u[model.fixed].copy()
    s=model.initial(u=u,equilibrate=True,T=lambda x,y:1+.06*x,c=lambda x,y:.3+.03*y,
        eta=lambda x,y:.25+.1*x,damage=lambda x,y:.12+.13*y)
    rng=np.random.default_rng(994);s.v=rng.normal(size=len(u))*.01;s.v[model.fixed]=0
    return model,s


@pytest.mark.parametrize('degree',[1,2])
@pytest.mark.parametrize('kind',['AT1','AT2'])
def test_nodal_energy_derivatives_and_local_global_balances(degree,kind):
    model,s=example(degree,kind);r=model.rate(s);eps=1e-6
    for index in range(len(s.damage)):
        a,b=copy(s),copy(s);a.damage[index]+=eps;b.damage[index]-=eps
        numerical=(model.rate(a)['diagnostics']['free_energy']-model.rate(b)['diagnostics']['free_energy'])/(2*eps)
        assert abs(numerical-r['damage_force'][index])<2e-9
    direction=np.linspace(-.2,.3,9);a,b=copy(s),copy(s);a.eta+=eps*direction;b.eta-=eps*direction
    numerical=(model.rate(a)['diagnostics']['free_energy']-model.rate(b)['diagnostics']['free_energy'])/(2*eps)
    assert abs(numerical-model.mechanics.mesh.h**2*np.dot(r['phase_force'],direction))<2e-9
    a=model.rate(perturb(s,r,eps));b=model.rate(perturb(s,r,-eps))
    power=sum(r[key] for key in ('cell_mechanical_power','cell_heat_power','chemical_power','microstress_power'))
    np.testing.assert_allclose((a['cell_internal_energy']-b['cell_internal_energy'])/(2*eps),power,atol=3e-9)
    assert abs((a['diagnostics']['total_energy']-b['diagnostics']['total_energy'])/(2*eps))<3e-9
    assert abs((a['diagnostics']['entropy']-b['diagnostics']['entropy'])/(2*eps)-r['diagnostics']['entropy_production'])<3e-9
    assert r['diagnostics']['damage_power']>0 and np.all(r['damage_dissipation']>=0)
    assert abs(r['damage_assembly_power'].sum())<1e-14
    assert np.linalg.norm(r['damage_assembly_power'])>1e-7


def test_mixed_element_work_has_positive_heat_and_pairwise_conservative_power():
    model=NodalThermalFracture2D(2,degree=2);element=np.zeros((4,4));node=4
    for e,value in [(0,-2.),(1,1.)]:element[e,np.flatnonzero(model.damage_cells[e]==node)[0]]=value
    force=model.assemble_damage(element);dot=np.zeros(9);dot[node]=.3
    nodal,heat,power=model.damage_heat_and_power(element,force,dot)
    work=np.sum(element*dot[model.damage_cells],axis=1)
    assert work.min()<0<work.max() and np.all(heat>=0)
    np.testing.assert_allclose(heat,np.full(4,.075),atol=1e-16)
    np.testing.assert_allclose(power,[-.525,.375,.075,.075],atol=2e-16)
    assert abs(power.sum())<2e-16 and abs(nodal.sum()-.3)<1e-16
    pair=np.zeros((4,4))
    for e in range(4):
        ae=int(np.flatnonzero(model.damage_cells[e]==node)[0])
        for f in range(4):
            af=int(np.flatnonzero(model.damage_cells[f]==node)[0])
            pair[e,f]=(element[e,ae]*model.damage_element_mass[f,af]-element[f,af]*model.damage_element_mass[e,ae])*dot[node]/model.damage_mass[node]
    np.testing.assert_allclose(pair,-pair.T,atol=1e-16)
    np.testing.assert_allclose(pair.sum(axis=1),power,atol=2e-16)


@pytest.mark.parametrize('degree',[1,2])
def test_undamaged_frozen_nodal_recovery(degree):
    a=ThermalChemical2D(3,degree=degree)
    b=NodalThermalFracture2D(3,degree=degree,fracture=ThermalFractureParameters(mobility=0))
    args=dict(T=lambda x,y:1+.1*x,c=lambda x,y:.3+.05*x,eta=lambda x,y:.3+.1*y)
    ra=a.rate(a.initial(**args));rb=b.rate(b.initial(**args))
    for key in ('u','v','c','eta','T','gamma','Lp'):np.testing.assert_array_equal(ra[key],rb[key])
    for key in ra['diagnostics']:assert ra['diagnostics'][key]==rb['diagnostics'][key]


@pytest.mark.parametrize('degree',[1,2])
@pytest.mark.parametrize('kind',['AT1','AT2'])
def test_nodal_restart_irreversibility_and_basis_rejection(tmp_path,degree,kind):
    model,s=example(degree,kind);start=s.damage.copy();s,_=model.step(s,.001)
    assert np.all(s.damage>=start)
    path=tmp_path/'nodal.npz';model.save(path,s,.001,1)
    restored,t,clock=NodalThermalFracture2D.load(path);assert clock=={'time':.001,'step':1}
    a,ad=model.step(s,.001);b,bd=restored.step(t,.001)
    for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert ad==bd and np.all(a.damage>=s.damage)
    with pytest.raises(ValueError,match='damage basis'):ThermalFracture2D.load(path)
    cell=ThermalFracture2D(3,degree=degree);path2=tmp_path/'cell.npz';cell.save(path2,cell.initial())
    with pytest.raises(ValueError,match='damage basis'):NodalThermalFracture2D.load(path2)


def test_nodal_bounds_reject_without_clipping():
    model,s=example(2,'AT2');s.damage[0]=1.001
    with pytest.raises(ValueError,match='bounded nodal damage'):model.step(s,.001)
    assert s.damage[0]==1.001


def near_upper_bound():
    model=NodalThermalFracture2D(2,mechanical=FiniteParameters(alpha_c=0,alpha_eta=0,alpha_T=0),
        transport=ThermalPlasticParameters(yield_stress=(100,100)),
        chemistry=ChemicalParameters(mobility=0,phase_mobility=0,partition=0),fixed=np.arange(18))
    u=np.zeros(18);u[0::2]=.6*model.mechanics.mesh.coordinates[:,1];model.values=u.copy()
    s=model.initial(u=u,c=.2,eta=0,damage=.3);s.damage[4]=.99999
    return model,s


def test_upper_bound_is_solved_with_its_heat_and_no_healing():
    model,s=near_upper_bound();initial=model.rate(s)['diagnostics']
    with pytest.raises(ValueError,match='bounded nodal'):ThermalPlastic2D.step(model,s,.01)
    t,d=model.step(s,.01)
    assert t.damage[4]==1 and np.all(t.damage>=s.damage) and t.damage.max()==1
    assert d['newly_saturated_damage_nodes']>=1 and d['midpoint_damage_obstacle_power']>0
    assert d['midpoint_damage_power']>=d['midpoint_damage_obstacle_power']
    assert abs(d['total_energy']-initial['total_energy'])<1e-10
    assert d['entropy']>=initial['entropy']
    u,dd=model.step(t,.01);assert u.damage[4]==1 and np.all(u.damage>=t.damage)
    assert dd['midpoint_damage_power']>=0


def test_inactive_obstacle_recovers_original_midpoint():
    model,s=example(2,'AT2')
    a,ad=model.step(s,.001);b,bd=ThermalPlastic2D.step(model,s,.001)
    for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    for key in bd:assert ad[key]==bd[key]
    assert ad['midpoint_damage_obstacle_power']<1e-16 and ad['newly_saturated_damage_nodes']==0


def test_bound_arrival_time_refinement_and_energy():
    results=[]
    for dt in (.01,.005,.0025,.00125):
        model,s=near_upper_bound();energy=model.rate(s)['diagnostics']['total_energy']
        for _ in range(round(.04/dt)):
            s,d=model.step(s,dt)
            assert s.damage.max()<=1 and d['midpoint_damage_power']>=0
        assert abs(d['total_energy']-energy)<1e-10
        results.append(np.r_[s.damage,s.T])
    errors=[np.linalg.norm(s-results[-1]) for s in results[:-1]]
    assert errors[0]>errors[1]>errors[2]
