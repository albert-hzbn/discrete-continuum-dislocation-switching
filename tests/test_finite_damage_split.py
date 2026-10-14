import numpy as np
import pytest
from bmvr.finite_elasticity import FiniteParameters,constitutive
from bmvr.thermofracture import ThermalFracture2D,ThermalFractureParameters
from test_thermofracture import copy,perturb

SPLIT='volumetric-deviatoric'


@pytest.mark.parametrize('stretch',[.78,.91,1.06,1.2])
def test_hydrostatic_compression_is_preserved_and_expansion_degrades(stretch):
    p=FiniteParameters(alpha_c=0,alpha_eta=0,alpha_T=0);F=stretch*np.eye(3)
    base=constitutive(F,parameters=p);r=constitutive(F,parameters=p,elastic_scale=.15,elastic_split=SPLIT)
    factor=1 if stretch<1 else .15
    np.testing.assert_allclose(r['P'],factor*base['P'],atol=2e-15)
    assert abs(r['elastic']-factor*base['elastic'])<2e-15
    if stretch<1:assert abs(r['damage_elastic'])<2e-15


@pytest.mark.parametrize('stretch',[.84,1.17])
def test_split_derivatives_and_undamaged_recovery(stretch):
    F=np.array([[stretch,.18,0.],[.01,stretch,0.],[0.,0.,1.]])
    H=np.array([[.2,-.3,.02],[.1,.3,.03],[.01,0.,-.04]])
    kwargs=dict(c=.32,eta=.41,T=1.2,elastic_scale=.37,elastic_split=SPLIT);eps=1e-6
    r=constitutive(F,**kwargs);a=constitutive(F+eps*H,**kwargs);b=constitutive(F-eps*H,**kwargs)
    assert abs((a['energy']-b['energy'])/(2*eps)-np.sum(r['P']*H))<1e-9
    np.testing.assert_allclose((a['P']-b['P'])/(2*eps),np.einsum('ijkl,kl->ij',r['tangent'],H),atol=1e-9)
    np.testing.assert_allclose(r['tangent'],r['tangent'].transpose(2,3,0,1),atol=1e-14)
    for key,target in [('c','gc'),('eta','geta'),('T','entropy')]:
        ka,kb=dict(kwargs),dict(kwargs);ka[key]+=eps;kb[key]-=eps
        a,b=constitutive(F,**ka),constitutive(F,**kb)
        numerical=(a['energy']-b['energy'])/(2*eps)
        assert abs(numerical-(-1 if key=='T' else 1)*r[target])<2e-9
        if key=='T':
            assert abs(kwargs['T']*(a['entropy']-b['entropy'])/(2*eps)-r['heat_capacity'])<2e-8
            np.testing.assert_allclose((a['P']-b['P'])/(2*eps),r['PT'],atol=1e-9)
    unscaled=dict(kwargs,elastic_scale=1.)
    old=constitutive(F,**dict(unscaled,elastic_split='isotropic'));new=constitutive(F,**unscaled)
    for key in ('energy','P','M','tangent','entropy','heat_capacity','PT','gc','geta'):
        np.testing.assert_allclose(new[key],old[key],atol=1e-14,rtol=0)


def test_split_objectivity_and_switch_continuity():
    p=FiniteParameters(alpha_c=0,alpha_eta=0,alpha_T=0)
    kwargs=dict(parameters=p,elastic_scale=.2,elastic_split=SPLIT)
    theta=.4;R=np.array([[np.cos(theta),-np.sin(theta),0.],[np.sin(theta),np.cos(theta),0.],[0.,0.,1.]])
    F=np.array([[.9,.2,0.],[0.,.85,0.],[0.,0.,1.]])
    a,b=constitutive(F,**kwargs),constitutive(R@F,**kwargs)
    assert abs(a['energy']-b['energy'])<1e-14
    np.testing.assert_allclose(b['P'],R@a['P'],atol=1e-14)
    eps=1e-7;zero=constitutive(np.eye(3),**kwargs)
    minus=constitutive((1-eps)*np.eye(3),**kwargs);plus=constitutive((1+eps)*np.eye(3),**kwargs)
    assert abs(minus['energy'])<1e-12 and abs(plus['energy'])<1e-12
    assert np.linalg.norm(minus['P'])<2e-6 and np.linalg.norm(plus['P'])<2e-6
    np.testing.assert_allclose(plus['P']/eps,np.einsum('ijkl,kl->ij',zero['tangent'],np.eye(3)),atol=1e-6)


@pytest.mark.parametrize('sign',[-1,1])
def test_small_strain_volumetric_limit(sign):
    p=FiniteParameters(alpha_c=0,alpha_eta=0,alpha_T=0);eps=sign*1e-5
    r=constitutive((1+eps)*np.eye(3),parameters=p,elastic_scale=.2,elastic_split=SPLIT)
    expected=(.2 if sign>0 else 1)*.5*(p.lam+2*p.mu/3)*(3*eps)**2
    assert abs(r['elastic']/expected-1)<3e-5


def compressed_example(degree):
    model=ThermalFracture2D(3,degree=degree,fracture=ThermalFractureParameters(elastic_split=SPLIT))
    m=model.mechanics.mesh;x,y=m.coordinates.T;u=np.zeros(2*m.node_count)
    u[0::2]=-.05*x+.30*y;u[1::2]=-.05*y;model.values=u[model.fixed].copy()
    state=model.initial(u=u,equilibrate=True,T=lambda x,y:1+.06*x,c=lambda x,y:.3+.03*y,
        eta=lambda x,y:.25+.1*x,damage=lambda x,y:.02+.01*y)
    rng=np.random.default_rng(847);state.v=rng.normal(size=len(u))*.01;state.v[model.fixed]=0
    return model,state


@pytest.mark.parametrize('degree',[1,2])
def test_compressed_shear_local_first_law_entropy_and_damage_force(degree):
    model,s=compressed_example(degree);r=model.rate(s);eps=1e-6
    assert np.max(r['local']['Je'])<1 and r['diagnostics']['damage_power']>0
    a=model.rate(perturb(s,r,eps));b=model.rate(perturb(s,r,-eps))
    power=sum(r[key] for key in ('cell_mechanical_power','cell_heat_power','chemical_power','microstress_power'))
    np.testing.assert_allclose((a['cell_internal_energy']-b['cell_internal_energy'])/(2*eps),power,atol=3e-9)
    assert abs((a['diagnostics']['total_energy']-b['diagnostics']['total_energy'])/(2*eps))<3e-9
    assert abs((a['diagnostics']['entropy']-b['diagnostics']['entropy'])/(2*eps)-r['diagnostics']['entropy_production'])<3e-9
    a,b=copy(s),copy(s);direction=np.linspace(-.2,.3,9);a.damage+=eps*direction;b.damage-=eps*direction
    numerical=(model.rate(a)['diagnostics']['free_energy']-model.rate(b)['diagnostics']['free_energy'])/(2*eps)
    assert abs(numerical-model.mechanics.mesh.h**2*np.dot(r['damage_force'],direction))<2e-9


@pytest.mark.parametrize('degree',[1,2])
def test_split_restart_retains_compression_response(tmp_path,degree):
    model,s=compressed_example(degree);s,_=model.step(s,.001)
    path=tmp_path/'split.npz';model.save(path,s,.001,1);restored,t,clock=ThermalFracture2D.load(path)
    assert restored.fracture.elastic_split==SPLIT and clock['step']==1
    a,ad=model.step(s,.001);b,bd=restored.step(t,.001)
    for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert ad==bd
