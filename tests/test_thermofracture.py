import numpy as np
import pytest
from bmvr.finite_elasticity import FiniteParameters,constitutive
from bmvr.thermoplastic import ThermalPlasticParameters,exponential_slip
from bmvr.thermochemical import ThermalChemical2D,ChemicalParameters
from bmvr.thermofracture import ThermalFracture2D,ThermalFractureState,ThermalFractureParameters


def copy(s):return ThermalFractureState(**{k:v.copy() for k,v in vars(s).items()})


def example(degree=2):
    model=ThermalFracture2D(3,degree=degree);m=model.mechanics.mesh;u=np.zeros(2*m.node_count)
    u[0::2]=.16*m.coordinates[:,1];u[1::2]=.12*m.coordinates[:,1];model.values=u[model.fixed].copy()
    s=model.initial(T=lambda x,y:1+.08*x-.05*y,u=u,c=lambda x,y:.25+.12*x,
        eta=lambda x,y:.25+.35*y,damage=lambda x,y:.1+.15*y,equilibrate=True)
    rng=np.random.default_rng(892);s.v=rng.normal(size=len(u))*.02;s.v[model.fixed]=0
    return model,s


def perturb(s,r,dt):
    values={k:v+dt*r[k] for k,v in vars(s).items() if k!='Fp'}
    values['Fp']=exponential_slip(r['Lp'],dt)@s.Fp
    return ThermalFractureState(**values)


@pytest.mark.parametrize('degree',[1,2])
def test_damage_and_reciprocal_phase_derivatives(degree):
    model,s=example(degree);r=model.rate(s);eps=1e-6;V=model.mechanics.mesh.h**2
    for key,target in [('damage','damage_force'),('eta','phase_force'),('c','mu')]:
        for index in range(9):
            a,b=copy(s),copy(s);getattr(a,key)[index]+=eps;getattr(b,key)[index]-=eps
            derivative=(model.rate(a)['diagnostics']['free_energy']-model.rate(b)['diagnostics']['free_energy'])/(2*eps)
            assert abs(derivative-V*r[target][index])<2e-9


@pytest.mark.parametrize('degree',[1,2])
def test_damage_local_and_global_first_laws_and_entropy(degree):
    model,s=example(degree);r=model.rate(s);eps=1e-6
    a=model.rate(perturb(s,r,eps));b=model.rate(perturb(s,r,-eps))
    numerical=(a['cell_internal_energy']-b['cell_internal_energy'])/(2*eps)
    power=sum(r[key] for key in ('cell_mechanical_power','cell_heat_power','chemical_power','microstress_power'))
    np.testing.assert_allclose(numerical,power,atol=3e-9,rtol=1e-7)
    assert abs((a['diagnostics']['total_energy']-b['diagnostics']['total_energy'])/(2*eps))<3e-9
    assert abs((a['diagnostics']['entropy']-b['diagnostics']['entropy'])/(2*eps)-r['diagnostics']['entropy_production'])<3e-9
    assert r['diagnostics']['damage_power']>0
    assert np.linalg.norm(r['damage_thermal_mixed'])>1e-7
    assert np.linalg.norm(r['damage_microstress_power'])>1e-7


def test_damage_entropy_mixed_derivative():
    model,s=example();r=model.rate(s);eps=1e-6;direction=np.linspace(-.3,.4,9)
    a,b=copy(s),copy(s);a.damage+=eps*direction;b.damage-=eps*direction
    numerical=(model.rate(a)['diagnostics']['entropy']-model.rate(b)['diagnostics']['entropy'])/(2*eps)
    expected=np.sum(r['additional_energy']['degradation_derivative']*model.mechanics.p.alpha_T*
        np.sum(model.mechanics.mesh.weights*r['local']['undamaged_M_trace'],axis=1)*direction)
    assert abs(numerical-expected)<2e-9


def test_scaled_constitutive_force_and_tangent():
    F=np.array([[1.2,.15,0.],[.02,1.08,0.],[0.,0.,1.]])
    H=np.array([[.2,-.3,0.],[.1,.3,0.],[0.,0.,0.]])
    kwargs=dict(c=.32,eta=.41,T=1.2,elastic_scale=.37);eps=1e-6
    r=constitutive(F,**kwargs);a=constitutive(F+eps*H,**kwargs);b=constitutive(F-eps*H,**kwargs)
    assert abs((a['energy']-b['energy'])/(2*eps)-np.sum(r['P']*H))<1e-9
    np.testing.assert_allclose((a['P']-b['P'])/(2*eps),np.einsum('ijkl,kl->ij',r['tangent'],H),atol=1e-9)


@pytest.mark.parametrize('degree',[1,2])
def test_undamaged_frozen_recovery(degree):
    a=ThermalChemical2D(3,degree=degree)
    b=ThermalFracture2D(3,degree=degree,fracture=ThermalFractureParameters(mobility=0))
    kwargs=dict(c=lambda x,y:.3+.05*x,eta=lambda x,y:.2+.2*y,T=lambda x,y:1+.1*x)
    s=a.initial(**kwargs);t=b.initial(**kwargs)
    ra=a.rate(s);rb=b.rate(t)
    for key in ('u','v','c','eta','T','gamma','Lp'):np.testing.assert_array_equal(ra[key],rb[key])
    for key in ra['diagnostics']:assert ra['diagnostics'][key]==rb['diagnostics'][key]


@pytest.mark.parametrize('degree',[1,2])
def test_complete_fracture_restart_and_irreversibility(tmp_path,degree):
    model,s=example(degree);initial=s.damage.copy()
    for _ in range(4):
        s,_=model.step(s,.001);assert np.all(s.damage>=initial);initial=s.damage.copy()
    path=tmp_path/'fracture.npz';model.save(path,s,.004,4)
    restored,t,clock=ThermalFracture2D.load(path);assert clock=={'time':.004,'step':4}
    a,ad=model.step(s,.001);b,bd=restored.step(t,.001)
    for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert ad==bd


def test_homogeneous_damage_analytic_limit_and_unloaded_arrest():
    p=FiniteParameters(alpha_c=0,alpha_eta=0,alpha_T=0)
    model=ThermalFracture2D(2,mechanical=p,transport=ThermalPlasticParameters(yield_stress=(100,100)),
        chemistry=ChemicalParameters(mobility=0,phase_mobility=0,partition=0),fixed=np.arange(18))
    m=model.mechanics.mesh;u=(.12*m.coordinates).ravel();model.values=u.copy()
    s=model.initial(u=u,c=.2,eta=.3,damage=.1);start=model.rate(s);f=model.fracture
    A=2*(1-f.residual)*start['local']['undamaged_elastic'][0,0];B=f.toughness/f.ell+2*f.phase_resistance*.3**2*(3-2*.3)
    target=A/(A+B)+(.1-A/(A+B))*np.exp(-f.mobility*(A+B)*.1)
    for _ in range(40):s,_=model.step(s,.0025)
    np.testing.assert_allclose(s.damage,target,atol=1e-6,rtol=0)
    z=f.mobility*(A+B)*.0025/2
    discrete=A/(A+B)+(.1-A/(A+B))*((1-z)/(1+z))**40
    np.testing.assert_allclose(s.damage,discrete,atol=1e-12,rtol=0)
    assert abs(model.rate(s)['diagnostics']['total_energy']-start['diagnostics']['total_energy'])<1e-9
    model.values[:]=0;s.u[:]=0;s.damage[:]=.4;s.T[:]=1
    assert np.all(model.rate(s)['damage']==0)
    s,_=model.step(s,.001);np.testing.assert_array_equal(s.damage,np.full(4,.4))


def test_invalid_damage_is_rejected_without_clipping():
    model,s=example();s.damage[0]=1.01
    with pytest.raises(ValueError,match='bounded damage'):model.step(s,.001)
    assert s.damage[0]==1.01
