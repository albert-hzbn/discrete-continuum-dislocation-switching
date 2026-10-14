import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from bmvr.finite_elasticity import FiniteParameters,constitutive
from bmvr.thermoplastic import ThermalPlasticParameters
from bmvr.thermochemical import ChemicalParameters
from bmvr.thermofracture import ThermalFracture2D,ThermalFractureParameters,crack_density_terms
from test_thermofracture import copy,perturb


@pytest.mark.parametrize('kind',['AT1','AT2'])
def test_isolated_surface_energy_normalization(kind):
    G=.031;ell=.14
    def density(x):
        if kind=='AT1':d=max(1-x/(2*ell),0)**2;derivative=-max(1-x/(2*ell),0)/ell
        else:d=np.exp(-x/ell);derivative=-d/ell
        w,_,cw=crack_density_terms(d,kind)
        return 2*G/cw*(w/ell+ell*derivative**2)
    value,_=quad(density,0,2*ell if kind=='AT1' else np.inf,epsabs=1e-12)
    assert abs(value-G)<1e-12


def homogeneous(ratio,degree=1):
    p=FiniteParameters(alpha_c=0,alpha_eta=0,alpha_T=0)
    f=ThermalFractureParameters(crack_density='AT1',elastic_split='volumetric-deviatoric')
    target=ratio*3*f.toughness/(16*f.ell*(1-f.residual))
    stretch=brentq(lambda a:constitutive(np.diag([a,a,1.]),parameters=p)['elastic']-target,1,2)
    model=ThermalFracture2D(2,degree=degree,mechanical=p,fracture=f,
        transport=ThermalPlasticParameters(yield_stress=(100,100)),
        chemistry=ChemicalParameters(mobility=0,phase_mobility=0,partition=0),fixed=np.arange(2*(2*degree+1)**2))
    u=((stretch-1)*model.mechanics.mesh.coordinates).ravel();model.values=u.copy()
    return model,model.initial(u=u,c=.2,eta=.3,damage=0),target


def test_below_threshold_is_exactly_stationary():
    model,s,_=homogeneous(.8);initial=copy(s)
    for _ in range(10):s,d=model.step(s,.01)
    for key in vars(s):np.testing.assert_array_equal(getattr(s,key),getattr(initial,key))
    assert d['damage_power']==0


def test_above_threshold_analytic_damage_and_heat():
    model,s,psi=homogeneous(4);p=model.fracture;A=2*(1-p.residual)*psi
    C=3*p.toughness/(8*p.ell);B=2*p.phase_resistance*.3**2*(3-2*.3)
    target=(A-C)/(A+B)*(1-np.exp(-p.mobility*(A+B)*.2))
    for _ in range(80):s,_=model.step(s,.0025)
    def energy(d):return (p.residual+(1-p.residual)*(1-d)**2)*psi+C*d+B*d*d/2
    temperature=1+(energy(0)-energy(target))/model.mechanics.p.cv
    np.testing.assert_allclose(s.damage,target,atol=1e-8,rtol=0)
    np.testing.assert_allclose(s.T,temperature,atol=1e-9,rtol=0)
    z=p.mobility*(A+B)*.0025/2;discrete=(A-C)/(A+B)*(1-((1-z)/(1+z))**80)
    np.testing.assert_allclose(s.damage,discrete,atol=1e-12,rtol=0)


def example(degree,split):
    model=ThermalFracture2D(3,degree=degree,fracture=ThermalFractureParameters(crack_density='AT1',elastic_split=split))
    m=model.mechanics.mesh;x,y=m.coordinates.T;u=np.zeros(2*m.node_count)
    u[0::2]=-.05*x+.3*y;u[1::2]=-.05*y;model.values=u[model.fixed].copy()
    s=model.initial(u=u,equilibrate=True,T=lambda x,y:1+.06*x,c=lambda x,y:.3+.03*y,
        eta=lambda x,y:.25+.1*x,damage=lambda x,y:.02+.01*y)
    rng=np.random.default_rng(8745);s.v=rng.normal(size=len(u))*.01;s.v[model.fixed]=0
    return model,s


@pytest.mark.parametrize('degree',[1,2])
@pytest.mark.parametrize('split',['isotropic','volumetric-deviatoric'])
def test_at1_local_first_law_entropy_and_damage_phase_derivatives(degree,split):
    model,s=example(degree,split);r=model.rate(s);eps=1e-6
    a=model.rate(perturb(s,r,eps));b=model.rate(perturb(s,r,-eps))
    power=sum(r[key] for key in ('cell_mechanical_power','cell_heat_power','chemical_power','microstress_power'))
    np.testing.assert_allclose((a['cell_internal_energy']-b['cell_internal_energy'])/(2*eps),power,atol=3e-9)
    assert abs((a['diagnostics']['total_energy']-b['diagnostics']['total_energy'])/(2*eps))<3e-9
    assert abs((a['diagnostics']['entropy']-b['diagnostics']['entropy'])/(2*eps)-r['diagnostics']['entropy_production'])<3e-9
    assert r['diagnostics']['damage_power']>0
    for key,target in [('damage','damage_force'),('eta','phase_force')]:
        a,b=copy(s),copy(s);direction=np.linspace(-.2,.3,9);getattr(a,key)[:]+=eps*direction;getattr(b,key)[:]-=eps*direction
        numerical=(model.rate(a)['diagnostics']['free_energy']-model.rate(b)['diagnostics']['free_energy'])/(2*eps)
        assert abs(numerical-model.mechanics.mesh.h**2*np.dot(r[target],direction))<2e-9


@pytest.mark.parametrize('degree',[1,2])
def test_at1_restart_and_no_healing(tmp_path,degree):
    model,s=example(degree,'volumetric-deviatoric');s,_=model.step(s,.001)
    path=tmp_path/'at1.npz';model.save(path,s,.001,1);restored,t,clock=ThermalFracture2D.load(path)
    assert restored.fracture.crack_density=='AT1' and clock['step']==1
    a,ad=model.step(s,.001);b,bd=restored.step(t,.001)
    for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert ad==bd and np.all(a.damage>=s.damage)
    model,s,_=homogeneous(.8,degree);s.damage[:]=.8
    a,_=model.step(s,.001);np.testing.assert_array_equal(a.damage,s.damage)
