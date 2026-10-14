import numpy as np
import pytest
from bmvr.finite_elasticity import FiniteParameters
from bmvr.thermoplastic import ThermalPlastic2D, ThermalPlasticParameters, exponential_slip
from bmvr.thermochemical import ThermalChemical2D, ThermalChemicalState, ChemicalParameters


def example():
    model=ThermalChemical2D(3);m=model.mechanics.mesh;u=np.zeros(2*m.node_count)
    u[0::2]=.12*m.coordinates[:,1];model.values=u[model.fixed].copy()
    s=model.initial(T=lambda x,y:1+.08*x-.05*y,u=u,c=lambda x,y:.25+.12*x,
                    eta=lambda x,y:.25+.35*y,equilibrate=True)
    rng=np.random.default_rng(195);s.v=rng.normal(size=len(u))*.02;s.v[model.fixed]=0
    return model,s


def perturb(s,r,dt):
    values={k:v+dt*r[k] for k,v in vars(s).items() if k!='Fp'}
    values['Fp']=exponential_slip(r['Lp'],dt)@s.Fp
    return ThermalChemicalState(**values)


def test_discrete_chemical_and_phase_energy_derivatives():
    model,s=example();r=model.rate(s);eps=1e-6;V=model.mechanics.mesh.h**2
    for key,target in (('c','mu'),('eta','phase_force')):
        for index in range(9):
            a=ThermalChemicalState(**{k:v.copy() for k,v in vars(s).items()});b=ThermalChemicalState(**{k:v.copy() for k,v in vars(s).items()})
            getattr(a,key)[index]+=eps;getattr(b,key)[index]-=eps
            derivative=(model.rate(a)['diagnostics']['free_energy']-model.rate(b)['diagnostics']['free_energy'])/(2*eps)
            assert abs(derivative-V*r[target][index])<1e-9


def test_local_first_law_with_chemical_and_microstress_fluxes():
    model,s=example();r=model.rate(s);eps=1e-6
    a=model.rate(perturb(s,r,eps));b=model.rate(perturb(s,r,-eps))
    derivative=(a['cell_internal_energy']-b['cell_internal_energy'])/(2*eps)
    power=sum(r[key] for key in ('cell_mechanical_power','cell_heat_power','chemical_power','microstress_power'))
    np.testing.assert_allclose(derivative,power,atol=2e-9,rtol=1e-7)
    assert np.linalg.norm(r['chemical_power'])>1e-5
    assert np.linalg.norm(r['microstress_power'])>1e-5


def test_global_energy_entropy_and_inventory_rates():
    model,s=example();r=model.rate(s);eps=1e-6
    a=model.rate(perturb(s,r,eps))['diagnostics'];b=model.rate(perturb(s,r,-eps))['diagnostics']
    assert abs((a['total_energy']-b['total_energy'])/(2*eps))<2e-9
    assert abs((a['entropy']-b['entropy'])/(2*eps)-r['diagnostics']['entropy_production'])<2e-9
    assert abs(r['c'].sum())<1e-14
    assert r['diagnostics']['diffusion_power']>0 and r['diagnostics']['phase_power']>0


def test_chemical_thermal_mixed_derivatives():
    model,s=example();r=model.rate(s);eps=1e-6;m=model.mechanics.mesh;p=model.mechanics.p
    dc=np.linspace(-.3,.3,9);de=np.linspace(.2,-.4,9)
    a=ThermalChemicalState(**{k:v.copy() for k,v in vars(s).items()});b=ThermalChemicalState(**{k:v.copy() for k,v in vars(s).items()})
    a.c+=eps*dc;b.c-=eps*dc;a.eta+=eps*de;b.eta-=eps*de
    numerical=(model.rate(a)['diagnostics']['entropy']-model.rate(b)['diagnostics']['entropy'])/(2*eps)
    expected=-m.weight*np.sum(p.alpha_T*(2*p.mu*r['elastic_trace']+9*p.lam)*
        (p.alpha_c*dc+p.alpha_eta*6*s.eta*(1-s.eta)*de)[:,None])
    assert abs(numerical-expected)<2e-9


def test_availability_and_phase_boundary_rates():
    model=ThermalChemical2D(2,chemistry=ChemicalParameters(partition=10,kappa_c=0,kappa_eta=0))
    s=model.initial(c=np.array([0.,1.,.3,.8]),eta=np.array([1.,0.,0.,1.]))
    r=model.rate(s)
    assert r['c'][0]>=0 and r['c'][1]<=0 and abs(r['c'].sum())<1e-13
    assert np.all(r['eta'][s.eta==0]>=0) and np.all(r['eta'][s.eta==1]<=0)
    assert r['diagnostics']['diffusion_power']>=0 and r['diagnostics']['phase_power']>=0


def test_frozen_fields_recover_thermoplastic_rates():
    model=ThermalChemical2D(3,chemistry=ChemicalParameters(mobility=0,phase_mobility=0,partition=0))
    s=model.initial(T=1.1,c=lambda x,y:.2+.1*x,eta=lambda x,y:.2+.2*y)
    frozen=ThermalPlastic2D(3,c=np.repeat(s.c[:,None],4,axis=1),eta=np.repeat(s.eta[:,None],4,axis=1))
    t=frozen.initial(T=s.T,u=s.u)
    a=model.rate(s);b=frozen.rate(t)
    for key in ('u','v','T','gamma','Lp'):np.testing.assert_array_equal(a[key],b[key])


def test_full_restart(tmp_path):
    model,s=example();s,_=model.step(s,.001);p=tmp_path/'chemistry.npz';model.save(p,s,.001,1)
    restored,t,clock=ThermalChemical2D.load(p);assert clock=={'time':.001,'step':1}
    a,ad=model.step(s,.001);b,bd=restored.step(t,.001)
    for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert ad==bd


def test_invalid_scalar_state_rejected_without_clipping():
    model,s=example();s.c[0]=-1e-5
    with pytest.raises(ValueError,match='bounded c'):model.step(s,.001)
    assert s.c[0]==-1e-5


def uncoupled(n,chemistry):
    return ThermalChemical2D(n,mechanical=FiniteParameters(alpha_c=0,alpha_eta=0,alpha_T=0),
        transport=ThermalPlasticParameters(yield_stress=(100,100)),chemistry=chemistry,fixed=np.arange(2*(n+1)**2))


def test_constant_mobility_discrete_mode():
    n=4;model=uncoupled(n,ChemicalParameters(mobility_kind='constant',partition=0,phase_mobility=0))
    s=model.initial(c=lambda x,y:.3+.05*np.cos(np.pi*x)*np.cos(np.pi*y),eta=0)
    dt=.002;new,_=model.step(s,dt)
    k2=8*np.sin(np.pi/(2*n))**2/model.mechanics.mesh.h**2
    rate=.02*k2*(1+.002*k2)
    target=.3+(s.c-.3)*(1-dt*rate/2)/(1+dt*rate/2)
    np.testing.assert_allclose(new.c,target,atol=1e-13)


def test_homogeneous_phase_reaction_and_heat():
    model=uncoupled(2,ChemicalParameters(mobility=0,partition=0));s=model.initial(c=.3,eta=.3)
    for _ in range(20):s,d=model.step(s,.005)
    y0=-.4;y=y0/np.sqrt(y0*y0+(1-y0*y0)*np.exp(-.1));eta=(1+y)/2
    T=1+(.3**2*.7**2-eta**2*(1-eta)**2)/10
    np.testing.assert_allclose(s.eta,eta,atol=1e-8)
    np.testing.assert_allclose(s.T,T,atol=1e-8)
