"""Section 50: reactive vacancies in the finite thermochemical nodal-damage model."""
import numpy as np
import pytest
from bmvr.thermoplastic import ThermalPlasticParameters,exponential_slip
from bmvr.thermofracture import ThermalFractureParameters
from bmvr.nodal_damage import NodalThermalFracture2D
from bmvr.vacancy_fracture import NodalVacancyFracture2D,VacancyParameters,VacancyFractureState

EDGES=(np.array([0.,.2,.55,1.]),np.array([0.,.45,.7,1.]))
ACTIVE=VacancyParameters(theta=.02,equilibrium=.05,binding=.003,sink=.5,damage_sink=5.,mobility=.25,generation=4.)


def build(cls=NodalVacancyFracture2D,vacancy=ACTIVE,edges=EDGES,**extra):
    kwargs=dict(degree=2,edges=edges,transport=ThermalPlasticParameters(yield_stress=(.006,.008)),
                fracture=ThermalFractureParameters(elastic_split='volumetric-deviatoric'),**extra)
    if cls is NodalVacancyFracture2D:kwargs['vacancy']=vacancy
    model=cls(3,**kwargs);m=model.mechanics.mesh;x,y=m.coordinates.T;u=np.zeros(2*m.node_count)
    u[0::2]=-.05*x+.3*y;u[1::2]=-.05*y;model.values=u[model.fixed].copy()
    fields=dict(u=u,equilibrate=True,T=lambda x,y:1+.06*x,c=lambda x,y:.3+.03*y,eta=lambda x,y:.25+.1*x,damage=lambda x,y:.12+.13*y)
    if cls is NodalVacancyFracture2D:fields['vac']=lambda x,y:.05+.01*x+.004*y
    s=model.initial(**fields)
    rng=np.random.default_rng(50);s.v=rng.normal(size=len(u))*.01;s.v[model.fixed]=0
    return model,s


def copy(s):return type(s)(**{k:v.copy() for k,v in vars(s).items()})


def perturb(s,r,dt):
    values={k:v+dt*r[k] for k,v in vars(s).items() if k!='Fp'}
    values['Fp']=exponential_slip(r['Lp'],dt)@s.Fp
    return type(s)(**values)


def test_free_energy_derivatives_include_vacancies_and_binding():
    model,s=build();r=model.rate(s);eps=1e-6;V=np.broadcast_to(model.V,s.vac.shape)
    assert np.max(abs(r['gamma']))>0 and np.max(r['vacancy_generation'])>0
    for key,target,scale in [('vac','vacancy_mu',V),('damage','damage_force',np.ones(len(s.damage))),('c','mu',V),('eta','phase_force',V)]:
        exact=r['additional_energy'][target] if target=='vacancy_mu' else r[target]
        for index in range(len(getattr(s,key))):
            a,b=copy(s),copy(s);getattr(a,key)[index]+=eps;getattr(b,key)[index]-=eps
            numerical=(model.rate(a)['diagnostics']['free_energy']-model.rate(b)['diagnostics']['free_energy'])/(2*eps)
            assert abs(numerical-scale[index]*exact[index])<3e-9


def test_local_first_law_global_energy_and_entropy_with_all_vacancy_processes():
    model,s=build();r=model.rate(s);eps=1e-6
    assert r['diagnostics']['slip_stored_power']>0 and r['diagnostics']['vacancy_dissipation']>0
    a=model.rate(perturb(s,r,eps));b=model.rate(perturb(s,r,-eps))
    power=sum(r[key] for key in ('cell_mechanical_power','cell_heat_power','chemical_power','microstress_power'))
    np.testing.assert_allclose((a['cell_internal_energy']-b['cell_internal_energy'])/(2*eps),power,atol=3e-9)
    assert abs((a['diagnostics']['total_energy']-b['diagnostics']['total_energy'])/(2*eps))<3e-9
    assert abs((a['diagnostics']['entropy']-b['diagnostics']['entropy'])/(2*eps)-r['diagnostics']['entropy_production'])<3e-9
    assert abs(r['chemical_power'].sum())<1e-14 and abs(r['diagnostics']['vacancy_flux_sum'])<1e-15


def test_inventory_ledger_and_flow_rule_dissipation_identity():
    model,s=build();r=model.rate(s);V=np.broadcast_to(model.V,s.vac.shape);p=model.vacancy
    assert np.sum(V*r['vac'])==pytest.approx(np.sum(V*(r['vacancy_generation']+r['vacancy_sink'])),abs=1e-15)
    mu=r['additional_energy']['vacancy_mu']
    # Stored generation power equals mu times the generated inventory rate.
    assert r['diagnostics']['slip_stored_power']==pytest.approx(np.sum(V*mu*r['vacancy_generation']),rel=1e-12)
    m=model.mechanics.mesh;slip=r['gamma'];y=np.asarray(model.p.yield_stress)
    assert r['diagnostics']['plastic_power']>=float(np.sum(m.cell_weights[...,None]*y*abs(slip)))-1e-15
    assert np.all(r['vacancy_transport_dissipation']>=-1e-16) and np.all(r['vacancy_sink_dissipation']>=0)


def test_vacancy_free_limit_reproduces_nodal_model_bitwise():
    base,a=build(NodalThermalFracture2D);model,b=build(vacancy=VacancyParameters());b.vac[:]=model.vacancy.equilibrium
    for _ in range(3):
        a,da=base.step(a,.002);b,db=model.step(b,.002)
    for key in vars(a):assert np.array_equal(getattr(a,key),getattr(b,key)),key
    for key in da:assert da[key]==db[key],key


def test_restart_is_deterministic(tmp_path):
    model,s=build(preconditioned='full');s,_=model.step(s,.001);path=tmp_path/'vac.npz';model.save(path,s,.001,1)
    restored,t,clock=NodalVacancyFracture2D.load(path)
    assert restored.vacancy==model.vacancy and clock['step']==1
    a,da=model.step(s,.001);b,db=restored.step(t,.001)
    for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert da==db


def test_sink_rate_formula_and_well_posedness_guard():
    p=VacancyParameters(theta=.02,equilibrium=.05,sink=.7)
    model,s=build(vacancy=p);s.damage[:]=0;s.vac[:]=.06;r=model.rate(s)
    expected=-p.sink*(p.formation+p.theta*np.log(.06/.94))
    np.testing.assert_allclose(r['vacancy_sink'],expected,rtol=1e-12)
    strong=VacancyParameters(theta=.02,equilibrium=.05,binding=.5,generation=4.)
    with pytest.raises(ValueError,match='well posed'):build(vacancy=strong)
    with pytest.raises(ValueError):VacancyParameters(equilibrium=1.)
