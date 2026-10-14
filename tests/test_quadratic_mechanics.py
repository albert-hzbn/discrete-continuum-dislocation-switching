import numpy as np
import pytest
from bmvr.quadratic_mesh import QuadraticMesh,quadratic_shape
from bmvr.finite_elasticity import FiniteElasticity2D
from bmvr.thermochemical import ThermalChemical2D,ThermalChemicalState
from bmvr.thermoplastic import exponential_slip


def test_quadratic_shape_partition_and_integrals():
    mesh=QuadraticMesh(3);N,grad=quadratic_shape(np.array([[.13,-.29],[-.8,.77],[0,0]]),mesh.h)
    np.testing.assert_allclose(N.sum(axis=1),1,atol=1e-15)
    np.testing.assert_allclose(grad.sum(axis=1),0,atol=2e-15)
    np.testing.assert_allclose(mesh.weights.sum(),mesh.h**2,atol=1e-16)
    integral=np.einsum('g,ga->a',mesh.weights,mesh.N)
    np.testing.assert_allclose(integral,mesh.h**2*np.outer([1,4,1],[1,4,1]).ravel()/36,atol=1e-16)


def test_exact_quadratic_deformation_gradient():
    model=FiniteElasticity2D(3,degree=2);m=model.mesh;x,y=m.coordinates.T
    u=np.stack((.1*x*x*y,.07*x*y*y),axis=-1).ravel();state=model.evaluate(u)
    x,y=m.quadrature_points.transpose(2,0,1)
    F=np.broadcast_to(np.eye(3),(m.n*m.n,9,3,3)).copy()
    F[...,0,0]+=.2*x*y;F[...,0,1]=.1*x*x;F[...,1,0]=.07*y*y;F[...,1,1]+=.14*x*y
    np.testing.assert_allclose(state['F'],F,atol=2e-15)


def test_bernstein_reconstructs_determinant_and_bounds():
    mesh=QuadraticMesh(2);rng=np.random.default_rng(943);u=rng.normal(size=2*mesh.node_count)*.003
    coefficients=mesh.determinant_coefficients(u);t=np.linspace(0,1,11)
    B=np.stack(((1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3),axis=-1)
    reconstructed=np.einsum('ia,eab,jb->eij',B,coefficients,B)
    points=np.stack(np.meshgrid(2*t-1,2*t-1,indexing='ij'),axis=-1).reshape(-1,2)
    grad=quadratic_shape(points,mesh.h)[1]
    F=np.eye(2)+np.einsum('gaJ,eai->egiJ',grad,u.reshape(-1,2)[mesh.cells])
    actual=np.linalg.det(F).reshape(4,11,11)
    np.testing.assert_allclose(reconstructed,actual,atol=3e-15)
    assert coefficients.min()<=actual.min()+1e-15


def test_hidden_interior_inversion_rejected():
    model=FiniteElasticity2D(2,degree=2);m=model.mesh;x,y=m.coordinates.T
    u=np.stack((x*((y/m.h-.3)**2-1.01),np.zeros_like(x)),axis=-1).ravel()
    for gradient in (m.gradient,m.corner_gradient):
        F=np.eye(2)+np.einsum('gaJ,eai->egiJ',gradient,u.reshape(-1,2)[m.cells])
        assert np.min(np.linalg.det(F))>0
    assert m.orientation_lower_bound(u)<0
    with pytest.raises(ValueError,match='Bernstein'):model.evaluate(u)


def test_q2_assembled_energy_force_tangent():
    model=FiniteElasticity2D(2,degree=2,c=.3,eta=.4,T=1.1,body=[.1,-.1])
    rng=np.random.default_rng(493);u=rng.normal(size=2*model.mesh.node_count)*.001
    v=rng.normal(size=len(u));v/=np.linalg.norm(v);eps=1e-6
    state=model.evaluate(u,True);a=model.evaluate(u+eps*v);b=model.evaluate(u-eps*v)
    assert abs((a['potential']-b['potential'])/(2*eps)-state['force']@v)<2e-9
    np.testing.assert_allclose((a['force']-b['force'])/(2*eps),state['stiffness']@v,atol=3e-9)


def test_q2_affine_patch_and_positive_mass():
    model=FiniteElasticity2D(3,degree=2);X=model.mesh.coordinates
    exact=(X@np.array([[.15,.12],[.03,-.05]]).T).ravel();fixed=model.boundary()
    initial=exact.copy();free=np.ones(len(initial),dtype=bool);free[fixed]=False
    initial[free]+=.001*np.sin(np.arange(free.sum()))
    u,_,status=model.equilibrate(initial,fixed,exact[fixed])
    np.testing.assert_allclose(u,exact,atol=2e-10);assert status['relative_residual']<1e-9
    thermal=ThermalChemical2D(3,degree=2)
    assert thermal.mass.min()>0
    assert abs(thermal.mass.sum()/2-1)<1e-14


def coupled():
    model=ThermalChemical2D(3,degree=2);m=model.mechanics.mesh;u=np.zeros(2*m.node_count);u[0::2]=.12*m.coordinates[:,1]
    model.values=u[model.fixed].copy()
    s=model.initial(T=lambda x,y:1+.05*x,c=lambda x,y:.25+.1*y,eta=lambda x,y:.3+.15*x,u=u,equilibrate=True)
    s.v=.02*np.sin(np.arange(len(s.v)));s.v[model.fixed]=0
    return model,s


def test_q2_local_global_energy_entropy_and_mass():
    model,s=coupled();r=model.rate(s);eps=1e-6
    def perturb(dt):
        fields={k:v+dt*r[k] for k,v in vars(s).items() if k!='Fp'}
        fields['Fp']=exponential_slip(r['Lp'],dt)@s.Fp
        return ThermalChemicalState(**fields)
    a=model.rate(perturb(eps));b=model.rate(perturb(-eps))
    np.testing.assert_allclose((a['cell_internal_energy']-b['cell_internal_energy'])/(2*eps),
        sum(r[k] for k in ('cell_mechanical_power','cell_heat_power','chemical_power','microstress_power')),atol=2e-9)
    assert abs((a['diagnostics']['total_energy']-b['diagnostics']['total_energy'])/(2*eps))<2e-9
    assert abs((a['diagnostics']['entropy']-b['diagnostics']['entropy'])/(2*eps)-r['diagnostics']['entropy_production'])<2e-9
    assert abs(r['c'].sum())<1e-14


def test_q2_weighted_chemical_derivatives():
    model,s=coupled();r=model.rate(s);eps=1e-6;V=model.mechanics.mesh.h**2
    for key,target in (('c','mu'),('eta','phase_force')):
        a=ThermalChemicalState(**{k:v.copy() for k,v in vars(s).items()});b=ThermalChemicalState(**{k:v.copy() for k,v in vars(s).items()})
        getattr(a,key)[4]+=eps;getattr(b,key)[4]-=eps
        d=(model.rate(a)['diagnostics']['free_energy']-model.rate(b)['diagnostics']['free_energy'])/(2*eps)
        assert abs(d-V*r[target][4])<2e-9


def test_q2_complete_restart(tmp_path):
    model,s=coupled();s,_=model.step(s,.0005);path=tmp_path/'q2.npz';model.save(path,s,.0005,1)
    restored,t,clock=ThermalChemical2D.load(path)
    assert restored.mechanics.degree==2 and restored.mechanics.mesh.quadrature_count==9
    assert clock=={'time':.0005,'step':1}
    a,ad=model.step(s,.0005);b,bd=restored.step(t,.0005)
    for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert ad==bd


def test_q2_mechanical_restart(tmp_path):
    model=FiniteElasticity2D(2,degree=2);u=(model.mesh.coordinates*.03).ravel();path=tmp_path/'elastic.npz'
    model.save(path,u,.03,1);other,v,clock=FiniteElasticity2D.load(path)
    assert other.degree==2 and clock=={'load':.03,'step':1}
    np.testing.assert_array_equal(model.evaluate(u)['force'],other.evaluate(v)['force'])


def test_q2_same_grid_comparison_has_no_interpolation_error(tmp_path):
    from benchmarks.thermochemical import comparison
    model,s=coupled();s,_=model.step(s,.00025)
    model.save(tmp_path/'same-final.npz',s,.00025,1)
    errors=comparison({'name':'same','n':3},{'name':'same','n':3},tmp_path)
    assert errors==dict(c=0.,eta=0.,T=0.,slip=0.)
