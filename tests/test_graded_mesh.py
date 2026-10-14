import numpy as np
import pytest
from bmvr.quadratic_mesh import graded_edges,QuadraticMesh
from bmvr.finite_elasticity import FiniteElasticity2D,FiniteParameters
from bmvr.thermoplastic import ThermalPlastic2D,ThermalPlasticParameters
from bmvr.thermochemical import ThermalChemical2D,ChemicalParameters
from bmvr.thermofracture import ThermalFracture2D,ThermalFractureParameters
from bmvr.nodal_damage import NodalThermalFracture2D
from test_thermofracture import copy,perturb

# Deliberately lopsided cells: strong aspect ratios and unequal neighbors.
EDGES=(np.array([0.,.2,.55,1.]),np.array([0.,.45,.7,1.]))


def test_graded_edges_follow_the_declared_density():
    e=graded_edges(48,band=(.4,.6),ratio=3.,width=.03);h=np.diff(e)
    assert e[0]==0 and e[-1]==1 and np.all(h>0)
    centers=(e[:-1]+e[1:])/2
    assert abs(h[centers<.2].mean()/h[abs(centers-.5)<.05].mean()-3)<.1
    np.testing.assert_allclose(graded_edges(8,ratio=1.),np.linspace(0,1,9),atol=1e-12)
    with pytest.raises(ValueError):graded_edges(8,band=(.6,.4),ratio=2.)


def test_graded_affine_patch_quadrature_and_mass():
    model=FiniteElasticity2D(3,degree=2,edges=EDGES);m=model.mesh
    A=np.array([[.07,-.03],[.02,-.05]]);u=(m.coordinates@A.T).ravel()
    F=model.evaluate(u)['F'][...,:2,:2]
    np.testing.assert_allclose(F,np.broadcast_to(np.eye(2)+A,F.shape),atol=1e-14)
    assert abs(m.cell_weights.sum()-1)<1e-15 and abs(np.sum(m.cell_weights*m.quadrature_points[...,0])-.5)<1e-15
    assert abs(m.orientation_lower_bound(u)-np.linalg.det(np.eye(2)+A))<1e-14
    np.testing.assert_allclose(m.cell_volume,np.outer(np.diff(EDGES[0]),np.diff(EDGES[1])).ravel(),atol=1e-16)
    thermal=ThermalPlastic2D(3,degree=2,edges=EDGES)
    assert np.all(thermal.mass>0) and abs(thermal.mass.sum()/2-thermal.p.density)<1e-14
    with pytest.raises(ValueError,match='Q2'):FiniteElasticity2D(3,degree=1,edges=EDGES)
    with pytest.raises(ValueError,match='Graded edges'):FiniteElasticity2D(3,degree=2,edges=(EDGES[0],EDGES[1][::-1]))


def test_graded_mechanical_force_and_tangent_are_energy_derivatives():
    model=FiniteElasticity2D(3,degree=2,edges=EDGES,T=lambda x,y:1+.1*x,c=lambda x,y:.2+.05*y,eta=lambda x,y:.2+.2*x)
    rng=np.random.default_rng(461);u=.004*rng.normal(size=2*model.mesh.node_count)
    state=model.evaluate(u,tangent=True);direction=rng.normal(size=len(u))
    exact,Kd=state['force']@direction,state['stiffness']@direction;errors=[]
    for eps in (1e-5,1e-6):
        a=model.evaluate(u+eps*direction);b=model.evaluate(u-eps*direction)
        errors.append((abs((a['potential']-b['potential'])/(2*eps)-exact)/abs(exact),
                       np.linalg.norm((a['force']-b['force'])/(2*eps)-Kd)/np.linalg.norm(Kd)))
    # Central differences must show pure eps**2 truncation: exact analytic derivatives.
    for coarse,fine in zip(*errors):assert fine<1e-7 and 80<coarse/fine<120


def test_graded_two_point_flux_is_exact_for_linear_fields_and_conservative():
    model=ThermalChemical2D(3,degree=2,edges=EDGES);m=model.mechanics.mesh
    centers=m.coordinates[m.cells].mean(axis=1);z=.3*centers[:,0]-.7*centers[:,1]
    flux=model.edge_a*(z[model.edge_j]-z[model.edge_i]);split=len(flux)//2
    faces_x=np.broadcast_to(m.hy[None,:],(2,3)).ravel();faces_y=np.broadcast_to(m.hx[:,None],(3,2)).ravel()
    np.testing.assert_allclose(flux[:split],.3*faces_x,atol=1e-15)
    np.testing.assert_allclose(flux[split:],-.7*faces_y,atol=1e-15)
    s=model.initial(c=lambda x,y:.25+.2*x*y,eta=lambda x,y:.3+.1*y,T=lambda x,y:1+.2*x)
    r=model.rate(s)
    assert abs(np.sum(model.V*r['c']))<1e-15 and abs(r['cell_heat_power'].sum())<1e-15


def graded_example(cls,kind='AT2',quadrature='gauss'):
    model=cls(3,degree=2,edges=EDGES,quadrature=quadrature,fracture=ThermalFractureParameters(crack_density=kind,elastic_split='volumetric-deviatoric'))
    m=model.mechanics.mesh;x,y=m.coordinates.T;u=np.zeros(2*m.node_count)
    u[0::2]=-.05*x+.3*y;u[1::2]=-.05*y;model.values=u[model.fixed].copy()
    s=model.initial(u=u,equilibrate=True,T=lambda x,y:1+.06*x,c=lambda x,y:.3+.03*y,
        eta=lambda x,y:.25+.1*x,damage=lambda x,y:.12+.13*y)
    rng=np.random.default_rng(1994);s.v=rng.normal(size=len(u))*.01;s.v[model.fixed]=0
    return model,s


@pytest.mark.parametrize('quadrature',['gauss','lobatto'])
@pytest.mark.parametrize('cls',[ThermalFracture2D,NodalThermalFracture2D])
@pytest.mark.parametrize('kind',['AT1','AT2'])
def test_graded_energy_derivatives_and_local_global_balances(cls,kind,quadrature):
    model,s=graded_example(cls,kind,quadrature);r=model.rate(s);eps=1e-6;V=model.V
    nodal=cls is NodalThermalFracture2D
    for key,target in [('damage','damage_force'),('eta','phase_force'),('c','mu')]:
        for index in range(len(getattr(s,key))):
            a,b=copy(s),copy(s);getattr(a,key)[index]+=eps;getattr(b,key)[index]-=eps
            numerical=(model.rate(a)['diagnostics']['free_energy']-model.rate(b)['diagnostics']['free_energy'])/(2*eps)
            scale=1. if key=='damage' and nodal else V[index]
            assert abs(numerical-scale*r[target][index])<2e-9
    a=model.rate(perturb(s,r,eps));b=model.rate(perturb(s,r,-eps))
    power=sum(r[key] for key in ('cell_mechanical_power','cell_heat_power','chemical_power','microstress_power'))
    np.testing.assert_allclose((a['cell_internal_energy']-b['cell_internal_energy'])/(2*eps),power,atol=3e-9)
    assert abs((a['diagnostics']['total_energy']-b['diagnostics']['total_energy'])/(2*eps))<3e-9
    assert abs((a['diagnostics']['entropy']-b['diagnostics']['entropy'])/(2*eps)-r['diagnostics']['entropy_production'])<3e-9
    assert abs(r['microstress_power'].sum())<1e-14 and abs(r['chemical_power'].sum())<1e-14


@pytest.mark.parametrize('mode',[True,'full','adaptive'])
@pytest.mark.parametrize('quadrature',['gauss','lobatto'])
@pytest.mark.parametrize('cls',[ThermalFracture2D,NodalThermalFracture2D])
def test_graded_preconditioned_restart_is_deterministic(tmp_path,cls,quadrature,mode):
    model,s=graded_example(cls,quadrature=quadrature);model.preconditioned=mode
    s,_=model.step(s,.001);path=tmp_path/'graded.npz';model.save(path,s,.001,1)
    restored,t,clock=cls.load(path)
    assert restored.preconditioned==mode and not restored.mechanics.mesh.uniform and restored.mechanics.mesh.quadrature==quadrature
    np.testing.assert_array_equal(restored.mechanics.mesh.edges[0],EDGES[0])
    a,ad=model.step(s,.001);b,bd=restored.step(t,.001)
    for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert ad==bd


@pytest.mark.parametrize('mode',[True,'full','adaptive'])
@pytest.mark.parametrize('edges',[None,EDGES])
def test_preconditioned_and_picard_midpoints_agree(edges,mode):
    results=[]
    for preconditioned in (False,mode):
        model,s=graded_example(NodalThermalFracture2D) if edges is not None else graded_example_uniform()
        model.preconditioned=preconditioned
        for _ in range(3):s,d=model.step(s,.002)
        results.append((s,d))
    (a,ad),(b,bd)=results
    for key in vars(a):np.testing.assert_allclose(getattr(a,key),getattr(b,key),rtol=0,atol=1e-11)
    assert bd['midpoint_iterations']<=ad['midpoint_iterations']


def graded_example_uniform():
    model=NodalThermalFracture2D(3,degree=2);m=model.mechanics.mesh;u=np.zeros(2*m.node_count)
    u[0::2]=.1*m.coordinates[:,1];model.values=u[model.fixed].copy()
    return model,model.initial(u=u,equilibrate=True,T=lambda x,y:1+.06*x,c=lambda x,y:.3+.03*y,
        eta=lambda x,y:.25+.1*x,damage=lambda x,y:.12+.13*y)


def test_preconditioner_converges_where_picard_exceeds_its_allowance():
    def build(preconditioned):
        model=ThermalChemical2D(8,degree=2,mechanical=FiniteParameters(alpha_c=0,alpha_eta=0,alpha_T=0),
            transport=ThermalPlasticParameters(yield_stress=(100,100)),chemistry=ChemicalParameters(mobility=1.,kappa_c=.01),
            fixed=np.arange(2*17**2),preconditioned=preconditioned)
        return model,model.initial(c=lambda x,y:.3+.05*np.cos(np.pi*x)*np.cos(np.pi*y),eta=.3,T=lambda x,y:1+.1*np.cos(np.pi*x))
    model,s=build(False)
    # Picard divergence surfaces either as the iteration allowance or as an inadmissible iterate.
    with pytest.raises((RuntimeError,ValueError)):model.step(s,.05)
    model,s=build(True);start=model.rate(s)['diagnostics']
    for _ in range(4):s,d=model.step(s,.05)
    assert d['midpoint_iterations']<40 and abs(d['mass']-start['mass'])<1e-13
    assert abs(d['total_energy']-start['total_energy'])<1e-4*start['chemical_energy']


def test_lobatto_points_are_the_nodes_and_integrate_bicubics_exactly():
    gauss,lobatto=(QuadraticMesh(3,edges=EDGES,quadrature=q) for q in ('gauss','lobatto'))
    np.testing.assert_allclose(lobatto.N,np.eye(9),atol=1e-15)
    np.testing.assert_allclose(lobatto.quadrature_points,lobatto.coordinates[lobatto.cells],atol=1e-15)
    for m in (gauss,lobatto):
        x,y=m.quadrature_points[...,0],m.quadrature_points[...,1]
        for a in range(4):
            for b in range(4):assert abs(np.sum(m.cell_weights*x**a*y**b)-1/((a+1)*(b+1)))<1e-14
    # Degree 4 separates the rules: Gauss is exact, Lobatto is not.
    x4=[np.sum(m.cell_weights*m.quadrature_points[...,0]**4) for m in (gauss,lobatto)]
    assert abs(x4[0]-.2)<1e-14 and abs(x4[1]-.2)>1e-5
    with pytest.raises(ValueError,match='quadrature'):QuadraticMesh(3,quadrature='trapezoid')
    with pytest.raises(ValueError,match='Q2'):FiniteElasticity2D(3,degree=1,quadrature='lobatto')


@pytest.mark.parametrize('quadrature',['gauss','lobatto'])
def test_element_stiffness_has_only_rigid_null_modes(quadrature):
    """Section 47.2: small-strain element stiffness on lopsided cells has exactly three zero eigenvalues."""
    m=QuadraticMesh(3,edges=EDGES,quadrature=quadrature);lam,mu=1.3,.7
    D=np.array([[lam+2*mu,lam,0],[lam,lam+2*mu,0],[0,0,mu]])
    for e in range(m.n*m.n):
        g=m.gradient*m.grad_scale[e][None,None,:];B=np.zeros((9,3,18))
        B[:,0,0::2]=g[...,0];B[:,1,1::2]=g[...,1];B[:,2,0::2]=g[...,1];B[:,2,1::2]=g[...,0]
        K=np.einsum('g,gia,ij,gjb->ab',m.cell_weights[e],B,D,B);eig=np.linalg.eigvalsh(K)
        assert np.sum(eig<1e-10*eig.max())==3 and eig.min()>-1e-12*eig.max()


def test_lobatto_lumped_mass_is_the_diagonal_rule():
    model=ThermalPlastic2D(3,degree=2,edges=EDGES,quadrature='lobatto');m=model.mechanics.mesh
    expected=np.zeros(m.node_count)
    np.add.at(expected,m.cells,model.p.density*m.cell_weights)
    np.testing.assert_allclose(model.mass[0::2],expected,atol=1e-15);assert np.all(model.mass>0)


def test_invalid_preconditioner_mode_is_rejected():
    with pytest.raises(ValueError,match='preconditioned'):ThermalChemical2D(3,degree=2,preconditioned='mechanical')
