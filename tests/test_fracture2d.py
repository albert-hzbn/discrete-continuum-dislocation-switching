"""Independent constitutive, FE, bound and irreversible fracture verification."""
import numpy as np
import pytest
from scipy.optimize import minimize
from bmvr.fracture2d import Fracture2D,FractureParameters,spectral_elasticity,bound_quadratic


def test_spectral_stress_and_consistent_engineering_shear_tangent():
    e=np.array([[.1,-.03,.06],[-.05,-.1,.02],[.1,.1,0.]]);q=np.array([.2,.6,.1])
    d=np.array([[.3,.2,-.4],[-.2,.4,.1],[.4,-.3,.2]]);s=spectral_elasticity(e,q);delta=1e-7
    plus=spectral_elasticity(e+delta*d,q);minus=spectral_elasticity(e-delta*d,q)
    assert np.max(abs((plus['density']-minus['density'])/(2*delta)-np.sum(s['stress']*d,axis=-1)))<1e-9
    assert np.max(abs((plus['stress']-minus['stress'])/(2*delta)-np.einsum('...ij,...j->...i',s['tangent'],d)))<1e-8


def test_spectral_energy_is_rotation_invariant():
    angle=.37;Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    tensor=np.array([[.04,.02],[.02,-.01]]);rotated=Q@tensor@Q.T
    vector=lambda a:np.array([a[0,0],a[1,1],2*a[0,1]])
    a=spectral_elasticity(vector(tensor),np.array(.3));b=spectral_elasticity(vector(rotated),np.array(.3))
    assert a['density']==pytest.approx(b['density'],abs=1e-15)


def test_assembled_energy_force_damage_and_tangent_derivatives():
    model=Fracture2D(4);rng=np.random.default_rng(36);u=rng.normal(size=2*model.node_count)*.004
    q=rng.uniform(.1,.5,model.node_count);du=rng.normal(size=len(u));dq=rng.normal(size=len(q));delta=1e-7
    state=model.evaluate(u,q,tangent=True);A,rhs=model.damage_system(u)
    numerical=(model.evaluate(u+delta*du,q)['energy']-model.evaluate(u-delta*du,q)['energy'])/(2*delta)
    assert numerical==pytest.approx(state['force']@du,rel=1e-7,abs=1e-9)
    numerical=(model.evaluate(u,q+delta*dq)['energy']-model.evaluate(u,q-delta*dq)['energy'])/(2*delta)
    assert numerical==pytest.approx((A@q-rhs)@dq,rel=1e-7,abs=1e-9)
    derivative=(model.evaluate(u+delta*du,q)['force']-model.evaluate(u-delta*du,q)['force'])/(2*delta)
    assert np.max(abs(derivative-state['stiffness']@du))<1e-7


def test_affine_displacement_patch_and_hydrostatic_compression():
    model=Fracture2D(5);x,y=model.coordinates.T
    exact=np.stack((.02*x+.003*y,-.004*y+.003*x),axis=-1).ravel();q=np.full(model.node_count,.2)
    nodes=np.flatnonzero((x==0)|(x==1)|(y==0)|(y==1));fixed=(2*nodes[:,None]+np.arange(2)).ravel()
    u,_,status=model.equilibrate(np.zeros_like(exact),q,fixed,exact[fixed])
    assert np.max(abs(u-exact))<1e-10 and status['relative_residual']<1e-9
    u,q,state=model.solve_load(np.zeros_like(exact),np.zeros_like(q),.02,compression=True)
    assert np.max(q)<1e-12
    assert np.max(abs(u+.02*model.coordinates.ravel()))<1e-10


def test_damage_quadratic_agrees_with_independent_bound_optimizer():
    model=Fracture2D(4);u=np.zeros(2*model.node_count);u[1::2]=.035*model.coordinates[:,1]
    lower=model.seed();A,rhs=model.damage_system(u);actual,status=bound_quadratic(A,rhs,lower)
    ref=minimize(lambda q:(q@(A@q)/2-rhs@q,A@q-rhs),lower,jac=True,bounds=list(zip(lower,np.ones_like(lower))),
        method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-12,'maxiter':1000})
    assert ref.success and np.max(abs(actual-ref.x))<2e-6
    assert status['projected_residual']<1e-9


def test_uniform_tension_reaction_matches_closed_spectral_split_solution():
    model=Fracture2D(5);damage=.3;load=.02;q=np.full(model.node_count,damage)
    fixed,values=model.boundary(load);u,state,_=model.equilibrate(np.zeros(2*model.node_count),q,fixed,values)
    p=model.p;g=(1-damage)**2+p.residual
    transverse=-g*p.lam/(g*p.lam+2*p.shear)*load
    expected=np.stack((transverse*model.coordinates[:,0],load*model.coordinates[:,1]),axis=-1).ravel()
    modulus=2*g*p.shear+2*g*p.lam*p.shear/(g*p.lam+2*p.shear)
    top=2*np.flatnonzero(model.coordinates[:,1]==1)+1
    assert np.max(abs(u-expected))<1e-11
    assert np.sum(state['force'][top])==pytest.approx(modulus*load,rel=1e-10)


def test_planar_damage_profile_converges_at_fixed_width():
    errors=[];energies=[];ell=.1
    for n in (8,16,32):
        model=Fracture2D(n,parameters=FractureParameters(ell=ell));x=model.coordinates[:,0]
        lower=np.zeros(model.node_count);lower[x==.5]=1.;u=np.zeros(2*model.node_count)
        A,rhs=model.damage_system(u);q,_=bound_quadratic(A,rhs,lower)
        exact=np.cosh((.5-abs(x-.5))/ell)/np.cosh(.5/ell)
        errors.append(np.sqrt(np.mean((q-exact)**2)))
        energies.append(abs(model.evaluate(u,q)['fracture_energy']/(model.p.toughness*np.tanh(.5/ell))-1))
    assert errors[2]<errors[1]<errors[0] and energies[2]<energies[1]<energies[0]
    assert errors[-1]<.005 and energies[-1]<.01


def test_loading_unloading_is_irreversible_with_fixed_load_energy_descent():
    model=Fracture2D(6);u=np.zeros(2*model.node_count);q=model.seed();previous=q.copy()
    for load in (0,.02,.035,0):
        u,q,state=model.solve_load(u,q,load)
        assert np.min(q-previous)>-1e-13 and q.max()<=1
        assert np.max(np.diff(state['fixed_load_energy']))<1e-11
        assert state['mechanical_residual']<=1e-6 and state['damage_residual']<=1e-6
        previous=q.copy()


def test_restart_preserves_next_loading_state(tmp_path):
    model=Fracture2D(6,toughness=lambda x,y:.001*(1-.3*(1+np.tanh((x-.55)/.05))/2))
    u,q,state=model.solve_load(np.zeros(2*model.node_count),model.seed(),.02)
    path=tmp_path/'fracture.npz';model.save(path,u,q,.02,1)
    restored,ru,rq,clock=Fracture2D.load(path);assert clock=={'load':.02,'step':1}
    a,b,s=model.solve_load(u,q,.025);ra,rb,rs=restored.solve_load(ru,rq,.025)
    assert np.array_equal(a,ra) and np.array_equal(b,rb) and s['reaction']==rs['reaction']
