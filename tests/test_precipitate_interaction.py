import numpy as np
from bmvr.grid import Grid
from bmvr.elasticity import coherent_elasticity
from bmvr.precipitate_interaction import tensor_elasticity,pair_evaluation,transformation_tensor


def test_tensor_solver_recovers_dilatational_mechanics():
    grid=Grid.uniform(20,33,periodic=True)
    phase=np.random.default_rng(89).uniform(-1,1,grid.shape)
    old=coherent_elasticity(grid,phase)
    new=tensor_elasticity(grid,.08*np.eye(3)[:,:,None,None]*(phase+1)/2)
    np.testing.assert_allclose(new['energy'],old['energy'],rtol=1e-12)
    for i,j,key in ((0,0,'sxx'),(1,1,'syy'),(2,2,'szz'),(0,1,'sxy')):
        np.testing.assert_allclose(new['stress'][i,j],old[key],atol=1e-12)


def test_compatible_manufactured_eigenstrain_is_stress_free():
    grid=Grid.uniform(20,33,periodic=True);eigen=np.zeros((3,3)+grid.shape)
    eigen[0,0]=.03*np.cos(2*np.pi*grid.X/20)
    result=tensor_elasticity(grid,eigen)
    assert result['energy']<1e-25
    assert np.max(np.abs(result['stress']))<1e-12


def test_cross_force_matches_independent_energy_translation():
    grid=Grid.uniform(40,129,periodic=True);left=np.array([-5.,-1.]);right=np.array([5.,1.])
    e2=transformation_tensor(.3);result=pair_evaluation(grid,left,right,tensor2=e2)
    step=1e-4;derivative=[]
    for axis in np.eye(2):
        plus=pair_evaluation(grid,left,right+step*axis,tensor2=e2)['interaction_energy']
        minus=pair_evaluation(grid,left,right-step*axis,tensor2=e2)['interaction_energy']
        derivative.append(-(plus-minus)/(2*step))
    np.testing.assert_allclose(result['force2'],derivative,rtol=1e-5,atol=1e-8)
    assert result['reciprocity_error']<1e-10


def test_disjoint_dilatational_pair_retains_only_inventory_term():
    grid=Grid.uniform(40,65,periodic=True);tensor=transformation_tensor(dilatational=True)
    result=pair_evaluation(grid,(-5.,0.),(5.,0.),tensor,tensor)
    assert result['overlap']==0
    expected=2*.08**2*8*np.prod(result['inventories'])/grid.area
    np.testing.assert_allclose(result['interaction_energy'],expected,rtol=1e-12,atol=1e-12)
