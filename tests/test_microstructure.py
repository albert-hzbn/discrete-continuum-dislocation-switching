import numpy as np
import pytest
from bmvr.grid import Grid
from bmvr.elasticity import coherent_elasticity
from bmvr.microstructure import dilatational_elasticity,contour_radii,junction_angles,grain_adjacency


def test_exact_dilatational_reduction_matches_full_fourier_mechanics():
    grid=Grid.uniform(20,31,periodic=True);rng=np.random.default_rng(451)
    phase=rng.normal(size=grid.shape)
    for lam,shear,misfit in [(2.,1.,.08),(1.3,2.1,.02)]:
        full=coherent_elasticity(grid,phase,misfit,lam,shear)
        reduced=dilatational_elasticity(grid,phase,misfit,lam,shear)
        assert full['energy']==pytest.approx(reduced['energy'],rel=1e-12)
        assert np.max(np.abs(full['force']-reduced['force']))<1e-12


def test_contour_radii_recover_ellipse_axes_without_pixel_mask_quantization():
    grid=Grid.uniform(16,129)
    field=1-(grid.X/3)**2-(grid.Y/5)**2
    radii=contour_radii(grid,field,angles=np.array([0,np.pi/2]))
    assert np.max(np.abs(radii-[3,5]))<.002


def test_junction_diagnostic_recovers_rotated_equal_sectors():
    grid=Grid.uniform(32,128);angles=.23+np.arange(3)*2*np.pi/3
    scores=np.array([.5*(grid.X*np.cos(a)+grid.Y*np.sin(a)) for a in angles])
    scores-=scores.max(axis=0);phases=np.exp(scores);phases/=phases.sum(axis=0)
    result=junction_angles(grid,phases,radius=6)
    assert result['mean_abs_error']<.1
    assert np.linalg.norm(result['center'])<.01


def test_periodic_stripe_adjacency_counts_wraparound_neighbors():
    grid=Grid.uniform(12,12,periodic=True)
    labels=np.tile(np.arange(12)//3,(12,1))
    phases=np.array([labels==i for i in range(4)],float)
    neighbors,contact=grain_adjacency(grid,phases)
    assert np.array_equal(neighbors,[2,2,2,2])
    assert contact[0,3]==pytest.approx(12)
