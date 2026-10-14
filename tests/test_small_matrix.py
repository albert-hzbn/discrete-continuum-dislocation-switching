import numpy as np
import pytest
from bmvr.small_matrix import determinant,inverse,plane_block


def test_two_dimensional_determinant():
    rng=np.random.default_rng(149);a=rng.normal(size=(11,7,2,2))
    np.testing.assert_allclose(determinant(a),np.linalg.det(a),atol=2e-15)


def test_plane_inverse_and_determinant():
    rng=np.random.default_rng(729);a=np.broadcast_to(np.eye(3),(37,3,3)).copy()
    a[:,:2,:2]+=rng.normal(size=(37,2,2))*.15;a[:,2,2]=rng.uniform(.7,1.4,37)
    assert plane_block(a)
    np.testing.assert_allclose(determinant(a),np.linalg.det(a),atol=1e-15)
    np.testing.assert_allclose(inverse(a),np.linalg.inv(a),atol=1e-15)
    np.testing.assert_allclose(a@inverse(a),np.broadcast_to(np.eye(3),a.shape),atol=3e-16)


def test_nonzero_out_of_plane_terms_use_general_path():
    a=np.array([[1.1,.2,1e-15],[.1,.9,0.],[0.,0.,1.2]])
    assert not plane_block(a)
    np.testing.assert_array_equal(determinant(a),np.linalg.det(a))
    np.testing.assert_array_equal(inverse(a),np.linalg.inv(a))


def test_singular_plane_block_rejected():
    with pytest.raises(np.linalg.LinAlgError):inverse(np.diag([1.,0.,1.]))
