"""Comparison transfer used by the graded/GLL notch studies (benchmarks/graded_notch.py)."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bmvr.quadratic_mesh import QuadraticMesh
from graded_notch import cellwise_slip,edges


@pytest.mark.parametrize('rule',['gauss','lobatto'])
def test_cellwise_slip_transfer_is_exact_for_quadratics(rule):
    A=QuadraticMesh(12,edges=edges(12),quadrature=rule);B=QuadraticMesh(16,edges=edges(16),quadrature='lobatto')
    f=lambda p:np.stack([1+.3*p[...,0]-.2*p[...,1]+.5*p[...,0]*p[...,1]+.7*p[...,0]**2,-.4*p[...,1]**2+.1*p[...,0]],-1)
    np.testing.assert_allclose(cellwise_slip(A,B,f(B.quadrature_points)),f(A.quadrature_points),atol=1e-13)
