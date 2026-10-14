"""Exact block formulas for plane-strain matrices, with general 3D fallback."""
import numpy as np


def plane_block(a):
    return (a.shape[-2:]==(3,3) and np.all(a[...,:2,2]==0) and np.all(a[...,2,:2]==0))


def determinant(a):
    a=np.asarray(a)
    if a.shape[-2:]==(2,2):return a[...,0,0]*a[...,1,1]-a[...,0,1]*a[...,1,0]
    if plane_block(a):return (a[...,0,0]*a[...,1,1]-a[...,0,1]*a[...,1,0])*a[...,2,2]
    return np.linalg.det(a)


def inverse(a):
    a=np.asarray(a)
    if not plane_block(a):return np.linalg.inv(a)
    det2=a[...,0,0]*a[...,1,1]-a[...,0,1]*a[...,1,0]
    if np.any(det2==0) or np.any(a[...,2,2]==0):raise np.linalg.LinAlgError('Singular matrix')
    result=np.zeros(a.shape,dtype=np.result_type(a,float))
    result[...,0,0]=a[...,1,1]/det2;result[...,0,1]=-a[...,0,1]/det2
    result[...,1,0]=-a[...,1,0]/det2;result[...,1,1]=a[...,0,0]/det2
    result[...,2,2]=1/a[...,2,2]
    return result
