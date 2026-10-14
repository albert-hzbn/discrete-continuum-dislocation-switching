"""Periodic plane-strain tensor microelasticity and finite-width pair forces."""

import numpy as np


def tensor_elasticity(grid,eigenstrain,lam=2.,shear=1.):
    grid.spectrum()
    if any(n%2==0 for n in grid.shape) or shear<=0 or 3*lam+2*shear<=0:
        raise ValueError('Require odd grids and positive elastic energy.')
    eigenstrain=np.asarray(eigenstrain,dtype=float)
    if eigenstrain.shape!=(3,3)+grid.shape or not np.allclose(eigenstrain,eigenstrain.swapaxes(0,1),atol=1e-12,rtol=0):
        raise ValueError('Require a symmetric 3x3 tensor field.')
    if np.max(np.abs(eigenstrain[:2,2]))>1e-12:
        raise ValueError('Out-of-plane shear eigenstrain is outside this plane-strain closure.')
    identity=np.eye(3)[:,:,None,None]
    def stress(e):return 2*shear*e+lam*identity*np.einsum('ii...->...',e)
    k=np.array(np.meshgrid(2*np.pi*np.fft.fftfreq(len(grid.x),grid.dx[0]),
                          2*np.pi*np.fft.fftfreq(len(grid.y),grid.dy[0])))
    k2=np.sum(k*k,axis=0);inverse=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
    forcing=np.einsum('j...,ij...->i...',k,np.fft.fft2(stress(eigenstrain)[:2,:2]))
    longitudinal=k*np.einsum('i...,i...->...',k,forcing)*inverse
    uhat=-1j*(forcing-(lam+shear)/(lam+2*shear)*longitudinal)*inverse/shear
    gradient=1j*np.einsum('j...,i...->ij...',k,uhat)
    strain=np.zeros_like(eigenstrain)
    strain[:2,:2]=np.fft.ifft2((gradient+gradient.swapaxes(0,1))/2).real
    elastic_strain=strain-eigenstrain;sigma=stress(elastic_strain)
    divergence=np.fft.ifft2(1j*np.einsum('j...,ij...->i...',k,np.fft.fft2(sigma[:2,:2]))).real
    return {'energy':grid.integrate(.5*np.einsum('ij...,ij...->...',elastic_strain,sigma)),
            'stress':sigma,'strain':strain,'displacement':np.fft.ifft2(uhat).real,
            'equilibrium_rms':float(np.sqrt(np.mean(np.sum(divergence**2,axis=0))))}


def inclusion_profile(grid,center):
    length_x=grid.xe[-1]-grid.xe[0];length_y=grid.ye[-1]-grid.ye[0]
    x=(grid.X-center[0]+length_x/2)%length_x-length_x/2
    y=(grid.Y-center[1]+length_y/2)%length_y-length_y/2
    radius=np.hypot(x,y);t=np.clip((radius-2)/2,0.,1.)
    phi=1-10*t**3+15*t**4-6*t**5
    radial=(-30*t**2+60*t**3-30*t**4)/2
    factor=np.divide(radial,radius,out=np.zeros_like(radius),where=radius>0)
    return phi,np.array([factor*x,factor*y])


def transformation_tensor(angle=0.,dilatational=False):
    if dilatational:return .08*np.eye(3)
    c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    return rotation@np.diag([.12,-.04,-.04])@rotation.T


def pair_evaluation(grid,center1,center2,tensor1=None,tensor2=None,save_fields=False):
    tensor1=transformation_tensor() if tensor1 is None else tensor1
    tensor2=transformation_tensor() if tensor2 is None else tensor2
    phi1,grad1=inclusion_profile(grid,center1);phi2,grad2=inclusion_profile(grid,center2)
    eigen1=tensor1[:,:,None,None]*phi1;eigen2=tensor2[:,:,None,None]*phi2
    first=tensor_elasticity(grid,eigen1);second=tensor_elasticity(grid,eigen2)
    total=tensor_elasticity(grid,eigen1+eigen2)
    contraction12=np.einsum('ij...,ij->...',first['stress'],tensor2)
    contraction21=np.einsum('ij...,ij->...',second['stress'],tensor1)
    force2=np.array([-grid.integrate(contraction12*g) for g in grad2])
    force1=np.array([-grid.integrate(contraction21*g) for g in grad1])
    mask=(phi2>0)&(phi2<1)
    interaction=total['energy']-first['energy']-second['energy']
    cross12=-grid.integrate(contraction12*phi2);cross21=-grid.integrate(contraction21*phi1)
    result={'interaction_energy':interaction,'combined_energy':total['energy'],
            'self_energies':[first['energy'],second['energy']],
            'cross_work_12':cross12,'cross_work_21':cross21,
            'reciprocity_error':max(abs(interaction-cross12),abs(interaction-cross21)),
            'force1':force1.tolist(),'force2':force2.tolist(),
            'force_sum_norm':float(np.linalg.norm(force1+force2)),
            'cross_phase_drive_layer_rms':float(np.sqrt(np.mean(contraction12[mask]**2))),
            'inventories':[grid.integrate(phi1),grid.integrate(phi2)],'overlap':grid.integrate(phi1*phi2),
            'equilibrium_rms':max(first['equilibrium_rms'],second['equilibrium_rms'],total['equilibrium_rms'])}
    if save_fields:
        result['fields']={'phi1':phi1,'phi2':phi2,'stress':total['stress'],'strain':total['strain'],
                          'cross_phase_drive':-contraction12,'x':grid.x,'y':grid.y}
    return result
