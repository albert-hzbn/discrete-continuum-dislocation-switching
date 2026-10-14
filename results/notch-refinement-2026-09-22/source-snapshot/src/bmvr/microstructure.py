"""Quantitative diagnostics and restricted equilibrium reductions."""

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import root
from .energy import grain_energy,grain_force
from .solvers import project_simplex


def dilatational_elasticity(grid,phase,misfit=.08,lam=2.,shear=1.):
    """Exact odd-grid, zero-mean-strain, homogeneous isotropic plane-strain reduction."""
    grid.spectrum()
    if len(grid.x)%2==0 or len(grid.y)%2==0:
        raise ValueError('The exact reduction requires odd spectral grids without Nyquist null modes.')
    if shear<=0 or 3*lam+2*shear<=0:raise ValueError('Elastic moduli must be positive definite.')
    phi=(phase+1)/2;mean=grid.integrate(phi)/grid.area
    b0=3*shear+4.5*lam
    b=b0-(2*shear+3*lam)**2/(2*(lam+2*shear))
    return {'energy':misfit**2*(b*grid.integrate((phi-mean)**2)+b0*grid.area*mean**2),
            'force':misfit**2*(b*(phi-mean)+b0*mean)}


def contour_radii(grid,field,level=0.,angles=None,center=(0.,0.),samples=1025):
    """Interpolated first outward contour crossing on specified physical rays."""
    angles=np.linspace(0,2*np.pi,128,endpoint=False) if angles is None else np.asarray(angles)
    maximum=min(grid.xe[-1]-center[0],center[0]-grid.xe[0],grid.ye[-1]-center[1],center[1]-grid.ye[0])*.98
    radius=np.linspace(0,maximum,samples)
    x=center[0]+np.cos(angles)[:,None]*radius
    y=center[1]+np.sin(angles)[:,None]*radius
    coordinates=np.array([(y-grid.y[0])/grid.dy[0],(x-grid.x[0])/grid.dx[0]])
    values=map_coordinates(field,coordinates,order=1,mode='nearest')-level
    crossings=(values[:,:-1]>=0)&(values[:,1:]<0)
    if not np.all(crossings.any(axis=1)):raise ValueError('A closed central contour was not found on every ray.')
    index=np.argmax(crossings,axis=1);rows=np.arange(len(angles))
    fraction=values[rows,index]/(values[rows,index]-values[rows,index+1])
    return radius[index]+fraction*(radius[1]-radius[0])


def junction_angles(grid,phases,radius=6.,angular_samples=7200):
    if phases.shape[0]!=3:raise ValueError('This stationary junction diagnostic expects exactly three phases.')
    score=np.sum((phases-1/3)**2,axis=0)
    margin=max(2,int(radius/min(grid.dx[0],grid.dy[0])))
    score[:margin]=score[-margin:]=np.inf;score[:,:margin]=score[:,-margin:]=np.inf
    guess=np.array(np.unravel_index(np.argmin(score),score.shape),dtype=float)
    def differences(position):
        values=np.array([map_coordinates(p,position[:,None],order=1,mode='nearest')[0] for p in phases])
        return values[1:]-values[0]
    solution=root(differences,guess)
    if np.linalg.norm(differences(solution.x))>1e-7:raise RuntimeError('Equal-fraction junction point was not resolved.')
    y,x=solution.x
    theta=np.arange(angular_samples)*2*np.pi/angular_samples
    coords=np.array([y+radius/grid.dy[0]*np.sin(theta),x+radius/grid.dx[0]*np.cos(theta)])
    values=np.array([map_coordinates(p,coords,order=1,mode='nearest') for p in phases])
    labels=np.argmax(values,axis=0);crossings=np.flatnonzero(labels!=np.roll(labels,1))
    if len(crossings)!=3:raise RuntimeError('The measurement ring does not intersect exactly three boundaries.')
    positions=[]
    for j in crossings:
        previous=(j-1)%angular_samples;a,b=labels[previous],labels[j]
        left=values[a,previous]-values[b,previous];right=values[a,j]-values[b,j]
        fraction=left/(left-right)
        positions.append(((j-1+fraction)*2*np.pi/angular_samples)%(2*np.pi))
    positions=np.sort(positions)
    sectors=np.diff(np.r_[positions,positions[0]+2*np.pi])*180/np.pi
    return {'center':[float(grid.x[0]+x*grid.dx[0]),float(grid.y[0]+y*grid.dy[0])],
            'radius':radius,'degrees':sectors.tolist(),'mean_abs_error':float(np.mean(np.abs(sectors-120)))}


def stationary_junction(grid,tolerance=1e-5,max_iterations=6000,initial_phases=None):
    angles=np.arange(3)*2*np.pi/3
    scores=np.array([np.cos(a)*grid.X+np.sin(a)*grid.Y for a in angles])
    ordered=np.argsort(scores,axis=0)
    gap=np.take_along_axis(scores,ordered[-1:],axis=0)[0]-np.take_along_axis(scores,ordered[-2:-1],axis=0)[0]
    top=.5*(1+np.sin(np.minimum(gap/np.sqrt(3),np.pi/2)))
    boundary=np.zeros((3,)+grid.shape)
    np.put_along_axis(boundary,ordered[-1:],top[None],axis=0)
    np.put_along_axis(boundary,ordered[-2:-1],(1-top)[None],axis=0)
    shifted=np.array([1.5*(np.cos(a)*(grid.X-1)+np.sin(a)*(grid.Y+.8)) for a in angles])
    shifted-=shifted.max(axis=0);phases=np.exp(shifted);phases/=phases.sum(axis=0)
    mask=np.zeros(grid.shape,bool);mask[[0,-1],:]=True;mask[:,[0,-1]]=True
    def project(value):
        projected=project_simplex(value);projected[:,mask]=boundary[:,mask];return projected
    if initial_phases is not None:
        if initial_phases.shape!=phases.shape:raise ValueError('Restarted stationary phases must match the grid.')
        phases=np.array(initial_phases,copy=True)
    phases=project(phases);y=phases.copy();momentum=1.;step=.20*min(grid.dx[0],grid.dy[0])**2
    energy=grain_energy(grid,phases);history=[energy];residual=np.inf
    for iteration in range(1,max_iterations+1):
        candidate=project(y-step*grain_force(grid,y));value=grain_energy(grid,candidate)
        if value>energy+1e-12*max(1,abs(energy)):
            y=phases;momentum=1.
            candidate=project(y-step*grain_force(grid,y));value=grain_energy(grid,candidate)
        if value>energy+1e-10:raise RuntimeError('Stationary projected iteration failed descent.')
        previous=phases;phases=candidate;energy=value
        next_momentum=(1+np.sqrt(1+4*momentum**2))/2
        y=phases+(momentum-1)/next_momentum*(phases-previous);momentum=next_momentum
        if iteration%20==0:
            mapping=(phases-project(phases-step*grain_force(grid,phases)))/step
            residual=float(np.sqrt(np.mean(mapping**2)));history.append(energy)
            if residual<tolerance:break
    return phases,{'iterations':iteration,'projected_force_rms':residual,'energy':history,
                   'converged':residual<tolerance,'method':'Monotone restarted accelerated projected equilibrium; no physical time assigned.'}


def grain_adjacency(grid,phases,minimum_contact=1.):
    labels=np.argmax(phases,axis=0);count=len(phases);contact=np.zeros((count,count))
    for axis,length in ((0,grid.dx[0]),(1,grid.dy[0])):
        other=np.roll(labels,-1,axis=axis)
        a,b=np.minimum(labels,other),np.maximum(labels,other)
        mask=a!=b
        if not grid.periodic:
            if axis==0:mask[-1,:]=False
            else:mask[:,-1]=False
        contact+=np.bincount((a[mask]*count+b[mask]).ravel(),minlength=count*count).reshape(count,count)*length
    contact+=contact.T
    return (contact>minimum_contact).sum(axis=1),contact
