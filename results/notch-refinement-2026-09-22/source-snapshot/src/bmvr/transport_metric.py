"""Onsager projection of a planar loop/detail chart onto continuum transport.

The full mobility is unchanged, including pinned zero-mobility sites. Dense
correction fields count toward storage; low coordinate count is not low memory.
"""
import numpy as np
from .representation3d import NyeTransfer


def weighted_projection(columns, full_rate, mobility, volume, rcond=1e-11):
    """Project physical rates in the inverse-mobility norm with pinned constraints."""
    B=np.asarray(columns).reshape(len(columns),-1).T
    rate=np.asarray(full_rate).ravel();K=np.asarray(mobility).ravel()
    if volume<=0 or np.any(K<0) or not np.isfinite(K).all() or not np.isfinite(B).all() or not np.isfinite(rate).all():
        raise ValueError('Finite columns and nonnegative mobility are required.')
    active=K>0;coordinates=B.shape[1]
    if np.any(rate[~active]!=0):
        raise ValueError('A nonzero full rate at zero mobility has no finite dissipation norm.')
    Z=np.eye(coordinates)
    pinned=B[~active]
    constraint_rank=0
    if len(pinned) and np.any(pinned):
        _,singular,vt=np.linalg.svd(pinned,full_matrices=len(pinned)<coordinates)
        constraint_rank=int(np.count_nonzero(singular>rcond*singular[0]))
        Z=vt[constraint_rank:].T
    weight=np.sqrt(volume/K[active])
    J=(B[active]@Z)*weight[:,None]
    target=rate[active]*weight
    coefficients=np.zeros(coordinates);rank=0;singular=np.array([])
    if J.shape[1]:
        scale=np.linalg.norm(J,axis=0)
        live=scale>0
        if np.any(live):
            scaled=J[:,live]/scale[live]
            u,singular,vt=np.linalg.svd(scaled,full_matrices=False)
            keep=singular>rcond*singular[0]
            rank=int(np.count_nonzero(keep))
            reduced=np.zeros(J.shape[1])
            reduced[live]=(vt[keep].T@((u[:,keep].T@target)/singular[keep]))/scale[live]
            coefficients=Z@reduced
    projected=B@coefficients
    residual=rate-projected
    p=projected[active]*weight;r=residual[active]*weight
    full_norm=float(target@target);discarded=float(r@r);kept=float(p@p)
    return {'coordinates':coefficients,'rate':projected.reshape(full_rate.shape),
        'residual':residual.reshape(full_rate.shape),'full_norm_squared':full_norm,
        'discarded_norm_squared':discarded,'projected_norm_squared':kept,
        'relative_residual':float(np.sqrt(discarded/max(full_norm,1e-30))),
        'orthogonality':float(p@r),'pythagorean_error':float(abs(full_norm-kept-discarded)),
        'pinned_rate_max':float(np.max(np.abs(projected[~active]))) if np.any(~active) else 0.,
        'constraint_rank':constraint_rank,'rank':rank,'singular_values':singular.tolist()}


class PlanarTransport:
    def __init__(self, grid, burgers=(.2,0,0), **kwargs):
        self.burgers=np.asarray(burgers,dtype=float)
        if self.burgers.shape!=(3,) or np.linalg.norm(self.burgers)==0 or abs(self.burgers[2])>1e-12:
            raise ValueError('Use a nonzero in-plane Burgers vector.')
        self.normal=np.array([0.,0.,1.]);self.grid=grid
        self.base=NyeTransfer(grid,glide_normal=self.normal,**kwargs)

    def embed(self,gamma):
        return np.outer(self.burgers,self.normal)[...,None,None,None]*gamma

    def evaluate(self,gamma):
        state=self.base.evaluate(self.embed(gamma))
        a=np.einsum('i,ij...->j...',self.burgers,state['alpha'])/(self.burgers@self.burgers)
        conjugate=np.einsum('i,ij...,j->...',self.burgers,state['H'],self.normal)
        mobility=self.base.field_mobility*np.sum(a*a,axis=0)
        state.update({'gamma':gamma,'conjugate':conjugate,'mobility':mobility,'full_rate':mobility*conjugate})
        return state

    def gamma_from_loops(self,loops,history=0.):
        for loop in loops:
            factor=loop.burgers@self.burgers/(self.burgers@self.burgers)
            if not np.allclose(loop.burgers,factor*self.burgers,rtol=0,atol=1e-12) or np.ptp(loop.nodes[:,2])>1e-10:
                raise ValueError('Loop chart requires one Burgers family and parallel xy planes.')
            if loop.core_tension!=0:
                raise ValueError('The shared density-core model replaces separate line tension.')
        source=self.base.mechanics.source(loops)
        a=np.einsum('i,ij...->j...',self.burgers,source['alpha_hat'])/(self.burgers@self.burgers)
        k=self.base.mechanics.k
        cross=np.cross(k,self.normal,axisa=0,axisb=0,axisc=0)
        kp2=np.sum(cross*cross,axis=0)
        hat=np.divide(1j*np.sum(cross*a,axis=0),kp2,out=np.zeros_like(kp2,dtype=complex),where=kp2>0)
        mask=kp2==0
        for loop in loops:
            factor=loop.burgers@self.burgers/(self.burgers@self.burgers)
            hat[mask]-=factor*loop.area[2]*np.exp(-loop.core**2*k[2][mask]**2/2)*np.exp(-1j*k[2][mask]*(loop.nodes[0,2]-self.base.mechanics.origin[2]))/self.grid.h**3
        return np.fft.ifftn(hat).real+history

    def tangent(self,loops):
        mechanics=self.base.mechanics;k=mechanics.k;result=[]
        for loop in loops:
            factor=loop.burgers@self.burgers/(self.burgers@self.burgers)
            columns=np.zeros((len(loop.nodes),2)+self.grid.shape,complex)
            smoothing=np.exp(-loop.core**2*mechanics.k2/2)/self.grid.h**3
            for index,(node,segment) in enumerate(zip(loop.nodes,loop.segments)):
                oriented=np.array([-segment[1],segment[0]])
                for s,weight in zip(mechanics.s,mechanics.weight):
                    point=node+s*segment-mechanics.origin
                    kernel=smoothing*np.exp(-1j*np.einsum('i...,i->...',k,point))
                    value=factor*weight*oriented[:,None,None,None]*kernel
                    columns[index]+=(1-s)*value
                    columns[(index+1)%len(loop.nodes)]+=s*value
            result.extend(np.fft.ifftn(columns,axes=(-3,-2,-1)).real.reshape((-1,)+self.grid.shape))
        return np.array(result)

    def chart_state(self,loops,details=None,amplitudes=None,history=0.):
        details=[] if details is None else details
        amplitudes=np.zeros(len(details)) if amplitudes is None else np.asarray(amplitudes)
        if len(details)!=len(amplitudes):raise ValueError('One amplitude per detail field is required.')
        gamma=self.gamma_from_loops(loops,history)
        for amplitude,detail in zip(amplitudes,details):gamma=gamma+amplitude*detail
        state=self.evaluate(gamma)
        columns=self.tangent(loops)
        if len(details):columns=np.concatenate((columns,np.asarray(details)),axis=0)
        projection=weighted_projection(columns,state['full_rate'],state['mobility'],self.grid.h**3)
        state.update({'projection':projection,'columns':columns,
                      'projected_energy_release':self.grid.integrate(state['conjugate']*projection['rate'])})
        return state

    def enrich(self,loops,details,amplitudes,budget=.05,max_details=3,history=0.):
        if budget<0 or max_details<0:raise ValueError('Nonnegative budget and detail cap required.')
        details=list(details);amplitudes=np.asarray(amplitudes).copy();added=0
        current=self.chart_state(loops,details,amplitudes,history)
        before=current['energy'];initial=current['projection']['relative_residual']
        while current['projection']['relative_residual']>budget and len(details)<max_details:
            p=current['projection']
            details.append(p['residual']/np.sqrt(p['discarded_norm_squared']))
            amplitudes=np.r_[amplitudes,0.];added+=1
            current=self.chart_state(loops,details,amplitudes,history)
        return details,amplitudes,current,{'initial_residual':initial,'final_residual':current['projection']['relative_residual'],
            'added_details':added,'energy_jump':current['energy']-before,
            'fallback_required':current['projection']['relative_residual']>budget,
            'detail_storage_bytes':sum(d.nbytes for d in details)}

    def chart_step(self,loops,details,amplitudes,dt,history=0.,current=None):
        if dt<=0:raise ValueError('Positive timestep required.')
        current=self.chart_state(loops,details,amplitudes,history) if current is None else current
        q=current['projection']['coordinates']
        for trials in range(1,31):
            candidate=[];offset=0
            for loop in loops:
                count=2*len(loop.nodes);rate=np.zeros_like(loop.nodes)
                rate[:,:2]=q[offset:offset+count].reshape(-1,2);offset+=count
                candidate.append(loop.moved(loop.nodes+dt*rate))
            coefficients=np.asarray(amplitudes)+dt*q[offset:]
            state=self.chart_state(candidate,details,coefficients,history)
            if state['energy']<=current['energy']-.1*dt*current['projection']['projected_norm_squared']+1e-12*max(1,abs(current['energy'])):
                return candidate,coefficients,state,dt,trials
            dt*=.5
        raise RuntimeError('Projected chart step failed energy descent.')

    def field_step(self,gamma,dt):
        if dt<=0:raise ValueError('Positive timestep required.')
        current=self.evaluate(gamma)
        for trials in range(1,31):
            candidate=gamma+dt*current['full_rate'];state=self.evaluate(candidate)
            if state['energy']<=current['energy']-.1*dt*current['dissipation']+1e-12*max(1,abs(current['energy'])):
                return candidate,state,dt,trials
            dt*=.5
        raise RuntimeError('Scalar transport step failed energy descent.')
