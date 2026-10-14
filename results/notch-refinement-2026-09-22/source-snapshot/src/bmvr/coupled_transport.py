"""Reciprocal phase/population/planar-dislocation evolution with exact block mechanics."""
from dataclasses import asdict
import json
import numpy as np
from .continuum3d import Cube,Parameters3D,PeriodicElasticity
from .response_cache import small_projection


class CoupledTransport:
    def __init__(self,grid,parameters=None,burgers=(.2,0,0),mean_strain=None,lam=2.,shear=1.,core_coefficient=.3,field_mobility=10.):
        self.grid=grid;self.p=parameters or Parameters3D();self.burgers=np.asarray(burgers,dtype=float)
        if self.burgers.shape!=(3,) or np.linalg.norm(self.burgers)==0 or abs(self.burgers[2])>1e-12:
            raise ValueError('A nonzero planar Burgers vector is required.')
        if core_coefficient<=0 or field_mobility<=0:raise ValueError('Positive core coefficient and mobility required.')
        self.core_coefficient=core_coefficient;self.field_mobility=field_mobility
        self.mean_strain=np.zeros((3,3)) if mean_strain is None else np.asarray(mean_strain,dtype=float)
        if self.mean_strain.shape!=(3,3) or not np.allclose(self.mean_strain,self.mean_strain.T,rtol=0,atol=1e-12):
            raise ValueError('Prescribed mean strain must be a symmetric 3x3 tensor.')
        p=self.p
        if min(p.W,*p.gradient,p.susceptibility,p.phase_mobility,p.diffusion_mobility)<=0:
            raise ValueError('Energy and kinetic coefficients must be positive.')
        self.elastic=PeriodicElasticity(grid,lam,shear);k=self.elastic.k;inv=self.elastic.inverse_k2
        normal=np.array([0.,0.,1.]);S=(np.outer(self.burgers,normal)+np.outer(normal,self.burgers))/2
        basis=np.array([np.eye(3),S]);stress=2*shear*basis+lam*np.trace(basis,axis1=1,axis2=2)[:,None,None]*np.eye(3)
        force=np.einsum('aij,j...->ai...',stress,k)
        kernel=np.zeros((2,2)+grid.shape)
        for a in range(2):
            for b in range(2):
                kernel[a,b]=np.sum(basis[a]*stress[b])-(np.sum(force[a]*force[b],axis=0)
                    -(lam+shear)/(lam+2*shear)*np.sum(k*force[a],axis=0)*np.sum(k*force[b],axis=0)*inv)*inv/shear
        cross=np.cross(k,normal,axisa=0,axisb=0,axisc=0)
        kernel[1,1]+=core_coefficient*(self.burgers@self.burgers)*np.sum(cross*cross,axis=0)
        self.kernel=kernel;self.gamma_multipliers=np.concatenate((kernel[:,1],-1j*cross[:2]),axis=0)
        mean_stress=2*shear*self.mean_strain+lam*np.trace(self.mean_strain)*np.eye(3)
        self.loading=np.einsum('ij,aij->a',mean_stress,basis)
        self.loading_energy=grid.volume*np.sum(self.mean_strain*mean_stress)/2

    def response_gamma(self,gamma):
        return np.fft.ifftn(self.gamma_multipliers*np.fft.fftn(gamma),axes=(-3,-2,-1)).real

    def evaluate(self,v,c,gamma,gamma_response=None,d_response=None):
        p=self.p;g=self.grid;dc=c-p.reference_population
        d=p.phase_misfit*(v+1)/2+p.solute_expansion*dc
        rd=np.fft.ifftn(self.kernel[:,0]*np.fft.fftn(d),axes=(-3,-2,-1)).real if d_response is None else d_response
        rg=self.response_gamma(gamma) if gamma_response is None else gamma_response
        image=rd+rg[:2];hd=self.loading[0]-image[0];hg=self.loading[1]-image[1]
        gv=p.W*(v**3-v)-g.lap(v,p.gradient)+2*p.binding*dc*v-p.phase_misfit*hd/2
        mu=p.susceptibility*dc-p.binding*(1-v*v)-p.solute_expansion*hd
        K=self.field_mobility*np.sum(rg[2:]**2,axis=0)
        rates=(-p.phase_mobility*gv,p.diffusion_mobility*g.lap(mu),K*hg)
        components={'structural':g.integrate(p.W*(1-v*v)**2/4),'chemical':g.integrate(p.susceptibility*dc*dc/2),
            'binding':g.integrate(-p.binding*dc*(1-v*v)),'interface':g.gradient_energy(v,p.gradient),
            'mechanical_and_core':self.loading_energy-g.integrate(self.loading[0]*d+self.loading[1]*gamma)+g.integrate(d*image[0]+gamma*image[1])/2}
        gamma_dissipation=g.integrate(K*hg*hg)
        field_dissipation=p.phase_mobility*g.integrate(gv*gv)+2*p.diffusion_mobility*g.gradient_energy(mu)
        return {'energy':sum(components.values()),'components':components,'rates':rates,'forces':(gv,mu,-hg),
            'gamma_conjugate':hg,'gamma_mobility':K,'gamma_dissipation':gamma_dissipation,
            'field_dissipation':field_dissipation,'dissipation':field_dissipation+gamma_dissipation,
            'trace_stress':hd,'gamma':gamma,'gamma_response':rg}

    def step(self,fields,dt,current=None,freeze_gamma=False,freeze_fields=False):
        if dt<=0:raise ValueError('Positive timestep required.')
        current=self.evaluate(*fields) if current is None else current
        rates=list(current['rates'])
        if freeze_gamma:rates[2]=np.zeros_like(rates[2])
        if freeze_fields:rates[:2]=[np.zeros_like(r) for r in rates[:2]]
        dissipation=(0 if freeze_fields else current['field_dissipation'])+(0 if freeze_gamma else current['gamma_dissipation'])
        for trials in range(1,31):
            candidate=tuple(f+dt*r for f,r in zip(fields,rates))
            if candidate[1].min()>0 and candidate[1].max()<1:
                following=self.evaluate(*candidate)
                if following['energy']<=current['energy']-.1*dt*dissipation+1e-12*max(1.,abs(current['energy'])):
                    return candidate,following,dt,trials
            dt*=.5
        raise RuntimeError('Coupled transport step failed positivity or energy descent.')

    def tensor_readback(self,v,c,gamma):
        p=self.p;normal=np.array([0.,0.,1.]);beta=np.outer(self.burgers,normal)[...,None,None,None]*gamma
        d=p.phase_misfit*(v+1)/2+p.solute_expansion*(c-p.reference_population)
        elastic=self.elastic.solve((beta+beta.swapaxes(0,1))/2+np.eye(3)[...,None,None,None]*d,self.mean_strain)
        derivatives=self.response_gamma(gamma)[2:]
        alpha=np.zeros_like(beta);alpha[:,:2]=self.burgers[:,None,None,None,None]*derivatives[None]
        core=self.core_coefficient*self.grid.integrate(alpha*alpha)/2
        div=np.einsum('j...,ij...->i...',self.elastic.k,np.fft.fftn(alpha,axes=(-3,-2,-1)))
        return {**elastic,'alpha':alpha,'core_energy':core,'plastic_trace':float(np.trace(beta.mean(axis=(2,3,4)))),
            'divergence_relative':float(np.linalg.norm(div)/max(np.linalg.norm(np.fft.fftn(alpha,axes=(-3,-2,-1))),1e-30))}


class CoupledGammaCache:
    def __init__(self,model,gamma):
        self.model=model;self.base_gamma=gamma.copy();self.base_response=model.response_gamma(gamma)
        self.modes=np.zeros((0,)+model.grid.shape);self.responses=np.zeros((0,4)+model.grid.shape)

    def state(self,v,c,coefficients,project=True):
        gamma=self.base_gamma+np.einsum('i,i...->...',coefficients,self.modes)
        response=self.base_response+np.einsum('i,i...->...',coefficients,self.responses)
        state=self.model.evaluate(v,c,gamma,response)
        if project:
            if len(coefficients):p=small_projection(self.modes,state['rates'][2],state['gamma_mobility'],self.model.grid.h**3,state['gamma_conjugate'])
            else:
                norm=state['gamma_dissipation'];p={'coordinates':np.array([]),'residual':state['rates'][2],
                    'discarded_norm_squared':norm,'projected_norm_squared':0.,'relative_residual':float(np.sqrt(norm/max(norm,1e-30)))}
            state['projection']=p
        return state

    def enrich(self,v,c,coefficients,budget=.05,max_modes=4):
        a=np.asarray(coefficients).copy();state=self.state(v,c,a);before=state['energy'];initial=state['projection']['relative_residual'];added=0
        while state['projection']['relative_residual']>budget and len(a)<max_modes:
            p=state['projection'];mode=p['residual']/np.sqrt(p['discarded_norm_squared'])
            self.modes=np.concatenate((self.modes,mode[None]),axis=0)
            self.responses=np.concatenate((self.responses,self.model.response_gamma(mode)[None]),axis=0)
            a=np.r_[a,0.];added+=1;state=self.state(v,c,a)
        return a,state,{'initial_residual':initial,'final_residual':state['projection']['relative_residual'],'added_modes':added,
            'energy_jump':state['energy']-before,'fallback_required':state['projection']['relative_residual']>budget}

    def step(self,v,c,coefficients,dt,current=None):
        if dt<=0:raise ValueError('Positive timestep required.')
        state=self.state(v,c,coefficients) if current is None else current
        dissipation=state['field_dissipation']+state['projection']['projected_norm_squared']
        for trials in range(1,31):
            nv=v+dt*state['rates'][0];nc=c+dt*state['rates'][1];a=coefficients+dt*state['projection']['coordinates']
            if nc.min()>0 and nc.max()<1:
                following=self.state(nv,nc,a,project=False)
                if following['energy']<=state['energy']-.1*dt*dissipation+1e-12*max(1.,abs(state['energy'])):
                    return nv,nc,a,following,dt,trials
            dt*=.5
        raise RuntimeError('Coupled cached step failed positivity or energy descent.')

    def save(self,path,v,c,coefficients,gamma,mode,time,step,budget=.05,max_modes=4):
        m=self.model
        metadata={'length':m.grid.length,'n':m.grid.n,'parameters':asdict(m.p),'burgers':m.burgers.tolist(),
            'mean_strain':m.mean_strain.tolist(),'lam':m.elastic.lam,'shear':m.elastic.shear,
            'core_coefficient':m.core_coefficient,'field_mobility':m.field_mobility,'mode':mode,'time':time,'step':step,
            'budget':budget,'max_modes':max_modes}
        np.savez_compressed(path,v=v,c=c,coefficients=coefficients,gamma=gamma,base_gamma=self.base_gamma,modes=self.modes,metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            p=json.loads(str(data['metadata']));grid=Cube(p.pop('length'),p.pop('n'));parameters=Parameters3D(**p.pop('parameters'))
            clock={key:p.pop(key) for key in ('mode','time','step','budget','max_modes')}
            model=CoupledTransport(grid,parameters=parameters,**p);cache=cls(model,data['base_gamma'].copy())
            cache.modes=data['modes'].copy();cache.responses=np.array([model.response_gamma(e) for e in cache.modes]).reshape((-1,4)+grid.shape)
            return cache,data['v'].copy(),data['c'].copy(),data['coefficients'].copy(),data['gamma'].copy(),clock
