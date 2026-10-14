"""Exact homogeneous scalar responses and budgeted affine transport reduction."""
import json
import numpy as np
from .continuum3d import Cube
from .transport_metric import PlanarTransport,weighted_projection


def small_projection(modes,rate,mobility,volume,conjugate):
    """Equivalent small Gram solve, retaining the constrained SVD for hard cases."""
    E=modes.reshape(len(modes),-1);K=mobility.ravel();active=K>0
    if np.any(E[:,~active]):return weighted_projection(modes,rate,mobility,volume)
    weights=np.divide(volume,K,out=np.zeros_like(K),where=active)
    G=(E*weights)@E.T;diagonal=np.sqrt(np.diag(G))
    if np.any(diagonal==0):return weighted_projection(modes,rate,mobility,volume)
    scaled=G/diagonal[:,None]/diagonal[None,:];spectrum=np.linalg.eigvalsh(scaled)
    if spectrum[0]<=0 or spectrum[-1]/spectrum[0]>=1e8:
        return weighted_projection(modes,rate,mobility,volume)
    rhs=volume*(E@conjugate.ravel());chol=np.linalg.cholesky(scaled)
    q=np.linalg.solve(chol.T,np.linalg.solve(chol,rhs/diagonal))/diagonal
    projected=q@E;residual=rate.ravel()-projected
    full_norm=volume*float(np.sum(mobility*conjugate**2))
    discarded=float(np.sum(weights*residual**2));kept=float(np.sum(weights*projected**2))
    return {'coordinates':q,'rate':projected.reshape(rate.shape),'residual':residual.reshape(rate.shape),
        'full_norm_squared':full_norm,'discarded_norm_squared':discarded,'projected_norm_squared':kept,
        'relative_residual':float(np.sqrt(discarded/max(full_norm,1e-30))),
        'orthogonality':float(np.sum(weights*projected*residual)),
        'pythagorean_error':abs(full_norm-kept-discarded),'solver':'scaled_small_cholesky'}


class ScalarTransportOperator:
    def __init__(self,model):
        self.grid=model.grid;self.burgers=model.burgers.copy();base=model.base
        self.mobility=float(base.field_mobility)
        lam,shear=base.mechanics.elastic.lam,base.mechanics.elastic.shear
        core=base.core_coefficient;k=base.mechanics.k
        S=(np.outer(self.burgers,model.normal)+np.outer(model.normal,self.burgers))/2
        T=2*shear*S+lam*np.trace(S)*np.eye(3)
        fixed=float(np.sum(S*T));k2=np.sum(k*k,axis=0)
        force=np.einsum('ij,j...->i...',T,k)
        inv=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
        contraction=(np.sum(force*force,axis=0)-(lam+shear)/(lam+2*shear)*np.sum(k*force,axis=0)**2*inv)*inv/shear
        cross=np.cross(k,model.normal,axisa=0,axisb=0,axisc=0)
        kp2=np.sum(cross*cross,axis=0)
        self.symbol=fixed-contraction+core*(self.burgers@self.burgers)*kp2
        self.symbol[(k2>0)&(kp2==0)]=0.
        self.multipliers=np.array([self.symbol,-1j*cross[0],-1j*cross[1]])
        zero=model.evaluate(np.zeros(self.grid.shape))
        self.background_force=zero['conjugate'].copy();self.background_energy=zero['energy']
        self.background=base.background.copy()
        self.configuration={'length':self.grid.length,'n':self.grid.n,'burgers':self.burgers.tolist(),
            'core_coefficient':core,'field_mobility':self.mobility,'mean_strain':base.mean_strain.tolist(),
            'lam':lam,'shear':shear,'quadrature_order':base.mechanics.quadrature_order}
        self.response_calls=0

    def response(self,direction):
        self.response_calls+=1
        transformed=np.fft.fftn(direction)
        values=np.fft.ifftn(self.multipliers*transformed,axes=(-3,-2,-1)).real
        return values[0],values[1:]

    def assemble(self,gamma,h,a,energy):
        mobility=self.mobility*np.sum(a*a,axis=0);rate=mobility*h
        return {'gamma':gamma,'conjugate':h,'a':a,'energy':float(energy),'mobility':mobility,
                'full_rate':rate,'dissipation':self.grid.integrate(h*rate)}

    def evaluate(self,gamma):
        image,a=self.response(gamma);h=self.background_force-image
        energy=self.background_energy-self.grid.integrate(self.background_force*gamma)+self.grid.integrate(gamma*image)/2
        return self.assemble(np.array(gamma,copy=True),h,a,energy)

    def step(self,current,dt):
        if dt<=0:raise ValueError('Positive timestep required.')
        rate=current['full_rate'];image,a_rate=self.response(rate)
        decrease=current['dissipation'];curvature=self.grid.integrate(rate*image)
        for trials in range(1,31):
            energy=current['energy']-dt*decrease+dt*dt*curvature/2
            if energy<=current['energy']-.1*dt*decrease+1e-12*max(1.,abs(current['energy'])):
                return self.assemble(current['gamma']+dt*rate,current['conjugate']-dt*image,current['a']+dt*a_rate,energy),dt,trials
            dt*=.5
        raise RuntimeError('Optimized scalar step failed energy descent.')


class AffineTransportCache:
    def __init__(self,operator,gamma):
        self.operator=operator;self.base=operator.evaluate(gamma)
        self.modes=np.zeros((0,)+operator.grid.shape)
        self.images=np.zeros_like(self.modes);self.derivatives=np.zeros((0,2)+operator.grid.shape)
        self.linear=np.array([]);self.stiffness=np.zeros((0,0))

    @property
    def storage_bytes(self):
        return sum(x.nbytes for x in (self.modes,self.images,self.derivatives,self.linear,self.stiffness,
                                      self.base['gamma'],self.base['conjugate'],self.base['a']))

    def add(self,direction):
        image,derivative=self.operator.response(direction)
        count=len(self.modes);G=np.zeros((count+1,count+1));G[:count,:count]=self.stiffness
        for i in range(count):
            G[i,count]=G[count,i]=self.operator.grid.integrate(self.modes[i]*image)
        G[count,count]=self.operator.grid.integrate(direction*image)
        self.stiffness=G
        self.linear=np.r_[self.linear,self.operator.grid.integrate(direction*self.base['conjugate'])]
        self.modes=np.concatenate((self.modes,direction[None]),axis=0)
        self.images=np.concatenate((self.images,image[None]),axis=0)
        self.derivatives=np.concatenate((self.derivatives,derivative[None]),axis=0)

    def state(self,coefficients,project=True):
        c=np.asarray(coefficients)
        if len(c)!=len(self.modes):raise ValueError('One coefficient per cached direction is required.')
        if len(c):
            gamma=self.base['gamma']+np.einsum('i,i...->...',c,self.modes)
            h=self.base['conjugate']-np.einsum('i,i...->...',c,self.images)
            a=self.base['a']+np.einsum('i,i...->...',c,self.derivatives)
        else:gamma,h,a=self.base['gamma'],self.base['conjugate'],self.base['a']
        energy=self.base['energy']-self.linear@c+c@self.stiffness@c/2
        state=self.operator.assemble(gamma,h,a,energy)
        if project:
            if len(c):p=small_projection(self.modes,state['full_rate'],state['mobility'],self.operator.grid.h**3,state['conjugate'])
            else:
                norm=state['dissipation']
                p={'coordinates':np.array([]),'rate':np.zeros_like(gamma),'residual':state['full_rate'].copy(),
                   'relative_residual':float(np.sqrt(norm/max(norm,1e-30))),
                   'full_norm_squared':norm,'discarded_norm_squared':norm,'projected_norm_squared':0.}
            state['projection']=p
        return state

    def enrich(self,coefficients,budget=.05,max_modes=3):
        if budget<0 or max_modes<0:raise ValueError('Nonnegative budget and mode cap required.')
        c=np.asarray(coefficients).copy();state=self.state(c);before=state['energy'];initial=state['projection']['relative_residual'];added=0
        while state['projection']['relative_residual']>budget and len(c)<max_modes:
            p=state['projection'];self.add(p['residual']/np.sqrt(p['discarded_norm_squared']))
            c=np.r_[c,0.];added+=1;state=self.state(c)
        return c,state,{'initial_residual':initial,'final_residual':state['projection']['relative_residual'],
            'added_modes':added,'energy_jump':state['energy']-before,'fallback_required':state['projection']['relative_residual']>budget}

    def step(self,coefficients,current,dt):
        if dt<=0:raise ValueError('Positive timestep required.')
        q=current['projection']['coordinates'];decrease=current['projection']['projected_norm_squared'];curvature=q@self.stiffness@q
        for trials in range(1,31):
            predicted=current['energy']-dt*decrease+dt*dt*curvature/2
            if predicted<=current['energy']-.1*dt*decrease+1e-12*max(1.,abs(current['energy'])):
                c=coefficients+dt*q
                return c,self.state(c,project=False),dt,trials
            dt*=.5
        raise RuntimeError('Affine cached step failed energy descent.')

    def save(self,path,coefficients,state,mode,time,step,budget=.05,max_modes=3):
        metadata={'configuration':self.operator.configuration,'mode':mode,'time':time,'step':step,'budget':budget,'max_modes':max_modes}
        np.savez_compressed(path,base_gamma=self.base['gamma'],modes=self.modes,coefficients=coefficients,
                            gamma=state['gamma'],conjugate=state['conjugate'],a=state['a'],energy=state['energy'],
                            background=self.operator.background,metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            metadata=json.loads(str(data['metadata']));p=metadata.pop('configuration');grid=Cube(p.pop('length'),p.pop('n'))
            model=PlanarTransport(grid,background=data['background'].copy(),**p)
            cache=cls(ScalarTransportOperator(model),data['base_gamma'].copy())
            for direction in data['modes']:cache.add(direction)
            c=data['coefficients'].copy()
            state=cache.state(c,project=False) if metadata['mode']=='affine' else cache.operator.assemble(data['gamma'].copy(),data['conjugate'].copy(),data['a'].copy(),float(data['energy']))
            if metadata['mode']=='affine' and not np.array_equal(state['gamma'],data['gamma']):raise ValueError('Affine checkpoint state mismatch.')
            return cache,c,state,metadata
