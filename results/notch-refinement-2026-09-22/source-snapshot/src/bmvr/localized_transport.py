"""Localized, conservative trial spaces for all three Section 33 fields.

Fine-grid quadrature/audits and dense response storage are retained and charged.
No physical mobility or interface width is modified by the reduction.
"""
from dataclasses import asdict
import json
import numpy as np
from .continuum3d import Cube,Parameters3D
from .coupled_transport import CoupledTransport
from .response_cache import small_projection


class DiffusionMetric:
    def __init__(self,grid,mobility):
        self.grid=grid;self.mobility=mobility
        wave=2*np.pi*np.fft.fftfreq(grid.n)
        self.symbol=4*mobility/grid.h**2*sum(np.sin(a/2)**2 for a in np.meshgrid(wave,wave,wave,indexing='ij'))

    def inverse(self,rate):
        rate=np.asarray(rate)
        if abs(float(np.mean(rate)))>1e-12*max(float(np.max(abs(rate))),1e-30):
            raise ValueError('Diffusion metric requires an inventory-conserving zero-mean rate.')
        transformed=np.fft.fftn(rate)
        return np.fft.ifftn(np.divide(transformed,self.symbol,out=np.zeros_like(transformed),where=self.symbol>0)).real

    def localized(self,potential,mask):
        result=np.zeros_like(potential)
        for axis in range(3):
            flux=mask*(np.roll(potential,-1,axis)-potential)/self.grid.h
            result-=self.mobility*(flux-np.roll(flux,1,axis))/self.grid.h
        return result


def constant_projection(modes,metric_modes,force,rate,metric_rate,volume,gram=None):
    """Project with a constant positive metric on its admissible subspace."""
    full=float(volume*np.sum(rate*metric_rate));coefficients=np.zeros(len(modes))
    if len(modes):
        E=modes.reshape(len(modes),-1);RE=metric_modes.reshape(len(modes),-1)
        G=volume*(E@RE.T) if gram is None else gram
        G=(G+G.T)/2
        scale=np.sqrt(np.maximum(np.diag(G),1e-300));scaled=G/scale[:,None]/scale[None,:]
        eigen,vectors=np.linalg.eigh(scaled);keep=eigen>max(eigen[-1]*1e-11,0.)
        rhs=-volume*(E@force.ravel())/scale
        coefficients=(vectors[:,keep]@((vectors[:,keep].T@rhs)/eigen[keep]))/scale
    projected=np.einsum('i,i...->...',coefficients,modes)
    metric_projected=np.einsum('i,i...->...',coefficients,metric_modes)
    residual=rate-projected;metric_residual=metric_rate-metric_projected
    kept=float(volume*np.sum(projected*metric_projected));discarded=max(0.,float(volume*np.sum(residual*metric_residual)))
    return {'coordinates':coefficients,'rate':projected,'residual':residual,'metric_residual':metric_residual,
        'full_norm_squared':full,'discarded_norm_squared':discarded,'projected_norm_squared':kept,
        'relative_residual':float(np.sqrt(discarded/max(full,1e-30))),
        'orthogonality':float(volume*np.sum(projected*metric_residual)),
        'pythagorean_error':abs(full-kept-discarded)}


class LocalizedTransport:
    names=('phase','population','gamma')

    def __init__(self,model,fields,budgets=(.05,.05,.05),caps=(32,32,32),reset_at_cap=False):
        self.model=model;self.bases=[np.array(f,copy=True) for f in fields]
        self.budgets=tuple(budgets);self.caps=tuple(caps)
        self.reset_at_cap=bool(reset_at_cap)
        if len(budgets)!=3 or len(caps)!=3 or min(budgets)<0 or min(caps)<0:
            raise ValueError('Three nonnegative budgets and caps are required.')
        self.modes=[np.zeros((0,)+model.grid.shape) for _ in range(3)]
        self.metric_modes=[np.zeros_like(a) for a in self.modes]
        self.grams=[np.zeros((0,0)) for _ in range(3)]
        self.responses=[np.zeros((0,2 if b<2 else 4)+model.grid.shape) for b in range(3)]
        self.base_responses=[self.response(b,f,base=True) for b,f in enumerate(self.bases)]
        self.coefficients=[np.array([]) for _ in range(3)];self.full=[False]*3
        self.metric=DiffusionMetric(model.grid,model.p.diffusion_mobility)
        x,y,z=model.grid.coordinates();labels=(x>=0).astype(int)*4+(y>=0).astype(int)*2+(z>=0).astype(int)
        self.masks=np.array([labels==i for i in range(8)])
        self.patch_history=[[] for _ in range(3)];self.time=0.;self.steps=0

    def response(self,block,field,base=False):
        m=self.model
        if block==2:return m.response_gamma(field)
        source=m.p.phase_misfit*(field+(1 if base else 0))/2 if block==0 else m.p.solute_expansion*(field-(m.p.reference_population if base else 0))
        return np.fft.ifftn(m.kernel[:,0]*np.fft.fftn(source),axes=(-3,-2,-1)).real

    def inverse(self,block,rate,state):
        if block==0:return rate/self.model.p.phase_mobility
        if block==1:return self.metric.inverse(rate)
        K=state['gamma_mobility']
        return np.divide(rate,K,out=np.zeros_like(rate),where=K>0)

    def state(self,coefficients=None,bases=None,project=True):
        coefficients=self.coefficients if coefficients is None else coefficients
        bases=self.bases if bases is None else bases
        fields=[];responses=[]
        for b in range(3):
            if self.full[b]:
                fields.append(bases[b]);responses.append(self.response(b,bases[b],base=True))
            else:
                fields.append(bases[b]+np.einsum('i,i...->...',coefficients[b],self.modes[b]))
                responses.append(self.base_responses[b]+np.einsum('i,i...->...',coefficients[b],self.responses[b]))
        state=self.model.evaluate(*fields,gamma_response=responses[2],d_response=responses[0]+responses[1])
        state['fields']=tuple(fields)
        if project:
            projections=[]
            for b,(force,rate) in enumerate(zip(state['forces'],state['rates'])):
                if self.full[b]:
                    norm=-self.model.grid.integrate(force*rate)
                    projections.append({'rate':rate,'residual':np.zeros_like(rate),'relative_residual':0.,
                        'full_norm_squared':norm,'projected_norm_squared':norm,'discarded_norm_squared':0.})
                    continue
                if b<2:
                    metric_rate=-force if b==0 else -force+np.mean(force)
                    p=constant_projection(self.modes[b],self.metric_modes[b],force,rate,metric_rate,self.model.grid.h**3,self.grams[b])
                elif len(self.modes[b]):
                    p=small_projection(self.modes[b],rate,state['gamma_mobility'],self.model.grid.h**3,-force)
                    p['metric_residual']=self.inverse(b,p['residual'],state)
                else:
                    norm=state['gamma_dissipation'];p={'coordinates':np.array([]),'rate':np.zeros_like(rate),'residual':rate,
                        'metric_residual':self.inverse(b,rate,state),'full_norm_squared':norm,'discarded_norm_squared':norm,
                        'projected_norm_squared':0.,'relative_residual':float(np.sqrt(norm/max(norm,1e-30)))}
                projections.append(p)
            state['projections']=projections
        return state

    def candidates(self,block,state):
        p=state['projections'][block]
        if block==1:return np.array([self.metric.localized(p['metric_residual'],mask) for mask in self.masks])
        return self.masks*p['residual']

    def activate(self):
        state=self.state();events=[]
        for b in range(3):
            reset_used=False
            while not self.full[b] and state['projections'][b]['relative_residual']>self.budgets[b]:
                p=state['projections'][b];before=state['energy'];fields=state['fields']
                if len(self.modes[b])>=self.caps[b]:
                    if self.reset_at_cap and self.caps[b]>0 and not reset_used:
                        self.reset(b);reset_used=True;state=self.state()
                        events.append({'block':self.names[b],'action':'reset','time':self.time,
                            'energy_jump':state['energy']-before,'residual_before':p['relative_residual'],
                            'state_jump_max':max(float(np.max(abs(a-c))) for a,c in zip(fields,state['fields']))})
                        continue
                    self.bases[b]=fields[b].copy();self.full[b]=True
                    state=self.state()
                    events.append({'block':self.names[b],'action':'promote','time':self.time,
                        'energy_jump':state['energy']-before,'residual_before':p['relative_residual']})
                    break
                choices=self.candidates(b,state);best=None
                for patch,direction in enumerate(choices):
                    inverse=self.inverse(b,direction,state);norm=self.model.grid.integrate(direction*inverse)
                    correlation=self.model.grid.integrate(direction*p['metric_residual'])
                    score=correlation*correlation/norm if norm>0 else 0.
                    if best is None or score>best[0]:best=(score,patch,direction,inverse,norm)
                if best[0]<=0:raise RuntimeError('No admissible residual direction despite a failed budget.')
                _,patch,direction,inverse,norm=best;mode=direction/np.sqrt(norm)
                self.modes[b]=np.concatenate((self.modes[b],mode[None]))
                if b<2:
                    self.metric_modes[b]=np.concatenate((self.metric_modes[b],self.inverse(b,mode,state)[None]))
                    self.grams[b]=self.model.grid.h**3*(self.modes[b].reshape(len(self.modes[b]),-1)@self.metric_modes[b].reshape(len(self.modes[b]),-1).T)
                self.responses[b]=np.concatenate((self.responses[b],self.response(b,mode)[None]))
                self.coefficients[b]=np.r_[self.coefficients[b],0.];self.patch_history[b].append(patch)
                state=self.state()
                events.append({'block':self.names[b],'action':'activate','patch':patch,'time':self.time,
                    'energy_jump':state['energy']-before,'residual_before':p['relative_residual'],
                    'residual_after':state['projections'][b]['relative_residual']})
        return state,events

    def step(self,dt,current=None):
        if dt<=0:raise ValueError('Positive timestep required.')
        state=self.state() if current is None else current
        if any(p['relative_residual']>budget for p,budget in zip(state['projections'],self.budgets)):
            raise ValueError('Activate or promote failed blocks before evolving.')
        dissipation=sum(p['projected_norm_squared'] for p in state['projections'])
        for trials in range(1,31):
            coefficients=[q.copy() for q in self.coefficients];bases=list(self.bases)
            for b in range(3):
                if self.full[b]:bases[b]=bases[b]+dt*state['rates'][b]
                else:coefficients[b]+=dt*state['projections'][b]['coordinates']
            following=self.state(coefficients,bases,project=False)
            c=following['fields'][1]
            if c.min()>0 and c.max()<1 and following['energy']<=state['energy']-.1*dt*dissipation+1e-12*max(1.,abs(state['energy'])):
                self.coefficients=coefficients;self.bases=bases;self.time+=dt;self.steps+=1
                return following,dt,trials
            dt*=.5
        raise RuntimeError('Localized coupled step failed positivity or energy descent.')

    def reset(self,block):
        state=self.state(project=False);self.bases[block]=state['fields'][block].copy()
        self.base_responses[block]=self.response(block,self.bases[block],base=True)
        self.modes[block]=np.empty((0,)+self.model.grid.shape);self.metric_modes[block]=np.empty_like(self.modes[block])
        self.responses[block]=np.empty((0,2 if block<2 else 4)+self.model.grid.shape);self.coefficients[block]=np.array([])
        self.grams[block]=np.zeros((0,0))
        self.patch_history[block]=[];self.full[block]=False

    def save(self,path):
        m=self.model;metadata={'length':m.grid.length,'n':m.grid.n,'parameters':asdict(m.p),'burgers':m.burgers.tolist(),
            'mean_strain':m.mean_strain.tolist(),'lam':m.elastic.lam,'shear':m.elastic.shear,
            'core_coefficient':m.core_coefficient,'field_mobility':m.field_mobility,
            'budgets':self.budgets,'caps':self.caps,'full':self.full,'time':self.time,'steps':self.steps,'patch_history':self.patch_history,
            'reset_at_cap':self.reset_at_cap}
        arrays={f'{key}_{b}':value[b] for key,value in [('base',self.bases),('modes',self.modes),('coefficients',self.coefficients)] for b in range(3)}
        np.savez_compressed(path,**arrays,metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            p=json.loads(str(data['metadata']));grid=Cube(p.pop('length'),p.pop('n'));parameters=Parameters3D(**p.pop('parameters'))
            controls={key:p.pop(key) for key in ('budgets','caps','full','time','steps','patch_history')}
            reset_at_cap=p.pop('reset_at_cap',False)
            obj=cls(CoupledTransport(grid,parameters=parameters,**p),[data[f'base_{b}'].copy() for b in range(3)],controls['budgets'],controls['caps'],reset_at_cap)
            obj.full=controls['full'];obj.time=controls['time'];obj.steps=controls['steps'];obj.patch_history=controls['patch_history']
            for b in range(3):
                obj.modes[b]=data[f'modes_{b}'].copy();obj.coefficients[b]=data[f'coefficients_{b}'].copy()
                obj.responses[b]=np.array([obj.response(b,e) for e in obj.modes[b]]).reshape((-1,2 if b<2 else 4)+grid.shape)
                if b<2 and len(obj.modes[b]):
                    obj.metric_modes[b]=np.array([obj.inverse(b,e,None) for e in obj.modes[b]])
                    obj.grams[b]=grid.h**3*(obj.modes[b].reshape(len(obj.modes[b]),-1)@obj.metric_modes[b].reshape(len(obj.modes[b]),-1).T)
            return obj
