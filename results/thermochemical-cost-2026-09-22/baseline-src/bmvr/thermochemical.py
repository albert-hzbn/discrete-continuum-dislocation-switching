"""Section 39 chemical/phase evolution with local thermomechanical ledgers."""
from dataclasses import asdict, dataclass
import json
import numpy as np
from .finite_elasticity import FiniteParameters
from .thermoplastic import ThermalPlastic2D, ThermalPlasticState, ThermalPlasticParameters


@dataclass(frozen=True)
class ChemicalParameters:
    mobility: float = .02
    phase_mobility: float = .5
    kappa_c: float = .002
    kappa_eta: float = .0032
    partition: float = -.15
    mobility_kind: str = 'availability'

    def __post_init__(self):
        for key in ('mobility','phase_mobility','kappa_c','kappa_eta','partition'):
            v=getattr(self,key)
            if not np.isfinite(v) or (key!='partition' and v<0):
                raise ValueError(f'Invalid chemical coefficient {key}.')
        if self.mobility_kind not in ('availability','constant'):
            raise ValueError('Unknown chemical mobility.')


@dataclass
class ThermalChemicalState(ThermalPlasticState):
    c: np.ndarray
    eta: np.ndarray


class ThermalChemical2D(ThermalPlastic2D):
    def __init__(self,n,length=1.,mechanical=None,transport=None,chemistry=None,fixed=None,values=None,degree=1):
        super().__init__(n,length,mechanical,transport,fixed=fixed,values=values,degree=degree)
        self.chemistry=chemistry or ChemicalParameters()
        nodes=np.arange(n*n).reshape(n,n)
        self.edge_i=np.r_[nodes[:-1,:].ravel(),nodes[:,:-1].ravel()]
        self.edge_j=np.r_[nodes[1:,:].ravel(),nodes[:,1:].ravel()]

    def sample_cell(self,value):
        m=self.mechanics.mesh
        if callable(value):value=value(*m.coordinates[m.cells].mean(axis=1).T)
        return np.broadcast_to(value,(m.n*m.n,)).astype(float,copy=True)

    def initial(self,T=1.,u=None,equilibrate=False,c=.3,eta=.3):
        m=self.mechanics.mesh
        state=ThermalChemicalState(np.zeros(2*m.node_count) if u is None else np.array(u,dtype=float,copy=True),
            np.zeros(2*m.node_count),self.sample_cell(T),
            np.broadcast_to(np.eye(3),(m.n*m.n,m.quadrature_count,3,3)).copy(),np.zeros((m.n*m.n,m.quadrature_count,2)),
            self.sample_cell(c),self.sample_cell(eta))
        state.u[self.fixed]=self.values;self.install(state)
        if equilibrate:state.u,_,_=self.mechanics.equilibrate(state.u,self.fixed,self.values)
        self.rate(state)
        return state

    def install(self,state):
        m=self.mechanics.mesh
        for name in ('c','eta'):
            value=getattr(state,name)
            if value.shape!=(m.n*m.n,) or not np.isfinite(value).all() or np.any((value<0)|(value>1)):
                raise ValueError(f'Invalid bounded {name} state.')
            setattr(self.mechanics,name,np.broadcast_to(value[:,None],(m.n*m.n,m.quadrature_count)))
        super().install(state)

    def edge_sum(self,values,opposite=True):
        result=np.zeros(self.mechanics.mesh.n**2)
        np.add.at(result,self.edge_i,values)
        np.add.at(result,self.edge_j,-values if opposite else values)
        return result

    def gradient(self,z,coefficient):
        delta=z[self.edge_j]-z[self.edge_i]
        cell_energy=self.edge_sum(coefficient*delta**2/4,False)
        force=self.edge_sum(-coefficient*delta)/self.mechanics.mesh.h**2
        return cell_energy,force

    def rate(self,state):
        rate=super().rate(state);m=self.mechanics.mesh;p=self.mechanics.p;ch=self.chemistry;V=m.h**2
        i,j=self.edge_i,self.edge_j;h=state.eta**2*(3-2*state.eta);hp=6*state.eta*(1-state.eta)
        Gc,gradc=self.gradient(state.c,ch.kappa_c);Ge,grade=self.gradient(state.eta,ch.kappa_eta)
        local=rate['local']
        mu=np.sum(m.weights*local['gc'],axis=1)/V+ch.partition*h+gradc
        g=np.sum(m.weights*local['geta'],axis=1)/V+ch.partition*(state.c-p.c0)*hp+grade
        delta=mu[i]-mu[j]
        availability=1. if ch.mobility_kind=='constant' else np.where(delta>=0,state.c[i]*(1-state.c[j]),state.c[j]*(1-state.c[i]))
        flow=ch.mobility*availability*delta
        cdot=self.edge_sum(-flow)/V
        etadot=-ch.phase_mobility*g
        etadot=np.where(state.eta==0,np.maximum(etadot,0),etadot)
        etadot=np.where(state.eta==1,np.minimum(etadot,0),etadot)
        Dc=self.edge_sum(flow*delta/2,False);De=-V*g*etadot
        factor=2*p.mu*rate['elastic_trace']+9*p.lam
        mixed=p.alpha_T*factor*(p.alpha_c*cdot+p.alpha_eta*hp*etadot)[:,None]
        rate['T']+=(Dc+De+state.T*np.sum(m.weights*mixed,axis=1))/rate['capacity']
        partition=V*ch.partition*(state.c-p.c0)*h
        extra=Gc+Ge+partition
        rate['cell_internal_energy']+=extra
        rate['chemical_power']=self.edge_sum(-(mu[i]+mu[j])*flow/2)
        micro=(ch.kappa_c*(state.c[j]-state.c[i])*(cdot[i]+cdot[j])
               +ch.kappa_eta*(state.eta[j]-state.eta[i])*(etadot[i]+etadot[j]))/2
        rate['microstress_power']=self.edge_sum(micro)
        rate.update(c=cdot,eta=etadot,mu=mu,phase_force=g,diffusion_dissipation=Dc,phase_dissipation=De)
        d=rate['diagnostics']
        for key in ('total_energy','internal_energy','free_energy'):d[key]+=float(extra.sum())
        d['entropy_production']+=float(np.sum((Dc+De)/state.T))
        d.update(mass=float(V*state.c.sum()),min_c=float(state.c.min()),max_c=float(state.c.max()),
            min_eta=float(state.eta.min()),max_eta=float(state.eta.max()),
            gradient_energy=float((Gc+Ge).sum()),partition_energy=float(partition.sum()),
            chemical_energy=float(V*np.sum((state.c-p.c0)**2/2+state.eta**2*(1-state.eta)**2)+partition.sum()),
            diffusion_power=float(Dc.sum()),phase_power=float(De.sum()),
            chemical_flux_sum=float(rate['chemical_power'].sum()),microstress_power_sum=float(rate['microstress_power'].sum()))
        return rate

    def save(self,path,state,time=0.,step=0):
        self.rate(state);m=self.mechanics.mesh
        metadata=dict(n=m.n,length=m.length,degree=self.mechanics.degree,mechanical=asdict(self.mechanics.p),transport=asdict(self.p),
            chemistry=asdict(self.chemistry),time=time,step=step)
        np.savez_compressed(path,**vars(state),fixed=self.fixed,values=self.values,metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']));clock={key:meta.pop(key) for key in ('time','step')}
            model=cls(mechanical=FiniteParameters(**meta.pop('mechanical')),transport=ThermalPlasticParameters(**meta.pop('transport')),
                chemistry=ChemicalParameters(**meta.pop('chemistry')),fixed=data['fixed'],values=data['values'],**meta)
            state=ThermalChemicalState(**{key:data[key].copy() for key in ('u','v','T','Fp','gamma','c','eta')})
            model.rate(state)
            return model,state,clock
