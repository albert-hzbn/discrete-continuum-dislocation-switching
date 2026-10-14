"""Section 39 chemical/phase evolution with local thermomechanical ledgers."""
from dataclasses import asdict, dataclass
import json
import numpy as np
from scipy.sparse import coo_matrix,identity,diags
from scipy.sparse.linalg import splu
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
    def __init__(self,n,length=1.,mechanical=None,transport=None,chemistry=None,fixed=None,values=None,degree=1,edges=None,preconditioned=False,quadrature='gauss'):
        super().__init__(n,length,mechanical,transport,fixed=fixed,values=values,degree=degree,edges=edges,quadrature=quadrature)
        self.chemistry=chemistry or ChemicalParameters()
        if preconditioned not in (False,True,'full','adaptive'):raise ValueError("preconditioned must be False, True, 'full' or 'adaptive'.")
        self.preconditioned=preconditioned
        nodes=np.arange(n*n).reshape(n,n);m=self.mechanics.mesh
        self.edge_i=np.r_[nodes[:-1,:].ravel(),nodes[:,:-1].ravel()]
        self.edge_j=np.r_[nodes[1:,:].ravel(),nodes[:,1:].ravel()]
        # Uniform meshes keep scalar h**2 and unit conductances so earlier products are reproduced exactly.
        ax,ay=m.edge_conductances();self.edge_a=np.r_[ax.ravel(),ay.ravel()]
        self.V=m.h**2 if m.uniform else m.cell_volume.copy()

    def integrate(self,values):
        """Cell-volume integral of a cell field."""
        return float(self.V*np.sum(values)) if np.isscalar(self.V) else float(np.sum(self.V*values))

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
        cell_energy=self.edge_sum(coefficient*self.edge_a*delta**2/4,False)
        force=self.edge_sum(-coefficient*self.edge_a*delta)/self.V
        return cell_energy,force

    def edge_laplacian(self,weights):
        n2=self.mechanics.mesh.n**2;i,j=self.edge_i,self.edge_j;w=np.broadcast_to(weights,i.shape)
        return coo_matrix((np.r_[w,w,-w,-w],(np.r_[i,j,i,j],np.r_[i,j,j,i])),shape=(n2,n2)).tocsc()

    def _preconditioner(self,state,dt,start):
        """Section 46.3 frozen stiff blocks; the fixed point and its tolerance are unchanged."""
        if not self.preconditioned:return None
        ch=self.chemistry;n2=self.mechanics.mesh.n**2;i,j=self.edge_i,self.edge_j
        delta=start['mu'][i]-start['mu'][j]
        availability=1. if ch.mobility_kind=='constant' else np.where(delta>=0,state.c[i]*(1-state.c[j]),state.c[j]*(1-state.c[i]))
        I=identity(n2,format='csc');Vinv=diags(1/np.broadcast_to(self.V,(n2,)))
        Pc=I+(dt/2)*Vinv@self.edge_laplacian(ch.mobility*self.edge_a*availability)@(I+ch.kappa_c*Vinv@self.edge_laplacian(self.edge_a))
        PT=I+(dt/2)*diags(1/start['capacity'])@self.edge_laplacian(self.p.conductivity*self.edge_a)
        context=dict(c=splu(Pc.tocsc()),T=splu(PT.tocsc()))
        if self.preconditioned in ('full','adaptive'):
            # Section 47.5: the inertial block is built at a midpoint iterate, never at the initial state.
            # 'full' builds it at the first iterate; 'adaptive' only when a step is still unconverged after 12 updates.
            free=np.setdiff1d(np.arange(len(state.u)),self.fixed)
            context['mechanics']=dict(free=free,mass=self.mass[free],half=dt/2,calls=0,first=2 if self.preconditioned=='full' else 12)
        return context

    def _mechanical_block(self,middle,block):
        """Factor (M+(dt/2)^2 K_ff) with K the internal-force tangent at the current iterate."""
        self.install(middle);free=block['free']
        K=self.mechanics.evaluate(middle.u,tangent=True)['stiffness'].tocsr()[free][:,free]
        try:block['factor']=splu((diags(block['mass'])+block['half']**2*K).tocsc())
        except RuntimeError as error:raise RuntimeError(f'Mechanical preconditioner factorization failed: {error}')
        block['K']=K

    def _iterate(self,middle,following,context):
        fields=dict(vars(following))
        for key,factor in context.items():
            if key=='mechanics':continue
            previous=getattr(middle,key);fields[key]=previous+factor.solve(getattr(following,key)-previous)
        block=context.get('mechanics')
        if block is not None:
            block['calls']+=1
            # The first update is plain Picard; the block is built at that iterate and refreshed every 8 calls.
            if block['calls']>=block['first'] and (block['calls']-block['first'])%8==0:self._mechanical_block(middle,block)
        if block is not None and 'factor' in block:
            free,half,mass=block['free'],block['half'],block['mass']
            ru=following.u[free]-middle.u[free];rv=following.v[free]-middle.v[free]
            du=block['factor'].solve(mass*(ru+half*rv));dv=rv-half*(block['K']@du)/mass
            u=following.u.copy();v=following.v.copy();u[free]=middle.u[free]+du;v[free]=middle.v[free]+dv
            fields.update(u=u,v=v)
        return type(following)(**fields)

    def additional_energy(self,state,local):
        """Temperature-independent extra cell energy and c/eta force densities."""
        return dict(cell_energy=0.,mu=0.,phase_force=0.)

    def rate(self,state):
        rate=super().rate(state);m=self.mechanics.mesh;p=self.mechanics.p;ch=self.chemistry;V=self.V;a=self.edge_a
        i,j=self.edge_i,self.edge_j;h=state.eta**2*(3-2*state.eta);hp=6*state.eta*(1-state.eta)
        Gc,gradc=self.gradient(state.c,ch.kappa_c);Ge,grade=self.gradient(state.eta,ch.kappa_eta)
        local=rate['local']
        extension=self.additional_energy(state,local)
        mu=np.sum(m.cell_weights*local['gc'],axis=1)/V+ch.partition*h+gradc+extension['mu']
        g=np.sum(m.cell_weights*local['geta'],axis=1)/V+ch.partition*(state.c-p.c0)*hp+grade+extension['phase_force']
        delta=mu[i]-mu[j]
        availability=1. if ch.mobility_kind=='constant' else np.where(delta>=0,state.c[i]*(1-state.c[j]),state.c[j]*(1-state.c[i]))
        flow=ch.mobility*a*availability*delta
        cdot=self.edge_sum(-flow)/V
        etadot=-ch.phase_mobility*g
        etadot=np.where(state.eta==0,np.maximum(etadot,0),etadot)
        etadot=np.where(state.eta==1,np.minimum(etadot,0),etadot)
        Dc=self.edge_sum(flow*delta/2,False);De=-V*g*etadot
        factor=local['thermal_factor']
        mixed=p.alpha_T*factor*(p.alpha_c*cdot+p.alpha_eta*hp*etadot)[:,None]
        rate['T']+=(Dc+De+state.T*np.sum(m.cell_weights*mixed,axis=1))/rate['capacity']
        partition=V*ch.partition*(state.c-p.c0)*h
        extra=Gc+Ge+partition+extension['cell_energy']
        rate['cell_internal_energy']+=extra
        rate['chemical_power']=self.edge_sum(-(mu[i]+mu[j])*flow/2)
        micro=(ch.kappa_c*a*(state.c[j]-state.c[i])*(cdot[i]+cdot[j])
               +ch.kappa_eta*a*(state.eta[j]-state.eta[i])*(etadot[i]+etadot[j]))/2
        rate['microstress_power']=self.edge_sum(micro)
        rate.update(c=cdot,eta=etadot,mu=mu,phase_force=g,diffusion_dissipation=Dc,phase_dissipation=De,additional_energy=extension)
        d=rate['diagnostics']
        for key in ('total_energy','internal_energy','free_energy'):d[key]+=float(extra.sum())
        d['entropy_production']+=float(np.sum((Dc+De)/state.T))
        d.update(mass=self.integrate(state.c),min_c=float(state.c.min()),max_c=float(state.c.max()),
            min_eta=float(state.eta.min()),max_eta=float(state.eta.max()),
            gradient_energy=float((Gc+Ge).sum()),partition_energy=float(partition.sum()),
            chemical_energy=self.integrate((state.c-p.c0)**2/2+state.eta**2*(1-state.eta)**2)+float(partition.sum()),
            diffusion_power=float(Dc.sum()),phase_power=float(De.sum()),
            chemical_flux_sum=float(rate['chemical_power'].sum()),microstress_power_sum=float(rate['microstress_power'].sum()))
        return rate

    def save(self,path,state,time=0.,step=0):
        self.rate(state);m=self.mechanics.mesh
        metadata=dict(n=m.n,length=m.length,degree=self.mechanics.degree,mechanical=asdict(self.mechanics.p),transport=asdict(self.p),
            chemistry=asdict(self.chemistry),time=time,step=step,preconditioned=self.preconditioned)
        if not m.uniform:metadata['edges']=[e.tolist() for e in m.edges]
        if getattr(m,'quadrature','gauss')!='gauss':metadata['quadrature']=m.quadrature
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
