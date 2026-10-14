"""Common-energy finite thermochemical damage, Sections 42–44."""
from dataclasses import asdict,dataclass
import json
import numpy as np
from .finite_elasticity import FiniteParameters
from .thermoplastic import ThermalPlasticParameters
from .thermochemical import ChemicalParameters,ThermalChemical2D,ThermalChemicalState


def crack_density_terms(damage,kind):
    """Surface density w, derivative w', and isolated-profile normalization."""
    d=np.asarray(damage)
    if kind=='AT2':return d*d,2*d,2.
    if kind=='AT1':return d,np.ones_like(d),8/3
    raise ValueError('Unknown crack density.')


@dataclass(frozen=True)
class ThermalFractureParameters:
    toughness: float = .002
    ell: float = .08
    residual: float = 1e-4
    mobility: float = 5.
    phase_resistance: float = .004
    elastic_split: str = 'isotropic'
    crack_density: str = 'AT2'

    def __post_init__(self):
        if (not all(np.isfinite(v) for key,v in asdict(self).items() if key not in ('elastic_split','crack_density'))
            or min(self.toughness,self.ell,self.residual)<=0 or self.residual>1
            or min(self.mobility,self.phase_resistance)<0):
            raise ValueError('Invalid thermofracture coefficients.')
        if self.elastic_split not in ('isotropic','volumetric-deviatoric'):raise ValueError('Unknown elastic split.')
        if self.crack_density not in ('AT2','AT1'):raise ValueError('Unknown crack density.')


@dataclass
class ThermalFractureState(ThermalChemicalState):
    damage: np.ndarray


class ThermalFracture2D(ThermalChemical2D):
    damage_basis='cell'

    def __init__(self,n,length=1.,mechanical=None,transport=None,chemistry=None,fracture=None,fixed=None,values=None,degree=1,edges=None,preconditioned=False,quadrature='gauss'):
        self.fracture=fracture or ThermalFractureParameters()
        super().__init__(n,length,mechanical,transport,chemistry,fixed,values,degree,edges,preconditioned,quadrature)

    def sample_damage(self,value):return self.sample_cell(value)

    def initial(self,T=1.,u=None,equilibrate=False,c=.3,eta=.3,damage=0.):
        m=self.mechanics.mesh
        state=ThermalFractureState(
            np.zeros(2*m.node_count) if u is None else np.array(u,dtype=float,copy=True),
            np.zeros(2*m.node_count),self.sample_cell(T),
            np.broadcast_to(np.eye(3),(m.n*m.n,m.quadrature_count,3,3)).copy(),
            np.zeros((m.n*m.n,m.quadrature_count,2)),self.sample_cell(c),self.sample_cell(eta),self.sample_damage(damage))
        state.u[self.fixed]=self.values;self.install(state)
        if equilibrate:state.u,_,_=self.mechanics.equilibrate(state.u,self.fixed,self.values)
        self.rate(state)
        return state

    def install(self,state):
        d=state.damage;m=self.mechanics.mesh;p=self.fracture
        if d.shape!=(m.n*m.n,) or not np.isfinite(d).all() or np.any((d<0)|(d>1)):
            raise ValueError('Invalid bounded damage state.')
        self.mechanics.elastic_scale=np.broadcast_to((p.residual+(1-p.residual)*(1-d)**2)[:,None],(m.n*m.n,m.quadrature_count))
        self.mechanics.elastic_split=p.elastic_split
        super().install(state)

    def additional_energy(self,state,local):
        m=self.mechanics.mesh;p=self.fracture;V=self.V;d=state.damage
        h=state.eta**2*(3-2*state.eta);hp=6*state.eta*(1-state.eta)
        w,dw,cw=crack_density_terms(d,p.crack_density);coefficient=2*p.toughness*p.ell/cw
        gradient,force=self.gradient(d,coefficient)
        surface=V*p.toughness*w/(cw*p.ell)+gradient
        binding=V*p.phase_resistance*d**2*h
        gp=-2*(1-p.residual)*(1-d)
        force+=gp*np.sum(m.cell_weights*local['damage_elastic'],axis=1)/V+p.toughness*dw/(cw*p.ell)+2*p.phase_resistance*d*h
        return dict(cell_energy=surface+binding,mu=0.,phase_force=p.phase_resistance*d**2*hp,
                    surface=surface,binding=binding,damage_force=force,degradation_derivative=gp,damage_gradient_coefficient=coefficient)

    def rate(self,state):
        rate=super().rate(state);extra=rate['additional_energy'];m=self.mechanics.mesh;p=self.fracture;V=self.V
        force=extra['damage_force'];ddot=p.mobility*np.maximum(-force,0)
        ddot=np.where(state.damage==1,0.,ddot)
        dissipation=-V*force*ddot
        mixed=-extra['degradation_derivative']*self.mechanics.p.alpha_T*np.sum(m.cell_weights*rate['local']['damage_M_trace'],axis=1)*ddot
        rate['T']+=(dissipation+state.T*mixed)/rate['capacity']
        i,j=self.edge_i,self.edge_j
        power=self.edge_sum(extra['damage_gradient_coefficient']*self.edge_a*(state.damage[j]-state.damage[i])*(ddot[i]+ddot[j])/2)
        rate['microstress_power']+=power
        rate.update(damage=ddot,damage_force=force,damage_dissipation=dissipation,damage_microstress_power=power,damage_thermal_mixed=mixed)
        rate['diagnostics']['entropy_production']+=float(np.sum(dissipation/state.T))
        rate['diagnostics'].update(fracture_energy=float(extra['surface'].sum()),damage_binding_energy=float(extra['binding'].sum()),
            damage_power=float(dissipation.sum()),min_damage=float(state.damage.min()),max_damage=float(state.damage.max()),
            damage_integral=self.integrate(state.damage),max_damage_rate=float(ddot.max()),
            microstress_power_sum=float(rate['microstress_power'].sum()))
        return rate

    def save(self,path,state,time=0.,step=0):
        self.rate(state);m=self.mechanics.mesh
        metadata=dict(n=m.n,length=m.length,degree=self.mechanics.degree,mechanical=asdict(self.mechanics.p),
            transport=asdict(self.p),chemistry=asdict(self.chemistry),fracture=asdict(self.fracture),damage_basis=self.damage_basis,time=time,step=step,
            preconditioned=self.preconditioned)
        if not m.uniform:metadata['edges']=[e.tolist() for e in m.edges]
        if getattr(m,'quadrature','gauss')!='gauss':metadata['quadrature']=m.quadrature
        active=getattr(self.mechanics,'active',None)  # Section 64 deleted cells, persisted only when present
        extra={} if active is None else dict(active=active)
        if getattr(self.mechanics,'subdivide',0):metadata['subdivide']=int(self.mechanics.subdivide)
        if getattr(self,'split',None):extra.update(split=np.array(list(self.split),dtype=int),split_row=np.array(self.split_row))
        if getattr(self,'discrete',None) is not None and self.discrete.any():  # Section 71 discrete-crack cells
            extra.update(discrete=self.discrete,reset_cells=self.mechanics.reset_cells,reset=self.mechanics.reset)
        if getattr(self,'elastic_discrete',False):extra.update(elastic_discrete=np.array(True))
        np.savez_compressed(path,**vars(state),**extra,fixed=self.fixed,values=self.values,metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']));clock={key:meta.pop(key) for key in ('time','step')}
            if meta.pop('damage_basis','cell')!=cls.damage_basis:raise ValueError('Checkpoint damage basis does not match model class.')
            active=data['active'].copy() if 'active' in data.files else None;subdivide=meta.pop('subdivide',0)
            model=cls(mechanical=FiniteParameters(**meta.pop('mechanical')),transport=ThermalPlasticParameters(**meta.pop('transport')),
                chemistry=ChemicalParameters(**meta.pop('chemistry')),fracture=ThermalFractureParameters(**meta.pop('fracture')),
                fixed=data['fixed'],values=data['values'],**meta)
            state=ThermalFractureState(**{key:data[key].copy() for key in ('u','v','T','Fp','gamma','c','eta','damage')})
            if active is not None:model.mechanics.active=active
            if subdivide:model.mechanics.subdivide=subdivide
            if 'split' in data.files:model._apply_split([int(a) for a in data['split']],int(data['split_row']))
            if 'discrete' in data.files:
                model.discrete=data['discrete'].copy();model.mechanics.reset_cells=data['reset_cells'].copy();model.mechanics.reset=data['reset'].copy()
            if 'elastic_discrete' in data.files:model.elastic_discrete=bool(data['elastic_discrete'])
            model.rate(state)
            return model,state,clock
