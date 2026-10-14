"""Common-energy finite thermochemical damage, Sections 42–43."""
from dataclasses import asdict,dataclass
import json
import numpy as np
from .finite_elasticity import FiniteParameters
from .thermoplastic import ThermalPlasticParameters
from .thermochemical import ChemicalParameters,ThermalChemical2D,ThermalChemicalState


@dataclass(frozen=True)
class ThermalFractureParameters:
    toughness: float = .002
    ell: float = .08
    residual: float = 1e-4
    mobility: float = 5.
    phase_resistance: float = .004
    elastic_split: str = 'isotropic'

    def __post_init__(self):
        if (not all(np.isfinite(v) for key,v in asdict(self).items() if key!='elastic_split')
            or min(self.toughness,self.ell,self.residual)<=0 or self.residual>1
            or min(self.mobility,self.phase_resistance)<0):
            raise ValueError('Invalid thermofracture coefficients.')
        if self.elastic_split not in ('isotropic','volumetric-deviatoric'):raise ValueError('Unknown elastic split.')


@dataclass
class ThermalFractureState(ThermalChemicalState):
    damage: np.ndarray


class ThermalFracture2D(ThermalChemical2D):
    def __init__(self,n,length=1.,mechanical=None,transport=None,chemistry=None,fracture=None,fixed=None,values=None,degree=1):
        self.fracture=fracture or ThermalFractureParameters()
        super().__init__(n,length,mechanical,transport,chemistry,fixed,values,degree)

    def initial(self,T=1.,u=None,equilibrate=False,c=.3,eta=.3,damage=0.):
        m=self.mechanics.mesh
        state=ThermalFractureState(
            np.zeros(2*m.node_count) if u is None else np.array(u,dtype=float,copy=True),
            np.zeros(2*m.node_count),self.sample_cell(T),
            np.broadcast_to(np.eye(3),(m.n*m.n,m.quadrature_count,3,3)).copy(),
            np.zeros((m.n*m.n,m.quadrature_count,2)),self.sample_cell(c),self.sample_cell(eta),self.sample_cell(damage))
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
        m=self.mechanics.mesh;p=self.fracture;V=m.h**2;d=state.damage
        h=state.eta**2*(3-2*state.eta);hp=6*state.eta*(1-state.eta)
        gradient,force=self.gradient(d,p.toughness*p.ell)
        surface=V*p.toughness*d**2/(2*p.ell)+gradient
        binding=V*p.phase_resistance*d**2*h
        gp=-2*(1-p.residual)*(1-d)
        force+=gp*np.sum(m.weights*local['damage_elastic'],axis=1)/V+p.toughness*d/p.ell+2*p.phase_resistance*d*h
        return dict(cell_energy=surface+binding,mu=0.,phase_force=p.phase_resistance*d**2*hp,
                    surface=surface,binding=binding,damage_force=force,degradation_derivative=gp)

    def rate(self,state):
        rate=super().rate(state);extra=rate['additional_energy'];m=self.mechanics.mesh;p=self.fracture;V=m.h**2
        force=extra['damage_force'];ddot=p.mobility*np.maximum(-force,0)
        ddot=np.where(state.damage==1,0.,ddot)
        dissipation=-V*force*ddot
        mixed=-extra['degradation_derivative']*self.mechanics.p.alpha_T*np.sum(m.weights*rate['local']['damage_M_trace'],axis=1)*ddot
        rate['T']+=(dissipation+state.T*mixed)/rate['capacity']
        i,j=self.edge_i,self.edge_j
        power=self.edge_sum(p.toughness*p.ell*(state.damage[j]-state.damage[i])*(ddot[i]+ddot[j])/2)
        rate['microstress_power']+=power
        rate.update(damage=ddot,damage_force=force,damage_dissipation=dissipation,damage_microstress_power=power,damage_thermal_mixed=mixed)
        rate['diagnostics']['entropy_production']+=float(np.sum(dissipation/state.T))
        rate['diagnostics'].update(fracture_energy=float(extra['surface'].sum()),damage_binding_energy=float(extra['binding'].sum()),
            damage_power=float(dissipation.sum()),min_damage=float(state.damage.min()),max_damage=float(state.damage.max()),
            damage_integral=float(V*state.damage.sum()),max_damage_rate=float(ddot.max()),
            microstress_power_sum=float(rate['microstress_power'].sum()))
        return rate

    def save(self,path,state,time=0.,step=0):
        self.rate(state);m=self.mechanics.mesh
        metadata=dict(n=m.n,length=m.length,degree=self.mechanics.degree,mechanical=asdict(self.mechanics.p),
            transport=asdict(self.p),chemistry=asdict(self.chemistry),fracture=asdict(self.fracture),time=time,step=step)
        np.savez_compressed(path,**vars(state),fixed=self.fixed,values=self.values,metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']));clock={key:meta.pop(key) for key in ('time','step')}
            model=cls(mechanical=FiniteParameters(**meta.pop('mechanical')),transport=ThermalPlasticParameters(**meta.pop('transport')),
                chemistry=ChemicalParameters(**meta.pop('chemistry')),fracture=ThermalFractureParameters(**meta.pop('fracture')),
                fixed=data['fixed'],values=data['values'],**meta)
            state=ThermalFractureState(**{key:data[key].copy() for key in ('u','v','T','Fp','gamma','c','eta','damage')})
            model.rate(state)
            return model,state,clock
