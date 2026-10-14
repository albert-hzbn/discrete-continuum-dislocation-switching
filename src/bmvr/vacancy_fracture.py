"""Reactive vacancies in finite thermochemical plasticity and nodal-damage fracture (Section 50)."""
from dataclasses import asdict,dataclass
import json
import numpy as np
from .finite_elasticity import FiniteParameters
from .thermoplastic import ThermalPlasticParameters
from .thermochemical import ChemicalParameters
from .thermofracture import ThermalFractureParameters,ThermalFractureState
from .nodal_damage import NodalThermalFracture2D


@dataclass(frozen=True)
class VacancyParameters:
    theta: float = .02
    equilibrium: float = .05
    binding: float = 0.
    sink: float = 0.
    damage_sink: float = 0.
    mobility: float = 0.
    generation: float = 0.

    def __post_init__(self):
        values=asdict(self)
        if not all(np.isfinite(v) for v in values.values()) or self.theta<=0 or not 0<self.equilibrium<1:
            raise ValueError('Invalid vacancy mixing scale or equilibrium fraction.')
        if min(self.binding,self.sink,self.damage_sink,self.mobility,self.generation)<0:
            raise ValueError('Vacancy coefficients must be nonnegative.')

    @property
    def formation(self):
        """e_f chosen so that the damage-free sink equilibrium is x_eq."""
        return self.theta*np.log((1-self.equilibrium)/self.equilibrium)


@dataclass
class VacancyFractureState(ThermalFractureState):
    vac: np.ndarray


def mixing(x):return x*np.log(x)+(1-x)*np.log(1-x)


class NodalVacancyFracture2D(NodalThermalFracture2D):
    damage_basis='Q1-nodal-vacancy'

    def __init__(self,*args,vacancy=None,**kwargs):
        self.vacancy=vacancy or VacancyParameters()
        super().__init__(*args,**kwargs)

    def initial(self,T=1.,u=None,equilibrate=False,c=.3,eta=.3,damage=0.,vac=None):
        m=self.mechanics.mesh
        state=VacancyFractureState(
            np.zeros(2*m.node_count) if u is None else np.array(u,dtype=float,copy=True),
            np.zeros(2*m.node_count),self.sample_cell(T),
            np.broadcast_to(np.eye(3),(m.n*m.n,m.quadrature_count,3,3)).copy(),
            np.zeros((m.n*m.n,m.quadrature_count,2)),self.sample_cell(c),self.sample_cell(eta),self.sample_damage(damage),
            self.sample_cell(self.vacancy.equilibrium if vac is None else vac))
        state.u[self.fixed]=self.values;self.install(state)
        if equilibrate:state.u,_,_=self.mechanics.equilibrate(state.u,self.fixed,self.values)
        self.rate(state)
        return state

    def damage_squared(self,state):
        """Cell integral of d^2 with the element quadrature of the nodal damage."""
        return np.sum(self.mechanics.mesh.cell_weights*self.damage_quadrature(state.damage)**2,axis=1)

    def vacancy_potential(self,state):
        x=state.vac;p=self.vacancy;V=np.broadcast_to(self.V,x.shape)
        if x.shape!=(self.mechanics.mesh.n**2,) or not np.isfinite(x).all() or np.any((x<=0)|(x>=1)):
            raise ValueError('Vacancy fraction must lie strictly inside (0,1).')
        return p.formation+p.theta*np.log(x/(1-x))-p.binding*self.damage_squared(state)/V

    def slip_shift(self,state):
        return self.vacancy.generation*self.vacancy_potential(state) if self.vacancy.generation>0 else None

    def additional_energy(self,state,local):
        extra=super().additional_energy(state,local);p=self.vacancy;m=self.mechanics.mesh
        x=state.vac;V=np.broadcast_to(self.V,x.shape);D=self.damage_squared(state)
        reference=p.formation*p.equilibrium+p.theta*mixing(p.equilibrium)
        energy=V*(p.formation*x+p.theta*mixing(x)-reference)-p.binding*x*D
        # d(-b x integral d^2)/dd_a = -2 b x sum_g w_g d_g N_a
        coupling=np.einsum('eg,ga->ea',-2*p.binding*x[:,None]*m.cell_weights*extra['damage_quadrature'],self.damage_N)
        extra['cell_energy']=extra['cell_energy']+energy
        extra['damage_element_force']=extra['damage_element_force']+coupling
        extra['damage_force']=extra['damage_force']+self.assemble_damage(coupling)
        extra.update(vacancy_energy=energy,vacancy_mu=self.vacancy_potential(state),damage_squared=D)
        return extra

    def rate(self,state,damage_rate_limit=None):
        rate=super().rate(state,damage_rate_limit);extra=rate['additional_energy'];p=self.vacancy
        x=state.vac;V=np.broadcast_to(self.V,x.shape);mu=extra['vacancy_mu'];i,j=self.edge_i,self.edge_j
        delta=mu[i]-mu[j];availability=np.where(delta>=0,x[i]*(1-x[j]),x[j]*(1-x[i]))
        flow=p.mobility*self.edge_a*availability*delta
        transport=self.edge_sum(-flow)/V;transport_dissipation=self.edge_sum(flow*delta/2,False)
        k=p.sink+p.damage_sink*extra['damage_squared']/V
        sink=-k*mu;sink_dissipation=V*k*mu*mu
        generation=p.generation*rate['cell_slip_magnitude']/V
        rate['vac']=transport+sink+generation
        rate['T']+=(transport_dissipation+sink_dissipation)/rate['capacity']
        rate['chemical_power']=rate['chemical_power']+self.edge_sum(-(mu[i]+mu[j])*flow/2)
        stored=rate['slip_stored_power']
        rate.update(vacancy_transport_dissipation=transport_dissipation,vacancy_sink_dissipation=sink_dissipation,
            vacancy_generation=generation,vacancy_sink=sink,vacancy_transport=transport)
        d=rate['diagnostics']
        d['entropy_production']+=float(np.sum((transport_dissipation+sink_dissipation)/state.T))
        d.update(vacancy_inventory=float(np.sum(V*x)),vacancy_generation_rate=float(np.sum(V*generation)),
            vacancy_sink_rate=float(np.sum(V*sink)),vacancy_energy=float(np.sum(extra['vacancy_energy'])),
            vacancy_dissipation=float(np.sum(transport_dissipation+sink_dissipation)),
            slip_stored_power=0. if stored is None else float(np.sum(stored)),
            min_vac=float(x.min()),max_vac=float(x.max()),min_vacancy_mu=float(mu.min()),max_vacancy_mu=float(mu.max()),
            vacancy_flux_sum=float(np.sum(transport*V)))
        return rate

    def save(self,path,state,time=0.,step=0):
        self.rate(state);m=self.mechanics.mesh
        metadata=dict(n=m.n,length=m.length,degree=self.mechanics.degree,mechanical=asdict(self.mechanics.p),
            transport=asdict(self.p),chemistry=asdict(self.chemistry),fracture=asdict(self.fracture),vacancy=asdict(self.vacancy),
            damage_basis=self.damage_basis,time=time,step=step,preconditioned=self.preconditioned)
        if not m.uniform:metadata['edges']=[e.tolist() for e in m.edges]
        if m.quadrature!='gauss':metadata['quadrature']=m.quadrature
        np.savez_compressed(path,**vars(state),fixed=self.fixed,values=self.values,metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']));clock={key:meta.pop(key) for key in ('time','step')}
            if meta.pop('damage_basis')!=cls.damage_basis:raise ValueError('Checkpoint damage basis does not match model class.')
            model=cls(mechanical=FiniteParameters(**meta.pop('mechanical')),transport=ThermalPlasticParameters(**meta.pop('transport')),
                chemistry=ChemicalParameters(**meta.pop('chemistry')),fracture=ThermalFractureParameters(**meta.pop('fracture')),
                vacancy=VacancyParameters(**meta.pop('vacancy')),fixed=data['fixed'],values=data['values'],**meta)
            state=VacancyFractureState(**{key:data[key].copy() for key in ('u','v','T','Fp','gamma','c','eta','damage','vac')})
            model.rate(state)
            return model,state,clock
