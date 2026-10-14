"""Continuous Q1 damage with common-quadrature thermochemical assembly power."""
import numpy as np
from .thermochemical import ThermalChemical2D
from .thermofracture import ThermalFracture2D,crack_density_terms


class NodalThermalFracture2D(ThermalFracture2D):
    damage_basis='Q1-nodal'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);m=self.mechanics.mesh;n=m.n
        axis=np.linspace(0,m.length,n+1)
        self.damage_coordinates=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
        nodes=np.arange((n+1)**2).reshape(n+1,n+1)
        self.damage_cells=np.stack((nodes[:-1,:-1],nodes[1:,:-1],nodes[1:,1:],nodes[:-1,1:]),axis=-1).reshape(-1,4)
        xi,eta=(2*m.quadrature_points[0]/m.h-1).T
        sx=np.array([-1,1,1,-1]);sy=np.array([-1,-1,1,1])
        self.damage_N=(1+xi[:,None]*sx)*(1+eta[:,None]*sy)/4
        self.damage_gradient=np.stack((sx*(1+eta[:,None]*sy),sy*(1+xi[:,None]*sx)),axis=-1)/(2*m.h)
        lump=np.einsum('g,ga->a',m.weights,self.damage_N)
        self.damage_element_mass=np.broadcast_to(lump,(n*n,4)).copy()
        self.damage_mass=self.assemble_damage(self.damage_element_mass)

    def assemble_damage(self,element):
        return np.bincount(self.damage_cells.ravel(),weights=np.asarray(element).ravel(),minlength=len(self.damage_coordinates))

    def sample_damage(self,value):
        if callable(value):value=value(*self.damage_coordinates.T)
        return np.broadcast_to(value,(len(self.damage_coordinates),)).astype(float,copy=True)

    def damage_quadrature(self,damage):
        return np.einsum('ga,ea->eg',self.damage_N,damage[self.damage_cells])

    def damage_heat_and_power(self,element_force,force,ddot):
        nodal=-force*ddot
        allocation=self.damage_element_mass/self.damage_mass[self.damage_cells]
        heat=np.sum(allocation*nodal[self.damage_cells],axis=1)
        power=np.sum((element_force-allocation*force[self.damage_cells])*ddot[self.damage_cells],axis=1)
        return nodal,heat,power

    def install(self,state):
        d=state.damage;p=self.fracture
        if d.shape!=(len(self.damage_coordinates),) or not np.isfinite(d).all() or np.any((d<0)|(d>1)):
            raise ValueError('Invalid bounded nodal damage state.')
        dq=self.damage_quadrature(d)
        self.mechanics.elastic_scale=p.residual+(1-p.residual)*(1-dq)**2
        self.mechanics.elastic_split=p.elastic_split
        ThermalChemical2D.install(self,state)

    def additional_energy(self,state,local):
        m=self.mechanics.mesh;p=self.fracture;V=m.h**2
        d=self.damage_quadrature(state.damage)
        grad=np.einsum('gaj,ea->egj',self.damage_gradient,state.damage[self.damage_cells])
        w,dw,cw=crack_density_terms(d,p.crack_density);a=p.toughness/cw;coefficient=2*a*p.ell
        h=state.eta**2*(3-2*state.eta);hp=6*state.eta*(1-state.eta)
        surface=np.sum(m.weights*(a*w/p.ell+coefficient*np.sum(grad*grad,axis=-1)/2),axis=1)
        binding=np.sum(m.weights*p.phase_resistance*d*d*h[:,None],axis=1)
        gp=-2*(1-p.residual)*(1-d)
        density=gp*local['damage_elastic']+a*dw/p.ell+2*p.phase_resistance*d*h[:,None]
        element=(np.einsum('g,ga,eg->ea',m.weights,self.damage_N,density)
            +coefficient*np.einsum('g,gaj,egj->ea',m.weights,self.damage_gradient,grad))
        force=self.assemble_damage(element)
        return dict(cell_energy=surface+binding,mu=0.,
            phase_force=p.phase_resistance*hp*np.sum(m.weights*d*d,axis=1)/V,
            surface=surface,binding=binding,damage_force=force,damage_element_force=element,
            degradation_derivative=gp,damage_quadrature=d)

    def rate(self,state):
        rate=ThermalChemical2D.rate(self,state);extra=rate['additional_energy'];m=self.mechanics.mesh
        force=extra['damage_force'];ddot=self.fracture.mobility*np.maximum(-force/self.damage_mass,0)
        ddot=np.where(state.damage==1,0.,ddot)
        nodal_dissipation,dissipation,power=self.damage_heat_and_power(extra['damage_element_force'],force,ddot)
        dqdot=self.damage_quadrature(ddot)
        mixed=-self.mechanics.p.alpha_T*np.sum(m.weights*extra['degradation_derivative']*rate['local']['damage_M_trace']*dqdot,axis=1)
        rate['T']+=(dissipation+state.T*mixed)/rate['capacity']
        rate['microstress_power']+=power
        rate.update(damage=ddot,damage_force=force,damage_nodal_dissipation=nodal_dissipation,
            damage_dissipation=dissipation,damage_assembly_power=power,damage_thermal_mixed=mixed)
        rate['diagnostics']['entropy_production']+=float(np.sum(dissipation/state.T))
        rate['diagnostics'].update(fracture_energy=float(extra['surface'].sum()),damage_binding_energy=float(extra['binding'].sum()),
            damage_power=float(dissipation.sum()),min_damage=float(state.damage.min()),max_damage=float(state.damage.max()),
            damage_integral=float(self.damage_mass@state.damage),max_damage_rate=float(ddot.max()),
            microstress_power_sum=float(rate['microstress_power'].sum()))
        return rate
