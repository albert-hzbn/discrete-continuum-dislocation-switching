"""Continuous Q1 damage with common-quadrature thermochemical assembly power."""
import numpy as np
from dataclasses import replace
from .thermochemical import ThermalChemical2D
from .thermofracture import ThermalFracture2D,crack_density_terms


class NodalThermalFracture2D(ThermalFracture2D):
    damage_basis='Q1-nodal'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);m=self.mechanics.mesh;n=m.n
        axes=(np.linspace(0,m.length,n+1),)*2 if m.uniform else m.edges
        self.damage_coordinates=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,2)
        nodes=np.arange((n+1)**2).reshape(n+1,n+1)
        self.damage_cells=np.stack((nodes[:-1,:-1],nodes[1:,:-1],nodes[1:,1:],nodes[:-1,1:]),axis=-1).reshape(-1,4)
        xi,eta=(2*m.quadrature_points[0]/m.h-1).T if m.uniform else m.reference_points.T
        sx=np.array([-1,1,1,-1]);sy=np.array([-1,-1,1,1])
        self.damage_N=(1+xi[:,None]*sx)*(1+eta[:,None]*sy)/4
        self.damage_gradient=np.stack((sx*(1+eta[:,None]*sy),sy*(1+xi[:,None]*sx)),axis=-1)/(2*m.h)
        if m.uniform:self.damage_element_mass=np.broadcast_to(np.einsum('g,ga->a',m.weights,self.damage_N),(n*n,4)).copy()
        else:self.damage_element_mass=np.einsum('eg,ga->ea',m.cell_weights,self.damage_N)
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
        scale=p.residual+(1-p.residual)*(1-dq)**2
        discrete=getattr(self,'discrete',None)
        if discrete is not None:scale[discrete]=1.  # Section 71: discrete-crack cells are no longer degraded
        self.mechanics.elastic_scale=scale
        self.mechanics.elastic_split=p.elastic_split
        ThermalChemical2D.install(self,state)

    def additional_energy(self,state,local):
        m=self.mechanics.mesh;p=self.fracture;V=self.V
        d=self.damage_quadrature(state.damage)
        grad=np.einsum('gaj,ea->egj',self.damage_gradient,state.damage[self.damage_cells])
        if not m.uniform:grad=grad*m.grad_scale[:,None,:]
        w,dw,cw=crack_density_terms(d,p.crack_density);a=p.toughness/cw;coefficient=2*a*p.ell
        h=state.eta**2*(3-2*state.eta);hp=6*state.eta*(1-state.eta)
        surface=np.sum(m.cell_weights*(a*w/p.ell+coefficient*np.sum(grad*grad,axis=-1)/2),axis=1)
        binding=np.sum(m.cell_weights*p.phase_resistance*d*d*h[:,None],axis=1)
        gp=-2*(1-p.residual)*(1-d)
        discrete=getattr(self,'discrete',None)
        if discrete is not None:gp=np.where(discrete[:,None],0.,gp)  # no damage-elastic coupling in discrete-crack cells
        density=gp*local['damage_elastic']+a*dw/p.ell+2*p.phase_resistance*d*h[:,None]
        weighted=grad if m.uniform else grad*m.grad_scale[:,None,:]*m.weight_scale[:,None,None]
        if not m.uniform:density=density*m.weight_scale[:,None]
        element=(np.einsum('g,ga,eg->ea',m.weights,self.damage_N,density)
            +coefficient*np.einsum('g,gaj,egj->ea',m.weights,self.damage_gradient,weighted))
        force=self.assemble_damage(element)
        return dict(cell_energy=surface+binding,mu=0.,
            phase_force=p.phase_resistance*hp*np.sum(m.cell_weights*d*d,axis=1)/V,
            surface=surface,binding=binding,damage_force=force,damage_element_force=element,
            degradation_derivative=gp,damage_quadrature=d)

    def rate(self,state,damage_rate_limit=None):
        rate=ThermalChemical2D.rate(self,state);extra=rate['additional_energy'];m=self.mechanics.mesh
        force=extra['damage_force'];ddot=self.fracture.mobility*np.maximum(-force/self.damage_mass,0)
        ddot=np.where(state.damage==1,0.,ddot)
        upper=np.zeros_like(ddot,dtype=bool)
        if damage_rate_limit is not None:
            limit=np.asarray(damage_rate_limit)
            if limit.shape!=ddot.shape or not np.isfinite(limit).all() or np.any(limit<0):
                raise ValueError('Invalid discrete damage-rate bound.')
            upper=ddot>=limit;ddot=np.minimum(ddot,limit)
        nodal_dissipation,dissipation,power=self.damage_heat_and_power(extra['damage_element_force'],force,ddot)
        dqdot=self.damage_quadrature(ddot)
        mixed=-self.mechanics.p.alpha_T*np.sum(m.cell_weights*extra['degradation_derivative']*rate['local']['damage_M_trace']*dqdot,axis=1)
        rate['T']+=(dissipation+state.T*mixed)/rate['capacity']
        rate['microstress_power']+=power
        rate.update(damage=ddot,damage_force=force,damage_nodal_dissipation=nodal_dissipation,
            damage_dissipation=dissipation,damage_assembly_power=power,damage_thermal_mixed=mixed,damage_upper_active=upper)
        rate['diagnostics']['entropy_production']+=float(np.sum(dissipation/state.T))
        rate['diagnostics'].update(fracture_energy=float(extra['surface'].sum()),damage_binding_energy=float(extra['binding'].sum()),
            damage_power=float(dissipation.sum()),min_damage=float(state.damage.min()),max_damage=float(state.damage.max()),
            damage_integral=float(self.damage_mass@state.damage),max_damage_rate=float(ddot.max()),
            microstress_power_sum=float(rate['microstress_power'].sum()))
        return rate

    def delete_broken(self,state,threshold=1.):
        """Section 64: remove the mechanics of cells with a fully broken edge (both edge damage nodes >= threshold).

        The removed elastic and thermoelastic internal energy of each deleted cell becomes heat in that cell. After
        deletion its internal energy is chemical plus c_v (T - T0), linear in T with slope c_v, so the temperature rise
        restores the cell's internal energy exactly and the total energy is unchanged.
        """
        new=self.broken_edge_cells(state,threshold)
        return self.remove_cells(state,new) if new.any() else (state,None)

    def broken_edge_cells(self,state,threshold=1.):
        """Active cells with a fully broken edge (both edge damage nodes >= threshold); the Section 64 criterion."""
        mech=self.mechanics;m=mech.mesh
        active=np.ones(m.n*m.n,dtype=bool) if getattr(mech,'active',None) is None else mech.active
        broken=state.damage[self.damage_cells]>=threshold
        return active&(broken&np.roll(broken,-1,axis=1)).any(axis=1)  # corners are ordered around the cell

    def erode_refused(self,state,threshold=1.):
        """Section 68: after a step refused by the orientation checks, remove the refusing cells if every one of them has a
        fully broken edge. Returns (state, event); event is None when some refusing cell is intact (a genuine failure)."""
        refused=getattr(self.mechanics,'refused_cells',None)
        if refused is None or not len(refused):return state,None
        candidates=self.broken_edge_cells(state,threshold)
        if not candidates[refused].all():return state,None
        new=np.zeros_like(candidates);new[refused]=True;self.mechanics.refused_cells=None
        return self.remove_cells(state,new)

    def remove_cells(self,state,new):
        """Delete the mechanics of the cells in the mask `new`, moving their mechanical internal energy to heat in place."""
        mech=self.mechanics;m=mech.mesh
        active=np.ones(m.n*m.n,dtype=bool) if getattr(mech,'active',None) is None else mech.active.copy()
        new=new&active
        self.install(state);before=self.rate(state)['diagnostics']['total_energy']
        e_old=np.sum(m.cell_weights*mech.evaluate(state.u)['local']['internal_energy'],axis=1)
        mech.active=active&~new
        e_new=np.sum(m.cell_weights*mech.evaluate(state.u)['local']['internal_energy'],axis=1)
        V=np.sum(m.cell_weights,axis=1);T=state.T.copy();moved=(e_old-e_new)[new]
        T[new]+=moved/(mech.p.cv*V[new])
        result=replace(state,T=T);after=self.rate(result)['diagnostics']['total_energy']
        return result,dict(cells=np.flatnonzero(new).tolist(),energy_moved=float(moved.sum()),total_before=float(before),total_after=float(after),
            ledger_error=float(abs(after-before)/max(abs(before),1e-30)),active_cells=int(mech.active.sum()))

    def slip_frozen(self):
        """Section 71.6 option: discrete-crack cells are elastic, since the regularization layer vanishes as ell -> 0."""
        return self.discrete if getattr(self,'elastic_discrete',False) and getattr(self,'discrete',None) is not None else None

    def released_face_cells(self,state=None,band=None):
        """Cells above and below every released line edge (the edge's midpoint node has been split).

        band (Section 71.4): also every cell in a released column whose four damage nodes all reach `band`, the part of
        the smeared crack that is fully formed. Restricting to released columns keeps the tip process zone smeared.
        """
        m=self.mechanics.mesh;n=m.n;mask=np.zeros(n*n,dtype=bool)
        if not getattr(self,'split',None):return mask
        j0=self.split_row;rows=2*n+1
        columns=[k for k in range(n) if (2*k+1)*rows+2*j0 in self.split]
        for k in columns:mask[k*n+j0]=mask[k*n+j0-1]=True
        if band is not None and columns:
            formed=(state.damage[self.damage_cells]>=band).all(axis=1).reshape(n,n)
            mask.reshape(n,n)[columns]|=formed[columns]
        return mask

    def discretize_released(self,state,band=None):
        """Section 71: smeared-to-discrete transition beside the released crack faces.

        Every active cell adjacent to a released line edge becomes a discrete-crack cell. Its elastic degradation and its
        damage-elastic coupling are switched off, and each point receives the reference reset R = a F^-1 Fp, so the elastic
        stretch F R Fp^-1 / a is the identity: the cell is intact and stress-free in its current shape. The damage field
        and its fracture energy are untouched (no damage decreases). The change of the cell's mechanical internal energy
        becomes heat in the same cell, so the total energy is unchanged.
        """
        mech=self.mechanics;m=mech.mesh;n=m.n
        if getattr(self,'discrete',None) is None:self.discrete=np.zeros(n*n,dtype=bool)
        active=np.ones(n*n,dtype=bool) if getattr(mech,'active',None) is None else mech.active
        new=self.released_face_cells(state,band)&~self.discrete&active
        if not new.any():return state,None
        self.install(state);before=self.rate(state)['diagnostics']['total_energy']
        ev=mech.evaluate(state.u);e_old=np.sum(m.cell_weights*ev['local']['internal_energy'],axis=1)
        idx=np.flatnonzero(new);F=ev['F'][idx];Fp=np.asarray(mech.Fp)[idx]
        p=mech.p;c,eta,T=(np.broadcast_to(np.asarray(v),(n*n,m.quadrature_count))[idx] for v in (mech.c,mech.eta,mech.T))
        a=np.exp(p.alpha_c*(c-p.c0)+p.alpha_eta*eta**2*(3-2*eta)+p.alpha_T*(T-p.T0))
        R=a[...,None,None]*np.linalg.inv(F)@Fp
        old_cells=getattr(mech,'reset_cells',None)
        mech.reset_cells=idx if old_cells is None else np.concatenate([old_cells,idx])
        mech.reset=R if old_cells is None else np.concatenate([mech.reset,R])
        self.discrete=self.discrete|new;self.install(state)
        e_new=np.sum(m.cell_weights*mech.evaluate(state.u)['local']['internal_energy'],axis=1);moved=(e_old-e_new)[new]
        # The cell keeps thermoelastic coupling, so its internal energy is not linear in T: solve e(T) = e_old by Newton.
        result=state
        for _ in range(30):
            self.install(result);local=mech.evaluate(result.u)['local']
            r=(e_old-np.sum(m.cell_weights*local['internal_energy'],axis=1))[new]
            if np.max(abs(r))<=1e-15*max(np.max(abs(e_old)),1e-30):break
            Tn=result.T.copy();Tn[new]+=r/np.sum(m.cell_weights*local['heat_capacity'],axis=1)[new];result=replace(result,T=Tn)
        after=self.rate(result)['diagnostics']['total_energy']
        return result,dict(cells=idx.tolist(),energy_moved=float(moved.sum()),min_moved=float(moved.min()),
            ledger_error=float(abs(after-before)/max(abs(before),1e-30)),discrete_cells=int(self.discrete.sum()))

    def split_broken_line(self,state,line=.5,threshold=1.):
        """Section 65: node release along a known crack line (a displacement discontinuity).

        Q2 nodes on the line y=`line` behind the crack tip are duplicated: a line edge is broken when both of its damage
        nodes reached `threshold`; its midpoint node and every corner whose adjacent line edges are all broken are split,
        and the cells above the line are reconnected to the duplicates. Duplicates start at the same position and
        velocity with the lumped mass of their cells, so the total energy is unchanged by the event.
        """
        mech=self.mechanics;m=mech.mesh;n=m.n
        if not hasattr(self,'split'):self.split={}
        yq=np.unique(np.round(m.coordinates[:,1],12));jq=np.flatnonzero(abs(yq-line)<1e-12)
        if len(jq)!=1 or jq[0]%2:raise ValueError('The crack line must be a row of cell corners.')
        j0=jq[0]//2;rows=2*n+1
        d=state.damage.reshape(n+1,n+1)[:,j0]
        broken=(d[:-1]>=threshold)&(d[1:]>=threshold)
        want=set()
        for k in np.flatnonzero(broken):
            want.add((2*k+1)*rows+2*j0)
            for c in (k,k+1):
                left=broken[c-1] if c>0 else True;right=broken[c] if c<n else False
                if left and right:want.add(2*c*rows+2*j0)
        new=sorted(want-set(self.split))
        if not new:return state,None
        before=self.rate(state)['diagnostics']['total_energy']
        self._apply_split(new,j0)
        index=np.ravel(np.stack([2*np.array(new),2*np.array(new)+1],1))
        result=replace(state,u=np.concatenate([state.u,state.u[index]]),v=np.concatenate([state.v,state.v[index]]))
        after=self.rate(result)['diagnostics']['total_energy']
        return result,dict(nodes=len(new),total_split=len(self.split),total_before=float(before),total_after=float(after),
            ledger_error=float(abs(after-before)/max(abs(before),1e-30)))

    def _apply_split(self,new,j0):
        """Append duplicates of Q2 nodes `new` and reconnect the cells directly above corner row `j0` to them."""
        mech=self.mechanics;m=mech.mesh;n=m.n
        if not hasattr(self,'split'):self.split={}
        if np.any(mech.external[np.ravel([[2*a,2*a+1] for a in new])]!=0):raise ValueError('Node release requires unloaded line nodes.')
        first=m.node_count;dup={a:first+i for i,a in enumerate(new)}
        m.coordinates=np.vstack([m.coordinates,m.coordinates[new]])
        for c in np.arange(n)*n+j0:
            m.cells[c]=[dup.get(a,a) for a in m.cells[c]]
        m.dofs=(2*m.cells[...,None]+np.arange(2)).reshape(-1,m.dofs.shape[1])
        mech.external=np.concatenate([mech.external,np.zeros(2*len(new))])
        mass_element=(np.broadcast_to(self.p.density*np.einsum('g,ga->a',m.weights,m.N),m.cells.shape) if m.uniform
                      else self.p.density*np.einsum('eg,ga->ea',m.cell_weights,m.N))
        self.mass=np.repeat(np.bincount(m.cells.ravel(),weights=mass_element.ravel(),minlength=m.node_count),2)
        self.split.update(dup);self.split_row=int(j0)

    def step(self,state,dt,tolerance=1e-12,max_iterations=40):
        """Coupled midpoint with the final nodal damage obstacle solved in rate space."""
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite time step required.')
        if not np.isfinite(tolerance) or tolerance<=0 or max_iterations<1:raise ValueError('Invalid midpoint solve controls.')
        context=self._preconditioner(state,dt,self.rate(state));limit=(1-state.damage)/dt
        middle=type(state)(**{key:value.copy() for key,value in vars(state).items()})
        for iteration in range(max_iterations):
            mid=self.rate(middle,limit);following=self._advance(state,mid,.5*dt)
            error=max(float(np.max(abs(getattr(following,key)-getattr(middle,key)))) for key in vars(state))
            middle=following if context is None or error<=tolerance else self._iterate(middle,following,context)
            if error<=tolerance:break
        else:raise RuntimeError(f'Constrained nodal midpoint did not converge: {error:.3e}.')
        mid=self.rate(middle,limit);result=self._advance(state,mid,dt)
        # This is the active obstacle equation, not an after-the-fact repair of heat.
        result.damage[mid['damage_upper_active']]=1.
        final=self.rate(result);obstacle=0.
        if self.fracture.mobility>0:
            reaction=np.maximum(-mid['damage_force']-self.damage_mass*mid['damage']/self.fracture.mobility,0)
            obstacle=float(reaction@mid['damage'])
        final['diagnostics'].update(midpoint_iterations=iteration+1,midpoint_residual=error,
            midpoint_entropy_production=mid['diagnostics']['entropy_production'],
            midpoint_plastic_power=mid['diagnostics']['plastic_power'],midpoint_damage_power=mid['diagnostics']['damage_power'],
            midpoint_damage_obstacle_power=obstacle,
            newly_saturated_damage_nodes=int(np.count_nonzero((result.damage==1)&(state.damage<1))))
        return result,final['diagnostics']
