"""Balanced dislocation graphs and reciprocal finite-width interface coupling."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.special import xlogy
from .loops3d import LoopMechanics
from .continuum3d import Cube,CoupledCrystal,Parameters3D


@dataclass
class DislocationGraph:
    nodes: np.ndarray
    edges: np.ndarray
    burgers: np.ndarray
    core: float = .8
    core_coefficients: object = .25
    kinds: object = None
    projectors: object = None
    mobility: float = 1.

    def __post_init__(self):
        self.nodes=np.asarray(self.nodes,dtype=float)
        raw=np.asarray(self.edges)
        if not np.issubdtype(raw.dtype,np.integer):
            raise ValueError('Edge indices must be integers.')
        self.edges=raw.astype(int)
        self.burgers=np.asarray(self.burgers,dtype=float)
        if self.nodes.ndim!=2 or self.nodes.shape[1]!=3 or self.edges.ndim!=2 or self.edges.shape[1]!=2:
            raise ValueError('Expected 3D nodes and directed endpoint pairs.')
        count=len(self.edges)
        if count<3 or self.burgers.shape!=(count,3) or self.edges.min()<0 or self.edges.max()>=len(self.nodes):
            raise ValueError('Invalid graph indices or Burgers-vector shape.')
        if not np.isfinite(self.nodes).all() or not np.isfinite(self.burgers).all():
            raise ValueError('Geometry and charge must be finite.')
        self.core_coefficients=np.broadcast_to(np.asarray(self.core_coefficients,dtype=float),(count,)).copy()
        self.kinds=np.zeros(count,dtype=int) if self.kinds is None else np.asarray(self.kinds,dtype=int)
        self.projectors=np.broadcast_to(np.eye(3),(len(self.nodes),3,3)).copy() if self.projectors is None else np.asarray(self.projectors,dtype=float)
        if self.kinds.shape!=(count,) or not np.isin(self.kinds,[0,1]).all():
            raise ValueError('Edge kinds are mobile (0) or interface residual (1).')
        if self.projectors.shape!=(len(self.nodes),3,3) or not np.allclose(self.projectors,self.projectors.swapaxes(1,2)) or not np.allclose(self.projectors@self.projectors,self.projectors):
            raise ValueError('Node mobility projectors must be orthogonal projectors.')
        if min(self.core,self.mobility)<=0 or self.core_coefficients.min()<0 or self.lengths.min()<1e-10 or np.any(np.linalg.norm(self.burgers,axis=1)==0):
            raise ValueError('Invalid core, mobility, core coefficient or segment.')
        if self.balance_error>1e-10*max(1.,float(np.max(np.abs(self.burgers)))):
            raise ValueError('Burgers vectors must balance at every graph node.')
        if len(np.unique(self.edges))!=len(self.nodes):
            raise ValueError('Unused nodes must be removed from the graph.')

    @property
    def segments(self): return self.nodes[self.edges[:,1]]-self.nodes[self.edges[:,0]]

    @property
    def lengths(self): return np.linalg.norm(self.segments,axis=1)

    @property
    def balance_error(self):
        charge=np.zeros_like(self.nodes)
        np.add.at(charge,self.edges[:,0],-self.burgers)
        np.add.at(charge,self.edges[:,1],self.burgers)
        return float(np.max(np.abs(charge)))

    @property
    def tensions(self): return self.core_coefficients*np.sum(self.burgers**2,axis=1)

    @property
    def core_energy(self): return float(self.tensions@self.lengths)

    def moved(self,nodes): return replace(self,nodes=np.array(nodes,copy=True))

    def to_dict(self):
        return {key:getattr(self,key).tolist() for key in ('nodes','edges','burgers','core_coefficients','kinds','projectors')}|{'core':self.core,'mobility':self.mobility}


def tangent_projector(normals):
    normals=np.asarray(normals,dtype=float).reshape(-1,3)
    if len(normals)==0: return np.eye(3)
    _,singular,vt=np.linalg.svd(normals,full_matrices=False)
    basis=vt[singular>1e-12]
    return np.eye(3)-basis.T@basis


class GraphMechanics(LoopMechanics):
    def sample_phase(self,phase,points,core):
        coefficients=np.fft.fftn(phase)*np.exp(-core**2*self.k2/2)/self.grid.n**3
        return np.array([np.sum(coefficients*np.exp(1j*np.einsum('i...,i->...',self.k,point-self.origin))).real for point in points])

    def source(self,graph):
        alpha=np.zeros((3,3)+self.grid.shape,complex)
        gaussian=np.exp(-graph.core**2*self.k2/2)
        area=np.zeros((3,3))
        for (i,j),b,segment in zip(graph.edges,graph.burgers,graph.segments):
            midpoint=(graph.nodes[i]+graph.nodes[j])/2-self.origin
            phase=np.exp(-1j*np.einsum('i...,i->...',self.k,midpoint))
            integral=np.sinc(np.einsum('i...,i->...',self.k,segment)/(2*np.pi))
            alpha+=np.outer(b,segment)[...,None,None,None]*gaussian*phase*integral/self.grid.h**3
            area+=np.outer(b,np.cross(graph.nodes[i],graph.nodes[j]))/2
        beta=np.array([-1j*np.cross(self.k,row,axisa=0,axisb=0,axisc=0)*self.elastic.inverse_k2 for row in alpha])
        beta[:,:,0,0,0]=-area/self.grid.h**3
        distortion=np.fft.ifftn(beta,axes=(-3,-2,-1)).real
        divergence=1j*np.einsum('j...,ij...->i...',self.k,alpha)
        curl=np.array([-1j*np.cross(self.k,row,axisa=0,axisb=0,axisc=0) for row in beta])
        norm=max(1.,float(np.linalg.norm(alpha)))
        return {'eigenstrain':(distortion+distortion.swapaxes(0,1))/2,'alpha_hat':alpha,'distortion':distortion,
                'divergence_relative':float(np.linalg.norm(divergence)/norm),'curl_relative':float(np.linalg.norm(curl-alpha)/norm)}

    def interactions(self,graph,stress,phase=None,mobile_barrier=0.,residual_binding=0.):
        """One quadrature for energy, endpoint forces and the field adjoint."""
        if min(mobile_barrier,residual_binding)<0: raise ValueError('Interface coefficients must be nonnegative.')
        gaussian=np.exp(-graph.core**2*self.k2/2)
        stress_hat=(np.fft.fftn(stress,axes=(-3,-2,-1))*gaussian/self.grid.n**3).reshape(3,3,-1)
        vhat=np.zeros(self.grid.shape,complex) if phase is None else np.fft.fftn(phase)*gaussian/self.grid.n**3
        field_hat=np.zeros(self.grid.shape,complex)
        forces=np.zeros_like(graph.nodes); interface_energy=0.; samples=[]
        for e,((i,j),b,segment,length) in enumerate(zip(graph.edges,graph.burgers,graph.segments,graph.lengths)):
            tangent=segment/length; normb=float(b@b)
            forces[i]+=graph.tensions[e]*tangent; forces[j]-=graph.tensions[e]*tangent
            for s,w in zip(self.s,self.weight):
                position=graph.nodes[i]+s*segment-self.origin
                exponential=np.exp(1j*np.einsum('i...,i->...',self.k,position))
                sigma=np.einsum('ijk,k->ij',stress_hat,exponential.ravel()).real
                pk=np.cross(sigma@b,segment)
                forces[i]+=w*(1-s)*pk; forces[j]+=w*s*pk
                value=float(np.sum(vhat*exponential).real)
                grad=np.sum(1j*self.k*vhat*exponential,axis=(1,2,3)).real
                if graph.kinds[e]==0:
                    psi=mobile_barrier*(1-value*value)**2
                    derivative=4*mobile_barrier*value*(value*value-1)
                else:
                    psi=residual_binding*value*value
                    derivative=2*residual_binding*value
                interface_energy+=normb*length*w*psi
                forces[i]+=normb*w*(tangent*psi-length*(1-s)*derivative*grad)
                forces[j]+=normb*w*(-tangent*psi-length*s*derivative*grad)
                field_hat+=normb*length*w*derivative*gaussian*np.conjugate(exponential)/self.grid.h**3
                samples.append(value)
        return {'forces':forces,'interface_energy':interface_energy,'field_force':np.fft.ifftn(field_hat).real,
                'sampled_phase':np.array(samples)}

    def evaluate(self,graph,mean_strain=None,phase=None,mobile_barrier=0.,residual_binding=0.):
        source=self.source(graph)
        elastic=self.elastic.solve(source['eigenstrain'],mean_strain)
        interaction=self.interactions(graph,elastic['stress'],phase,mobile_barrier,residual_binding)
        return source|elastic|interaction|{'energy':elastic['energy']+graph.core_energy+interaction['interface_energy']}


def graph_state(crystal,fields,graph,mechanics,mobile_barrier=.2,residual_binding=.4):
    source=mechanics.source(graph)
    value=crystal.evaluate(*fields,extra_eigenstrain=source['eigenstrain'])
    interaction=mechanics.interactions(graph,value['stress'],fields[0],mobile_barrier,residual_binding)
    value['components']['graph_core']=graph.core_energy
    value['components']['graph_interface']=interaction['interface_energy']
    value['energy']+=graph.core_energy+interaction['interface_energy']
    gv=value['forces'][0]+interaction['field_force']
    value['dissipation']-=crystal.p.phase_mobility*crystal.grid.integrate(value['forces'][0]**2)
    value['forces']=(gv,*value['forces'][1:])
    value['rates']=(-crystal.p.phase_mobility*gv,*value['rates'][1:])
    value['dissipation']+=crystal.p.phase_mobility*crystal.grid.integrate(gv**2)
    measure=np.zeros(len(graph.nodes))
    for endpoint in (0,1): np.add.at(measure,graph.edges[:,endpoint],graph.lengths/2)
    force=interaction['forces']
    rate=graph.mobility*np.einsum('nij,nj->ni',graph.projectors,force)/measure[:,None]
    value['node_forces']=force; value['node_rates']=rate
    value['dissipation']+=float(np.sum(force*rate))
    value['source_divergence']=source['divergence_relative']; value['source_curl']=source['curl_relative']
    value['sampled_phase']=interaction['sampled_phase']
    return value


def graph_step(crystal,fields,graph,mechanics,dt,current=None,mobile_barrier=.2,residual_binding=.4):
    if dt<=0: raise ValueError('Timestep must be positive.')
    current=graph_state(crystal,fields,graph,mechanics,mobile_barrier,residual_binding) if current is None else current
    used=dt
    for attempt in range(1,25):
        candidate=tuple(f+used*r for f,r in zip(fields,current['rates']))
        if candidate[1].min()<=0 or candidate[1].max()>=1:
            used*=.5; continue
        try: new_graph=graph.moved(graph.nodes+used*current['node_rates'])
        except ValueError:
            used*=.5; continue
        new=graph_state(crystal,candidate,new_graph,mechanics,mobile_barrier,residual_binding)
        bound=current['energy']-.1*used*current['dissipation']+1e-12*max(1.,abs(current['energy']))
        if np.isfinite(new['energy']) and new['energy']<=bound:
            return candidate,new_graph,new,used,attempt
        used*=.5
    raise RuntimeError('Graph/field step failed energy or population bounds.')


def encounter_graphs(angle_degrees=25.,core=.8,burgers=.2,transmission_depth=.4,reflection_depth=.9,base_graph=None):
    """One local finite-embryo encounter; graph charges are exact in every branch."""
    if not 0<=angle_degrees<85 or min(transmission_depth,reflection_depth)<=0:
        raise ValueError('Invalid local encounter angle or embryo depth.')
    a=np.deg2rad(angle_degrees)
    rotation=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    bin=np.array([burgers,0.,0.]); bout=rotation@bin; residual=bin-bout
    nodes=np.array([[0.,-1.,0.],[-.35,0.,0.],[0.,1.,0.],[-3.,0.,0.]])
    edges=np.array([[0,1],[1,2],[2,3],[3,0]])
    plane=tangent_projector([[0,0,1]])
    intersection=tangent_projector([[1,0,0],[0,0,1]])
    projectors=np.array([intersection,plane,intersection,plane])
    incident=DislocationGraph(nodes,edges,np.tile(bin,(4,1)),core=core,projectors=projectors)
    if base_graph is not None:
        if not np.array_equal(base_graph.edges,edges) or not np.allclose(base_graph.burgers,np.tile(bin,(4,1))) or base_graph.core!=core:
            raise ValueError('The local encounter requires the declared incident cycle and Burgers vector.')
        incident=base_graph;nodes=incident.nodes.copy();projectors=incident.projectors.copy()
    midpoint=(nodes[0]+nodes[2])/2
    tn=nodes.copy(); tn[1]=midpoint+rotation@np.array([transmission_depth,nodes[1,1]-midpoint[1],0.])
    tb=np.array([bout,bout,bin,bin]); te=edges.copy(); kinds=np.zeros(4,int); coefficients=np.full(4,.25)
    if np.linalg.norm(residual)>1e-14:
        te=np.vstack([te,[0,2]]);tb=np.vstack([tb,residual]);kinds=np.r_[kinds,1];coefficients=np.r_[coefficients,.15]
    tp=projectors.copy();tp[1]=tangent_projector([rotation@np.array([0.,0.,1.])])
    transmitted=DislocationGraph(tn,te,tb,core=core,kinds=kinds,core_coefficients=coefficients,projectors=tp)
    absorbed=DislocationGraph(nodes[[0,2,3]],np.array([[0,1],[1,2],[2,0]]),np.tile(bin,(3,1)),core=core,
                               kinds=np.array([1,0,0]),core_coefficients=np.array([.15,.25,.25]),projectors=projectors[[0,2,3]])
    rn=nodes.copy();rn[1,0]=-reflection_depth
    reflected=incident.moved(rn)
    return {'incident':incident,'transmitted':transmitted,'absorbed':absorbed,'reflected':reflected}


def frozen_reaction_family(energies,thermal_energy=.05,barriers=(.02,.02,.02),gate=1.,attempt_rate=1.):
    energies=np.asarray(energies,dtype=float)
    if energies.shape!=(4,) or not np.isfinite(energies).all() or np.shape(barriers)!=(3,) or thermal_energy<=0 or min(barriers)<0 or not 0<gate<=1 or attempt_rate<=0:
        raise ValueError('Invalid frozen encounter energies or kinetic parameters.')
    rates=np.zeros((4,4))
    for j,barrier in enumerate(barriers,1):
        saddle=max(energies[0],energies[j])+barrier
        rates[0,j]=attempt_rate*gate*np.exp(-(saddle-energies[0])/thermal_energy)
        rates[j,0]=attempt_rate*gate*np.exp(-(saddle-energies[j])/thermal_energy)
    generator=rates.T.copy();np.fill_diagonal(generator,-rates.sum(axis=1))
    stationary=np.exp(-(energies-energies.min())/thermal_energy);stationary/=stationary.sum()
    if rates[0].sum()==0:
        raise ValueError('All first-event rates underflow; choose a representable kinetic time/energy scale.')
    return {'rates':rates,'generator':generator,'stationary':stationary,
            'first_event':rates[0]/rates[0].sum()}


def ensemble_trajectory(energies,family,times,thermal_energy):
    initial=np.array([1.,0.,0.,0.])
    probability=np.array([expm(family['generator']*t)@initial for t in times])
    free=probability@energies+thermal_energy*np.sum(xlogy(probability,probability),axis=1)
    return probability,free


def save_graph_checkpoint(path,crystal,fields,graph,mechanics,time,dt,step,rng,
                          mobile_barrier=.2,residual_binding=.4,reaction_context=None):
    metadata={'grid':{'length':crystal.grid.length,'n':crystal.grid.n},'parameters':asdict(crystal.p),
               'mean_strain':crystal.mean_strain.tolist(),'schmid':crystal.schmid.tolist(),
               'lam':crystal.elastic.lam,'shear':crystal.elastic.shear,'graph':graph.to_dict(),
               'quadrature_order':mechanics.quadrature_order,'mobile_barrier':mobile_barrier,
               'residual_binding':residual_binding,'time':float(time),'dt':float(dt),'step':int(step),
               'rng_state':rng.bit_generator.state,'reaction_context':reaction_context}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,phase=fields[0],population=fields[1],slip=fields[2],metadata=json.dumps(metadata))


def load_graph_checkpoint(path):
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata']));fields=tuple(data[key].copy() for key in ('phase','population','slip'))
    grid=Cube(**metadata['grid'])
    crystal=CoupledCrystal(grid,Parameters3D(**metadata['parameters']),np.array(metadata['mean_strain']),
                           np.array(metadata['schmid']),metadata['lam'],metadata['shear'])
    mechanics=GraphMechanics(grid,metadata['lam'],metadata['shear'],metadata['quadrature_order'])
    graph=DislocationGraph(**metadata['graph']);rng=np.random.default_rng();rng.bit_generator.state=metadata['rng_state']
    clock={key:metadata[key] for key in ('time','dt','step','mobile_barrier','residual_binding','reaction_context')}
    return crystal,fields,graph,mechanics,rng,clock
