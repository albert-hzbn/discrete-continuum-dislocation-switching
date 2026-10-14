"""Local fracture and topology-ledger algebra; no spatial crack/event simulation."""
import json,platform,sys
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bmvr.continuum3d import Cube
from bmvr.loops3d import circular_loop,LoopMechanics


def split(strain):
    values,vectors=np.linalg.eigh(strain)
    positive=(vectors*np.maximum(values,0))@vectors.T
    negative=(vectors*np.minimum(values,0))@vectors.T
    trace=np.trace(strain)
    plus=max(trace,0)**2+np.sum(positive**2)
    minus=min(trace,0)**2+np.sum(negative**2)
    return plus,minus,2*max(trace,0)*np.eye(3)+2*positive,2*min(trace,0)*np.eye(3)+2*negative


def homogeneous(strain,q,Gc=1.,ell=.5,residual=1e-6):
    plus,minus,sp,sm=split(strain);degradation=(1-q)**2+residual
    return {'energy':degradation*plus+minus+Gc*q*q/(2*ell),
            'stress':degradation*sp+sm,'force':-2*(1-q)*plus+Gc*q/ell,'tensile_energy':plus}


def main():
    errors={};checks={};strain=np.array([[.02,.006,0],[.006,-.009,.002],[0,.002,-.003]])
    q=.3;step=1e-6;base=homogeneous(strain,q);rng=np.random.default_rng(2701);derivative_errors=[]
    for _ in range(6):
        direction=rng.normal(size=(3,3));direction=(direction+direction.T)/2
        numerical=(homogeneous(strain+step*direction,q)['energy']-homogeneous(strain-step*direction,q)['energy'])/(2*step)
        derivative_errors.append(abs(numerical-np.sum(base['stress']*direction)))
    errors['spectral_stress_derivative']=float(max(derivative_errors))
    numerical=(homogeneous(strain,q+step)['energy']-homogeneous(strain,q-step)['energy'])/(2*step)
    errors['damage_derivative']=float(abs(numerical-base['force']))
    Q=expm(np.array([[0,-.3,.1],[.3,0,0],[-.1,0,0]]));rotated=homogeneous(Q@strain@Q.T,q)
    errors['spectral_objectivity']=float(max(abs(rotated['energy']-base['energy']),np.max(np.abs(rotated['stress']-Q@base['stress']@Q.T))))
    profile=[]
    for ell in (.2,.5,1.):
        Gc=1.4
        energy=2*quad(lambda s:Gc*np.exp(-2*s/ell)/ell,0,np.inf,epsabs=1e-12)[0]
        profile.append({'ell':ell,'energy_per_area':energy,'target':Gc})
    errors['crack_profile_normalization']=max(abs(r['energy_per_area']-r['target']) for r in profile)
    A=2.;psi=.1;mobility=.7;q0=.01;target=2*psi/(A+2*psi)
    solution=lambda t:target+(q0-target)*np.exp(-mobility*(A+2*psi)*t)
    local_errors=[]
    for time in (.1,.5,2.):
        numerical=(solution(time+step)-solution(time-step))/(2*step)
        local_errors.append(abs(numerical-mobility*(2*(1-solution(time))*psi-A*solution(time))))
    errors['homogeneous_damage_rate']=float(max(local_errors))
    checks['no_compressive_growth']=homogeneous(-.02*np.eye(3),0)['tensile_energy']==0
    checks['no_healing_under_unload']=max(0.,-homogeneous(np.zeros((3,3)),.4)['force'])==0
    checks['no_finite_intact_AT2_threshold']=2*1e-8/(A+2*1e-8)>0
    E,Gc,ell=3.,.002,.4;A=Gc/ell;critical=np.sqrt(A/(3*E))
    stress=lambda e:E*e*(A/(A+E*e*e))**2
    expected=3*np.sqrt(3)/16*np.sqrt(E*Gc/ell)
    errors['homogeneous_peak_stress']=float(abs(stress(critical)-expected))
    checks['peak_is_local_maximum']=stress(critical)>max(stress(.999*critical),stress(1.001*critical))
    stoichiometry=np.array([1,-1,-1]);ledgers=np.array([[1,1,0],[1,0,1]])
    checks['recombination_atom_and_site_balance']=bool(np.all(ledgers@stoichiometry==0))
    count=32;vacancies=100.;atomic_volume=.01;burgers=.2
    radius=np.sqrt(vacancies*atomic_volume/(burgers*count*np.sin(2*np.pi/count)/2))
    loop=circular_loop(radius=radius,count=count,burgers=(0,0,burgers))
    grid=Cube(12.,17);mechanics=LoopMechanics(grid)
    source=mechanics.source([loop]);plastic_volume=grid.integrate(np.einsum('ii...->...',source['distortion']))
    errors['prismatic_inventory_from_source']=float(abs(-plastic_volume/atomic_volume-vacancies))
    shifted=loop.moved(loop.nodes+np.array([.7,-.2,.1]))
    errors['prismatic_origin_independence']=float(abs(loop.burgers@(loop.area-shifted.area))/atomic_volume)
    glide=circular_loop(radius=radius,count=count,burgers=(burgers,0,0))
    checks['glide_loop_zero_volumetric_inventory']=abs(glide.burgers@glide.area)<1e-12
    rotation=expm(np.array([[0,-np.pi/2,0],[np.pi/2,0,0],[0,0,0]]))
    b1=np.array([1.,0,0]);b2=np.array([0.,1.,0]);actual=b1+rotation@b2;naive=b1+b2
    errors['rigid_motion_inverse']=float(np.linalg.norm(b1+rotation@(-rotation.T@b1)))
    checks['rotation_translation_not_naively_additive']=np.linalg.norm(actual-naive)>1
    for name,error in errors.items():checks[name]=error<1e-7
    report={'scope':'Local spectral-fracture derivatives, analytical damage/profile identities and topology inventory algebra only.',
            'not_verified':['Spatial crack initiation/branching/arrest','Prismatic-loop creation kinetics or climb','Disclination constitutive dynamics'],
            'normalization':'Dimensionless synthetic checks, except the declared dimensional identities in the theory.',
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
            'finite_difference_step':step,'errors':errors,'profile_checks':profile,
            'homogeneous_peak':{'strain':float(critical),'stress':float(expected)},
            'prismatic_ledger':{'vacancies':vacancies,'atomic_volume':atomic_volume,'plastic_volume':plastic_volume,'polygon_radius':float(radius)},
            'rigid_motion_counterexample':{'correct_translation':actual.tolist(),'naive_translation':naive.tolist()},
            'checks':{k:bool(v) for k,v in checks.items()},'all_checks_pass':all(checks.values())}
    folder=ROOT/'results/fracture-topology-algebra-2026-09-13';folder.mkdir(parents=True,exist_ok=True)
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
