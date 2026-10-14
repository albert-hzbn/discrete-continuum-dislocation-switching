"""Check Section 26 derivatives; this is not a spatial or time-evolution solver."""
import json,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]


def evaluate(F,Fp,c,eta,T,gamma):
    lam,mu,cv,H=2.,1.,10.,.4
    h=eta**2*(3-2*eta);beta=.02*(c-.2)+.015*h+.005*(T-1.)
    a=np.exp(beta);Fe=F@np.linalg.inv(Fp)/a;Je=np.linalg.det(Fe);log=np.log(Je);Ce=Fe.T@Fe
    Pe=mu*Fe+(lam*log-mu)*np.linalg.inv(Fe).T
    M=Fe.T@Pe;P=Pe@np.linalg.inv(Fp).T/a
    elastic=mu/2*(np.trace(Ce)-3)-mu*log+lam/2*log**2
    energy=elastic+.5*(c-.2)**2+eta**2*(1-eta)**2+H/2*(gamma@gamma)+cv*(T-1-T*np.log(T))
    entropy=cv*np.log(T)+.005*np.trace(M)
    return {'energy':energy,'elastic':elastic,'P':P,'M':M,'sigma':P@F.T/np.linalg.det(F),
            'gc':c-.2-.02*np.trace(M),'geta':2*eta*(1-eta)*(1-2*eta)-.015*6*eta*(1-eta)*np.trace(M),
            'entropy':entropy,'internal_energy':energy+T*entropy,
            'heat_capacity':cv-T*.005**2*(2*mu*np.trace(Ce)+9*lam)}


def main():
    F=np.array([[1.12,.07,0],[0,.96,.02],[.01,0,1.03]])
    S=np.zeros((2,3,3));S[0,0,1]=1.;S[1,1,2]=1.
    Fp=expm(.08*S[0]-.03*S[1]);gamma=np.array([.08,-.03]);c=.3;eta=.4;T=1.1;step=1e-5
    base=evaluate(F,Fp,c,eta,T,gamma);errors={}
    numerical=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            direction=np.zeros((3,3));direction[i,j]=step
            numerical[i,j]=(evaluate(F+direction,Fp,c,eta,T,gamma)['energy']-evaluate(F-direction,Fp,c,eta,T,gamma)['energy'])/(2*step)
    errors['piola_derivative']=float(np.max(np.abs(numerical-base['P'])))
    for index,key in enumerate(('gc','geta','entropy')):
        plus=[c,eta,T];minus=plus.copy();plus[index]+=step;minus[index]-=step
        numerical=(evaluate(F,Fp,*plus,gamma)['energy']-evaluate(F,Fp,*minus,gamma)['energy'])/(2*step)
        target=-base[key] if key=='entropy' else base[key]
        errors[key+'_derivative']=float(abs(numerical-target))
    numerical=T*(evaluate(F,Fp,c,eta,T+step,gamma)['entropy']-evaluate(F,Fp,c,eta,T-step,gamma)['entropy'])/(2*step)
    errors['heat_capacity_derivative']=float(abs(numerical-base['heat_capacity']))
    Q=expm(np.array([[0,-.3,.1],[.3,0,-.2],[-.1,.2,0]]))
    rotated=evaluate(Q@F,Fp,c,eta,T,gamma)
    errors['frame_objectivity']=float(max(abs(rotated['energy']-base['energy']),np.max(np.abs(rotated['P']-Q@base['P']))))
    errors['cauchy_symmetry']=float(np.max(np.abs(base['sigma']-base['sigma'].T)))
    q=np.einsum('ij,sij->s',base['M'],S)-.4*gamma
    rates=np.sign(q)*np.maximum(np.abs(q)-.01,0)/2
    Lp=np.einsum('s,sij->ij',rates,S);phase_rate=-base['geta']
    def perturbed(t,temperature=T):
        return evaluate(F,expm(t*Lp)@Fp,c,eta+t*phase_rate,temperature,gamma+t*rates)
    energy_rate=(perturbed(step)['energy']-perturbed(-step)['energy'])/(2*step)
    dissipation=base['geta']**2+q@rates
    errors['plastic_and_phase_derivative']=float(abs(energy_rate+dissipation))
    entropy_state_rate=(perturbed(step)['entropy']-perturbed(-step)['entropy'])/(2*step)
    temperature_rate=(dissipation-T*entropy_state_rate)/base['heat_capacity']
    energy_state_rate=(perturbed(step)['internal_energy']-perturbed(-step)['internal_energy'])/(2*step)
    errors['adiabatic_internal_energy_rate']=float(abs(energy_state_rate+base['heat_capacity']*temperature_rate))
    errors['plastic_determinant']=float(abs(np.linalg.det(expm(.2*Lp)@Fp)-1))
    linear=[];A=np.array([[.3,.2,0],[-.1,-.2,.1],[0,0,.05]])
    target=2*(A+A.T)/2+2*np.trace(A)*np.eye(3)
    for epsilon in (.01,.005,.0025):
        state=evaluate(np.eye(3)+epsilon*A,np.eye(3),.2,0.,1.,np.zeros(2))
        linear.append({'epsilon':epsilon,'relative_stress_error':float(np.linalg.norm(state['P']/epsilon-target)/np.linalg.norm(target))})
    checks={key:value<1e-7 for key,value in errors.items()}
    checks['positive_heat_capacity']=base['heat_capacity']>0
    checks['nonnegative_dissipation']=dissipation>=0
    checks['small_strain_limit']=all(a['relative_stress_error']>b['relative_stress_error'] for a,b in zip(linear,linear[1:])) and linear[-1]['relative_stress_error']<.01
    report={'scope':'Homogeneous algebra/rate checks of the stated specialization; no spatial thermomechanical solver or integrated trajectory.',
            'normalization':'Dimensionless test state, not a calibrated material or Kelvin-valued experiment.',
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
            'finite_difference_step':step,'errors':errors,'small_strain_refinement':linear,
            'heat_capacity':float(base['heat_capacity']),'dissipation':float(dissipation),'adiabatic_temperature_rate':float(temperature_rate),
            'checks':{k:bool(v) for k,v in checks.items()},'all_checks_pass':all(checks.values())}
    output=ROOT/'results/thermomechanical-algebra-2026-09-13';output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
