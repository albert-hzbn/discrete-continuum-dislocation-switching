"""Section 84.1 post hoc: the 45-degree discrete interface energy, and the barrier deviation against CNT with the orientation mean."""
import numpy as np,json
from scipy.optimize import minimize
from bmvr.critical_nucleus import planar_interface_energy
h,kappa,W,n=.5,1.,1.,400
def diag_gamma():
    # phi depends on s = i + j (mod n): level lines at 45 degrees; each s-class has n nodes and two interfaces of length n h sqrt(2)
    s=np.arange(n);w=np.sqrt(kappa/(2*W));x=s*h/np.sqrt(2);L=n*h/np.sqrt(2)
    phi0=.5*(np.tanh((x-L/4)/(2*w))-np.tanh((x-3*L/4)/(2*w)))
    def fun(y):
        phi=.5+y-y.mean();d=np.roll(phi,-1)-phi
        E=n*(kappa*np.sum(d*d)+h*h*np.sum(W*phi**2*(1-phi)**2))  # two neighbour directions each differ by one s-step
        g=n*(-2*kappa*(np.roll(phi,1)+np.roll(phi,-1)-2*phi)+h*h*2*W*phi*(1-phi)*(1-2*phi));return E,g-g.mean()
    r=minimize(fun,phi0-.5,jac=True,method='L-BFGS-B',options=dict(maxiter=100000,gtol=1e-12,ftol=1e-16))
    return float(r.fun/(2*n*h*np.sqrt(2)))
g0=planar_interface_energy();g45=diag_gamma();gbar=(g0+g45)/2
print('gamma_0',g0,'gamma_45',g45,'mean',gbar,'continuum',np.sqrt(2)/6)
r=json.load(open('results/critical-nucleus-2026-09-28/report.json'))
out=[]
for row in r['rows']:
    c=np.pi*gbar**2/row['dg'];out.append(dict(dg=row['dg'],deviation_mean_gamma=(row['barrier']-c)/c));print(row['dg'],row['deviation'],(row['barrier']-c)/c)
json.dump(dict(gamma_0=g0,gamma_45=g45,gamma_mean=gbar,rows=out),open('results/critical-nucleus-2026-09-28/post-hoc-anisotropy.json','w'),indent=1)
