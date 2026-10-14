"""Section 75: configurational entropy (natural n >= 0 barrier) and gas pressure in the Section 52/58 pore model; declared gates."""
import argparse,json
from pathlib import Path
import numpy as np
from bmvr.pore_topology import PoreModel,EntropyPoreModel,GasPoreModel,GAMMA


def run(model,n,p,t_end,limiter):
    E=model.energy(n,p);inv0=model.inventory(n,p);t=0.;min_n=float(n.min());count=[];steps=0
    while t<t_end-1e-12:
        n,p,E,used=model.step(n,p,E,dt=min(model.dt,t_end-t),limiter=limiter);t+=used;steps+=1;min_n=min(min_n,float(n.min()))
        if steps%200==0:count.append((t,model.components(p)[1]))
    return n,p,dict(t=t,steps=steps,min_n=min_n,inventory_error=abs(model.inventory(n,p)-inv0)/inv0,final_count=model.components(p)[1],count=count)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    res={}
    m=EntropyPoreModel(n_eq=.005,length=32.,h=.5);p=m.pore((0.,0.),4.);n=np.full(p.shape,.005)
    try:_,_,res['G1_entropy']=run(m,n,p,400.,limiter=False)
    except RuntimeError as e:res['G1_entropy']=dict(error=str(e))
    q=PoreModel(length=32.,h=.5);p=q.pore((0.,0.),4.);n=np.zeros(p.shape)
    try:_,_,res['G1_control_quadratic']=run(q,n,p,400.,limiter=False)
    except RuntimeError as e:res['G1_control_quadratic']=dict(error=str(e))
    print(res,flush=True)
    # G1c: first-order agreement of the two forms about n_eq (no pore)
    e=EntropyPoreModel(n_eq=.005,length=32.,h=.5);X=e.grid.X;n0=.005*(1+.01*np.cos(2*np.pi*X/32.));p0=np.zeros_like(n0)
    from bmvr.energy import pore_energy,pore_forces
    class Quad(PoreModel):
        def energy(self,n,p):return pore_energy(self.grid,n,p,chi=1.,neq=.005)
        def forces(self,n,p):return pore_forces(self.grid,n,p,chi=1.,neq=.005)
    qd=Quad(length=32.,h=.5);a,_,_=run(e,n0.copy(),p0.copy(),20.,False);b,_,_=run(qd,n0.copy(),p0.copy(),20.,False)
    res['G1c']=dict(relative_difference=float(np.max(abs(a-b))/np.max(abs(b-.005))),deviation_remaining=float(np.max(abs(b-.005))/np.max(abs(n0-.005))))
    rows=[]
    for g in (4.,7.,10.):
        mg=GasPoreModel(gas=g,theta_g=1.,length=64.,h=.5);p=mg.pore((0.,0.),8.);n=np.full(p.shape,.02)
        n,p,info=run(mg,n,p,600.,limiter=True);A=mg.grid.integrate(p);R=np.sqrt(A/np.pi);P=mg.pressure(p);mu=float(np.median(n))
        rows.append(dict(gas=g,R=float(R),P=float(P),mu=mu,lhs=float(mu+P),rhs=float(GAMMA/R),rel_error=float(abs(mu+P-GAMMA/R)/(GAMMA/R)),n_spread=float(n.max()-n.min()),**{k:info[k] for k in ('inventory_error','min_n')}));print(rows[-1],flush=True)
    res['G2_rows']=rows
    ent=res['G1_entropy'];ctl=res['G1_control_quadratic']
    gates=dict(G1_entropy_barrier='error' not in ent and ent['min_n']>0 and ent['inventory_error']<=1e-12 and ent['final_count']==0,
        G1_control_fails='error' in ctl or ctl['min_n']<0,G1c_linear=res['G1c']['relative_difference']<=.02,G2_laplace=all(r['rel_error']<=.03 for r in rows))
    res['gates']=gates;(out/'report.json').write_text(json.dumps(res,indent=1,default=float)+'\n');print(gates,flush=True)


if __name__=='__main__':main()
