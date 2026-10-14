"""Section 74: stacking faults, disclinations, twin boundaries and antisites against independent results; declared gates."""
import argparse,json
from pathlib import Path
import numpy as np
from bmvr import extended_defects as X
from bmvr.grid import Grid

MU,NU=1.,.324


def stacking(n=1024,length=1024.):
    rows=[]
    for deg in (0,30,60,90):
        beta=np.radians(deg);b1,b2=X.partial_burgers(beta);K=b1[1]*b2[1]+b1[0]*b2[0]/(1-NU);gamma=MU*K/(2*np.pi*40.)
        ds=np.arange(24,64);E=np.array([X.ribbon_energy(n,length,int(d),beta,gamma,MU,NU) for d in ds]);i=int(np.argmin(E))
        c=np.polyfit(ds[i-3:i+4],E[i-3:i+4],2);d_fft=-c[1]/(2*c[0]);d_ref=X.analytic_separation(beta,gamma,MU,NU)
        rows.append(dict(beta_deg=deg,fault_energy=gamma,d_fft=float(d_fft),d_analytic=float(d_ref),rel_error=float(abs(d_fft-d_ref)/d_ref)));print(rows[-1],flush=True)
    cu=dict(mu=54.6e9,nu=.324,b=.255e-9)
    copper={f'{g*1e3:g} mJ/m2':{f'{deg}':float(X.analytic_separation(np.radians(deg),g,cu['mu'],cu['nu'],cu['b'])*1e9) for deg in (0,90)} for g in (.041,.045,.058)}
    return rows,copper


def disclination(n=1024,length=1024.,a=8.,width=1.5):
    b=1.;w=b/(2*a);c=length/2;x=(np.arange(n)+.5)*length/n;Xg,Yg=np.meshgrid(x,x,indexing='ij');dx,dy=Xg-c,Yg-c;r=np.hypot(dx,dy)
    g=X.gaussian(n,length,c,c,width);dgdy=-(dy/width**2)*g
    edge=X.airy_stress(-b*dgdy,length,MU,NU)
    D=MU/(2*np.pi*(1-NU));ref=D*b*dx*(dx**2-dy**2)/np.maximum(r,1e-9)**4
    ann=(r>=16)&(r<=64);R1=float(np.sqrt(np.sum((edge['xy'][ann]-ref[ann])**2)/np.sum(ref[ann]**2)))
    eta=w*(X.gaussian(n,length,c,c+a,width)-X.gaussian(n,length,c,c-a,width));dip=X.airy_stress(eta,length,MU,NU)
    far=(r>=4*a)&(r<=128);near=r<a
    diff=lambda m:float(np.sqrt(sum(np.sum((dip[k][m]-edge[k][m])**2) for k in ('xx','yy','xy'))/sum(np.sum(edge[k][m]**2) for k in ('xx','yy','xy'))))
    return dict(R1_edge_vs_analytic=R1,R2_far=diff(far),R2_near=diff(near),dipole_charge=float(eta.sum()*(length/n)**2),single_charge=float(w*X.gaussian(n,length,c,c+a,width).sum()*(length/n)**2))


def twins():
    grid=Grid.uniform(48.,96);rows=[]
    for ratio in (.5,1.,1.5):
        a=[[0,ratio,1.],[ratio,0,1.],[1.,1.,0]];phases,info=X.pairwise_junction(grid,a)
        measured=X.phase_sectors(grid,phases,radius=5.);pred=X.young_sectors(ratio,1.,1.)
        rows.append(dict(ratio=ratio,measured=measured,predicted=pred,max_error=float(max(abs(m-p) for m,p in zip(measured,pred))),**info));print(rows[-1],flush=True)
    cu_ratio=.024/.6;return rows,dict(ratio=cu_ratio,predicted=X.young_sectors(cu_ratio,1.,1.))


def antisites():
    order={f'{T:g}':dict(relaxed=float(X.relax_order(T)),self_consistent=float(X.bw_order(T))) for T in (.5,.9,.99,1.1)}
    for v in order.values():v['error']=abs(v['relaxed']-v['self_consistent'])
    x,eta,E=X.apb_profile(.5);ref=X.apb_analytic(.5)
    return order,dict(T=.5,relaxed=E,analytic=ref,rel_error=abs(E-ref)/ref)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    sf,cu=stacking();dc=disclination();tw,twcu=twins();order,apb=antisites()
    gates=dict(S1_stacking=all(r['rel_error']<=.05 for r in sf),R1_edge_solver=dc['R1_edge_vs_analytic']<=.05,R2_disclination_far_near=dc['R2_far']<=.05 and dc['R2_near']>.5,
        T1_twin_junction=all(r['max_error']<=3. and r['converged'] for r in tw),A1_order=all(v['error']<=1e-6 for v in order.values()),A2_apb=apb['rel_error']<=.02)
    report=dict(scope='Section 74 extended defect types; declared gates.',stacking=sf,stacking_copper_nm=cu,disclination=dc,twins=tw,twin_copper=twcu,antisite_order=order,apb=apb,gates=gates)
    (out/'report.json').write_text(json.dumps(report,indent=1)+'\n');print(gates,flush=True)


if __name__=='__main__':main()
