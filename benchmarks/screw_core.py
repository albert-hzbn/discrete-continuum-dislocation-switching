"""Section 82: molecular statics of the dissociated screw dislocation core in copper; declared gates P1-P4 (potential) and C1 (separation).

--potential emt reproduces the blocked Section 82.1 check (negative fault energy); --potential zjw uses the analytical
Zhou-Johnson-Wadley EAM of Section 82.2."""
import argparse,json,time
from pathlib import Path
import numpy as np
from ase.build import bulk
from ase.lattice.cubic import FaceCenteredCubic
from ase.calculators.emt import EMT
from ase.optimize import FIRE
from ase.constraints import FixAtoms,FixedLine
from scipy.optimize import minimize_scalar
from functools import partial
from bmvr.eam_zjw import ZhouEAM
from bmvr.eam_setfl import SetflEAM
from bmvr import extended_defects as X
from bmvr.extended_defects import analytic_separation

EV=1.602176634e-19;TO_GPA=EV*1e30/1e9;TO_MJ=EV*1e20*1e3
MEASURED=dict(a0=3.615,Ec=3.49,B=138.,C44=75.4,sfe=(41.,58.))
ROOT=Path(__file__).resolve().parents[1]
CALCS=dict(emt=EMT,zjw=ZhouEAM,mishin=partial(SetflEAM,ROOT/'data'/'potentials'/'Cu01.eam.alloy'))


def lattice(calc):
    def E(a):
        b=bulk('Cu','fcc',a=a,cubic=True);b.calc=calc();return b.get_potential_energy()/len(b)
    r=minimize_scalar(E,bracket=(3.5,3.6,3.7),tol=1e-10);return float(r.x),float(-r.fun)


def elastic(calc,a0):
    at=bulk('Cu','fcc',a=a0,cubic=True).repeat((3,3,3));at.calc=calc();V=at.get_volume();e0=at.get_potential_energy();cell=at.cell.copy();h=1e-3
    def E(eps):
        b=at.copy();b.calc=calc();b.set_cell(cell@(np.eye(3)+eps),scale_atoms=True);return b.get_potential_energy()
    ex=np.zeros((3,3));ex[0,0]=1;ey=np.zeros((3,3));ey[1,1]=1;g=np.zeros((3,3));g[0,1]=g[1,0]=.5
    c11=(E(h*ex)+E(-h*ex)-2*e0)/h**2/V;c12=((E(h*(ex+ey))+E(-h*(ex+ey))-2*e0)/h**2/V-2*c11)/2;c44=(E(2*h*g)+E(-2*h*g)-2*e0)/(2*h)**2/V
    c11,c12,c44=c11*TO_GPA,c12*TO_GPA,c44*TO_GPA;mu=(c11-c12+3*c44)/5;lam=(c11+4*c12-2*c44)/5
    return dict(C11=c11,C12=c12,C44=c44,B=(c11+2*c12)/3,mu=mu,nu=lam/(2*(lam+mu)))


def stacking_fault(calc,a0):
    """Intrinsic fault energy (mJ/m^2). The upper half shifts by a/sqrt(6) along [11-2] and the cell is tilted by the same
    shift, so the periodic boundary stays perfect and the cell holds exactly one fault."""
    at=FaceCenteredCubic(directions=[[1,1,-2],[-1,1,0],[1,1,1]],size=(2,2,12),symbol='Cu',latticeconstant=a0,pbc=True)
    at.calc=calc();e0=at.get_potential_energy();top=at.positions[:,2]>at.positions[:,2].mean()
    shift=np.array([a0/np.sqrt(6),0,0]);f=at.copy();f.positions[top]+=shift;c=f.cell.copy();c[2]+=shift;f.set_cell(c,scale_atoms=False);f.calc=calc()
    f.set_constraint([FixedLine(i,[0,0,1]) for i in range(len(f))]);FIRE(f,logfile=None).run(fmax=.001,steps=4000)
    return (f.get_potential_energy()-e0)/np.linalg.norm(np.cross(at.cell[0],at.cell[1]))*TO_MJ


def cylinder(a0,R):
    """Perfect-crystal cylinder (x = [11-2], y = [111], z = [1-10], periodic along z), its axis x0 and the glide-plane height yc."""
    dirs=[[1,1,-2],[1,1,1],[1,-1,0]]
    lx,ly,_=FaceCenteredCubic(directions=dirs,size=(1,1,1),symbol='Cu',latticeconstant=a0).cell.lengths()  # a sqrt(6)/2, a sqrt(3), a/sqrt(2)
    at=FaceCenteredCubic(directions=dirs,size=(int(np.ceil(2*R/lx))+2,int(np.ceil(2*R/ly))+2,3),symbol='Cu',latticeconstant=a0,pbc=(False,False,True))
    c=at.positions.mean(axis=0);at=at[np.hypot(*(at.positions-c)[:,:2].T)<=R]
    ys=np.unique(np.round(at.positions[:,1],3));yc=ys[np.argmin(abs(ys-c[1]))]+a0/np.sqrt(3)/2  # between two (111) planes
    assert np.all(np.ptp(at.positions[:,:2],axis=0)>2*R-2*a0),'cylinder truncated by the box'
    return at,float(c[0]),float(yc)


def screw(calc,a0,R=60.,shell=8.,fmax=.01):
    b=a0/np.sqrt(2);at,x0,yc=cylinder(a0,R)
    ref=at.copy();theta=np.arctan2(at.positions[:,1]-yc,at.positions[:,0]-x0);at.positions[:,2]+=b*theta/(2*np.pi)
    at.set_constraint(FixAtoms(mask=np.hypot(at.positions[:,0]-x0,at.positions[:,1]-yc)>R-shell));at.calc=calc()
    start=time.time();opt=FIRE(at,logfile=None);opt.run(fmax=fmax,steps=20000)
    return at,ref,x0,yc,b,time.time()-start,opt.nsteps


def disregistry(at,ref,x0,yc,b,a0,window=1.3):
    """Screw disregistry u_z(above) - u_z(below) across the glide plane, from the displacement against the perfect crystal."""
    uz=at.positions[:,2]-ref.positions[:,2];x=ref.positions[:,0];y=ref.positions[:,1];d=a0/np.sqrt(3)
    above=(y>yc)&(y<yc+d);below=(y<yc)&(y>yc-d);xs=np.arange(x0-40,x0+40.01,.25)
    prof=lambda m:np.array([uz[m&(abs(x-xx)<window)].mean() for xx in xs])
    return xs,(prof(above)-prof(below))/b


def crossing(xs,dis,level):
    k=np.where((dis[:-1]-level)*(dis[1:]-level)<=0)[0]
    if len(k)!=1:raise RuntimeError(f'{len(k)} crossings of {level}')
    k=k[0];return float(xs[k]+(level-dis[k])*(xs[k+1]-xs[k])/(dis[k+1]-dis[k]))


def predictions(a0,el,sfe):
    b=a0/np.sqrt(2);C=X.cubic_stiffness(el['C11'],el['C12'],el['C44'])*1e9;t=np.array([1.,-1,0])/np.sqrt(2)
    b1,b2=a0/6*np.array([2.,-1,-1])*1e-10,a0/6*np.array([1.,-2,1])*1e-10
    return X.anisotropic_separation(C,t,b1,b2,sfe*1e-3)*1e10,float(analytic_separation(0.,sfe*1e-3,el['mu']*1e9,el['nu'],b*1e-10))*1e10


def converged_core(calc,a0,el,sfe,gates,out):
    """Section 82.6: R = 120 A (declared) and R = 60 A (reported), relaxed to 1e-4 eV/A."""
    d_an,d_iso=predictions(a0,el,sfe);res={}
    for R in (60.,120.):
        at,ref,x0,yc,b,sec,steps=screw(calc,a0,R=R,fmax=1e-4);xs,dis=disregistry(at,ref,x0,yc,b,a0)
        d=crossing(xs,dis,.25)-crossing(xs,dis,.75);res[f'{R:g}']=dict(atoms=len(at),fire_steps=steps,seconds=sec,separation_A=d)
        np.savez_compressed(out/f'screw-mishin-R{int(R)}.npz',relaxed=at.positions,reference=ref.positions,xs=xs,disregistry=dis);print(R,res[f'{R:g}'],flush=True)
    d=res['120']['separation_A'];gates['C2_bracket']=bool(d_an<=d<=d_iso)
    return dict(core=res,anisotropic_prediction_A=d_an,isotropic_prediction_A=d_iso)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--potential',choices=CALCS,default='zjw')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);calc=CALCS[args.potential]
    a0,ec=lattice(calc);el=elastic(calc,a0);sfe=stacking_fault(calc,a0);print('a0',a0,'Ec',ec,'elastic',el,'SFE',sfe,flush=True)
    gates=dict(P1_lattice=abs(a0/MEASURED['a0']-1)<=.005,P2_cohesive=abs(ec/MEASURED['Ec']-1)<=.03,
        P3_elastic=abs(el['B']/MEASURED['B']-1)<=.10 and abs(el['C44']/MEASURED['C44']-1)<=.15,P4_positive_fault=sfe>0)
    report=dict(potential=args.potential,a0=a0,cohesive_eV=ec,elastic_GPa=el,sfe_mJ_m2=sfe,measured=MEASURED,gates=gates)
    if args.potential=='mishin':
        gates.update(P0_reader=abs(a0/3.615-1)<=1e-3 and abs(ec/3.54-1)<=1e-3,P5_fault_band=MEASURED['sfe'][0]<=sfe<=MEASURED['sfe'][1])
        if all(gates.values()):report.update(converged_core(calc,a0,el,sfe,gates,out))
    elif all(gates.values()):
        at,ref,x0,yc,b,sec,steps=screw(calc,a0);xs,dis=disregistry(at,ref,x0,yc,b,a0)
        left,right=crossing(xs,dis,.75),crossing(xs,dis,.25);d_atom=right-left
        bp=a0/np.sqrt(6);d_iso=float(analytic_separation(0.,sfe*1e-3,el['mu']*1e9,el['nu'],b*1e-10))*1e10
        gates['C1_separation']=abs(d_atom/d_iso-1)<=.30
        np.savez_compressed(out/f'screw-{args.potential}.npz',relaxed=at.positions,reference=ref.positions,xs=xs,disregistry=dis)
        plateau=dis[(xs>left+.25*d_atom)&(xs<right-.25*d_atom)]
        report.update(screw=dict(atoms=len(at),b=b,b_partial=bp,relax_seconds=sec,fire_steps=steps,x_partials=[left,right],separation_A=d_atom,
            isotropic_prediction_A=d_iso,ratio=d_atom/d_iso,plateau_mean=float(plateau.mean()) if plateau.size else None))
    (out/f'report-{args.potential}.json').write_text(json.dumps(report,indent=1,default=float)+'\n');print(json.dumps(report,indent=1,default=float))


if __name__=='__main__':main()
