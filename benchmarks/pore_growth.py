"""Section 52: sharp-interface pore growth; local interface-mobility test for two exchange mobilities."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bmvr.grid import Grid
from bmvr.energy import pore_energy,pore_forces

GAMMA=np.sqrt(2)/6;A=np.sqrt(2)
MOBILITIES=dict(single=(.5,1),squared=(10.,2))


def run(radius,excess,mobility,h,end=4.,transient=2.):
    k_ex,power=MOBILITIES[mobility];n_cells=round(64/h);grid=Grid.uniform(64,n_cells,periodic=True)
    x,y=np.meshgrid(grid.x,grid.y,indexing='ij');r=np.hypot(x,y)
    p=1/(1+np.exp(A*(r-radius)));n=np.full(grid.shape,excess);inventory=grid.integrate(n+p)
    dt=.2*h*h;time=0.;energy=pore_energy(grid,n,p);rows=[];increase=0.
    def sample():
        area=grid.integrate(p);gx,gy=np.gradient(p,h,h);weight=np.hypot(gx,gy)
        rows.append([time,area,float(np.sum(weight*n)/np.sum(weight)),grid.integrate(n+p)-inventory])
    sample()
    while time<end-1e-12:
        un,up=pore_forces(grid,n,p);rate=k_ex*(p*(1-p))**power*(up-un)
        dn,dp=grid.lap(un)+rate,-rate;used=min(dt,end-time)
        for _ in range(30):
            nn,pp=n+used*dn,p+used*dp;value=pore_energy(grid,nn,pp)
            if nn.min()>=0 and pp.min()>=0 and pp.max()<=1 and value<=energy+1e-11:break
            used*=.5
        else:raise RuntimeError('Pore step failed bounds or energy.')
        increase=max(increase,value-energy);n,p,energy=nn,pp,value;time+=used
        if abs(time/.05-round(time/.05))<1e-9:sample()
    rows=np.array(rows);late=rows[:,0]>=transient-1e-12
    t,area,mu_i=rows[late,0],rows[late,1],rows[late,2]
    slope=np.polyfit(t,area,1)[0];radius_a=np.sqrt(area.mean()/np.pi);velocity=slope/(2*np.pi*radius_a)
    drive=mu_i.mean()-GAMMA/radius_a
    return dict(radius=radius,excess=excess,mobility=mobility,h=h,velocity=float(velocity),interface_mu=float(mu_i.mean()),
        capillary=float(GAMMA/radius_a),drive=float(drive),effective_mobility=float(velocity/drive) if abs(drive)>1e-12 else None,
        predicted_mobility=k_ex/A if mobility=='single' else None,inventory_drift=float(np.max(np.abs(rows[:,3]))),max_energy_increase=float(increase))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);start=perf_counter();rows=[]
    for mobility in MOBILITIES:
        for h in (.5,.25):
            for radius in (8.,12.,16.):
                for excess in (.02,.08):
                    row=run(radius,excess,mobility,h);rows.append(row)
                    print(mobility,h,radius,excess,'M_eff',row['effective_mobility'],flush=True)
    single=[r for r in rows if r['mobility']=='single']
    fine={(r['radius'],r['excess']):r for r in single if r['h']==.25};coarse={(r['radius'],r['excess']):r for r in single if r['h']==.5}
    gates=dict(mobility_within_10_percent=all(abs(r['effective_mobility']/r['predicted_mobility']-1)<=.1 for r in fine.values()),
        mesh_change_within_5_percent=all(abs(fine[k]['effective_mobility']/coarse[k]['effective_mobility']-1)<=.05 for k in fine),
        inventory=all(r['inventory_drift']<1e-8 for r in rows),energy=all(r['max_energy_increase']<=1e-11 for r in rows))
    report=dict(scope='Section 52 sharp-interface pore growth; unchanged energy, two exchange mobilities; synthetic nondimensional parameters.',
        rows=rows,gates=gates,study_seconds=perf_counter()-start,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),threads=1))
    (out/'report.json').write_text(json.dumps(report,indent=1)+'\n');print('gates',gates,flush=True)


if __name__=='__main__':main()
