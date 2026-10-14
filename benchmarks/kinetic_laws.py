"""Section 56: velocity-curvature, anisotropic mobility and velocity-stress laws with declared gates."""
import argparse,json,os,platform
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from bmvr.grid import Grid,circle
from bmvr.energy import DoubleWell
from bmvr.diagnostics import radius,interface_radii
from bmvr.solvers import ac_step,anisotropic_mobility
from bmvr import parallel
from bmvr.continuum3d import Cube
from bmvr.loops3d import LoopMechanics,circular_loop,velocities

MODEL=DoubleWell()


def isotropic_stepper(grid,dt):
    buf=[np.empty(grid.shape),np.empty(grid.shape)];k=[0]
    def step(v):
        k[0]^=1;return parallel.ac_step(grid,v,MODEL,dt,out=buf[k[0]])
    return step


def curvature_run(R0,n,length=128.,dt=.02,sample=.5,stop=4.5):
    grid=Grid.uniform(length,n);v=circle(grid,R0);step=isotropic_stepper(grid,dt);every=round(sample/dt)
    t,times,R,E=0.,[0.],[radius(grid,v)],[MODEL.value(grid,v)];k=0
    while R[-1]>stop:
        v=step(v);k+=1
        if k%every==0:times.append(k*dt);R.append(radius(grid,v));E.append(MODEL.value(grid,v))
    times,R=np.array(times),np.array(R);V=np.gradient(R,times);keep=(times>=1)&(R>=5)
    return dict(R0=R0,n=n,dt=dt,time=times.tolist(),radius=R.tolist(),V=V[keep].tolist(),kappa=(1/R[keep]).tolist(),
        max_energy_increase=float(max(0,np.max(np.diff(E)))))


def fit(V,kappa):
    A=np.stack((-np.asarray(kappa),np.ones(len(kappa))),1);(m,b),*_=np.linalg.lstsq(A,np.asarray(V),rcond=None);return float(m),float(b)


def velocity_curvature():
    runs={str(R0):curvature_run(R0,256) for R0 in (8,12,16,24)}
    V=sum((r['V'] for r in runs.values()),[]);kappa=sum((r['kappa'] for r in runs.values()),[])
    m,b=fit(V,kappa);per={k:fit(r['V'],r['kappa'])[0] for k,r in runs.items()}
    fine=curvature_run(12,512,dt=.005);m_fine=fit(fine['V'],fine['kappa'])[0];m12=per['12']
    gates=dict(G1_slope=abs(m-1)<=.02,G2_intercept=abs(b)<=.02*max(abs(np.array(V))),G3_history=(max(per.values())-min(per.values()))/np.mean(list(per.values()))<=.01,
        G4_resolution=abs(m_fine-m12)<=.01 and abs(m_fine-1)<=abs(m12-1))
    return dict(pooled_slope=m,intercept=b,max_speed=float(max(abs(np.array(V)))),per_radius_slope=per,fine_h025_slope=m_fine,
        energy_monotone=all(r['max_energy_increase']<=1e-9 for r in list(runs.values())+[fine]),runs=runs,fine=fine,gates=gates)


def anisotropic_run(delta,R0=24.,n=256,length=128.,dt=.02,sample=.5,final=12.):
    grid=Grid.uniform(length,n);v=circle(grid,R0);every=round(sample/dt)
    times,R,E=[0.],[radius(grid,v)],[MODEL.value(grid,v)];k=0
    while R[-1]>final:
        v=ac_step(grid,v,MODEL,dt,mobility=anisotropic_mobility(grid,v,delta));k+=1
        if k%every==0:times.append(k*dt);R.append(radius(grid,v));E.append(MODEL.value(grid,v))
    times,R=np.array(times),np.array(R);A=np.pi*R**2;window=(R<=.9*R0)&(R>=final)
    rate=float(np.polyfit(times[window],A[window],1)[0])
    angles=np.linspace(0,2*np.pi,360,endpoint=False);r=interface_radii(grid,v,angles)
    basis=np.stack((np.ones_like(angles),np.cos(4*angles),np.sin(4*angles)),1);(a0,a4,b4),*_=np.linalg.lstsq(basis,r,rcond=None)
    return dict(delta=delta,area_rate=rate,final_radius=float(R[-1]),final_time=float(times[-1]),a0=float(a0),a4=float(a4),b4=float(b4),
        a4_over_a0=float(a4/a0),max_energy_increase=float(max(0,np.max(np.diff(E)))),radii=r.tolist())


def anisotropic_mobility_study():
    runs={f'{d:g}':anisotropic_run(d) for d in (-.3,0.,.3)};c=runs['0']
    shape={k:r['a4_over_a0']-c['a4_over_a0'] for k,r in runs.items() if k!='0'}
    predicted={k:-float(k)/14 for k in shape}
    gates=dict(A1_area_rate=all(abs(runs[k]['area_rate']/c['area_rate']-1)<=.01 for k in shape) and abs(c['area_rate']/(-2*np.pi)-1)<=.02,
        A2_shape=all(abs(shape[k]/predicted[k]-1)<=.2 for k in shape),A3_energy=all(r['max_energy_increase']<=1e-9 for r in runs.values()))
    return dict(runs=runs,shape_amplitude_minus_control=shape,predicted=predicted,control_lattice_amplitude=c['a4_over_a0'],gates=gates)


def loop_velocity(count,taus,grid):
    mech=LoopMechanics(grid);loop=circular_loop(radius=3.,count=count,burgers=(.2,0.,0.),core=.8,mobility=1.,glide_normal=np.array([0.,0.,1.]))
    out=[]
    for tau in taus:
        strain=np.zeros((3,3));strain[0,2]=strain[2,0]=tau/2
        state=mech.evaluate([loop],mean_strain=strain);rates,_=velocities([loop],state['forces'])
        radial=loop.nodes[:,:2]/np.linalg.norm(loop.nodes[:,:2],axis=1)[:,None]
        out.append(float(np.mean(np.sum(rates[0][:,:2]*radial,axis=1))))
    return np.array(out)


def velocity_stress():
    grid=Cube(16.,25);taus=np.array([-.02,-.01,0.,.01,.02]);rows={}
    for N in (12,24,48):
        v=loop_velocity(N,taus,grid);slope,intercept=np.polyfit(taus,v,1);resid=float(np.max(abs(v-(slope*taus+intercept))))
        predicted=.2*np.cos(np.pi/N)
        rows[str(N)]=dict(velocities=v.tolist(),slope=float(slope),intercept=float(intercept),predicted_slope=float(predicted),
            slope_relative_error=float(abs(slope/predicted-1)),residual=resid,deviation_from_Mb=float(abs(slope/.2-1)),equilibrium_stress=float(-intercept/slope))
    d=[rows[k]['deviation_from_Mb'] for k in ('12','24','48')]
    gates=dict(S1_slope=all(r['slope_relative_error']<=1e-8 for r in rows.values()),
        S2_linear=all(r['residual']<=1e-10*max(abs(np.array(r['velocities'])).max(),1e-30) for r in rows.values()),
        S3_order=d[0]>d[1]>d[2] and all(3.5<=d[i]/d[i+1]<=4.5 for i in range(2)))
    # Post hoc: the declaration omitted the orientation sign. For a counterclockwise loop with b=+e_x,
    # (sigma b) x xi with sigma_xz=tau>0 points inward, so v_r = -M b cos(pi/N) tau.
    for r in rows.values():
        r['corrected_slope_relative_error']=float(abs(r['slope']/(-r['predicted_slope'])-1));r['corrected_deviation_from_Mb']=float(abs(-r['slope']/.2-1))
    dc=[rows[k]['corrected_deviation_from_Mb'] for k in ('12','24','48')]
    corrected=dict(S1_slope=all(r['corrected_slope_relative_error']<=1e-8 for r in rows.values()),S3_order=dc[0]>dc[1]>dc[2] and all(3.5<=dc[i]/dc[i+1]<=4.5 for i in range(2)),
        order_ratios=[dc[0]/dc[1],dc[1]/dc[2]])
    return dict(taus=taus.tolist(),rows=rows,gates=gates,sign_corrected_post_hoc=corrected)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--part',choices=('curvature','anisotropic','stress','assemble'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);parallel.set_threads(4)
    parts=dict(curvature=velocity_curvature,anisotropic=anisotropic_mobility_study,stress=velocity_stress)
    for name,fn in parts.items():
        if args.part and name!=args.part:continue
        if args.part is None and (out/f'{name}.json').exists():continue
        start=perf_counter();res=fn();res['seconds']=perf_counter()-start
        (out/f'{name}.json').write_text(json.dumps(res,indent=1,default=lambda o:o.item())+'\n');print(name,res['gates'],flush=True)
    if args.part in (None,'assemble'):
        report=dict(scope='Section 56 declared kinetic-law tests.',**{k:json.loads((out/f'{k}.json').read_text()) for k in parts},
            environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
        (out/'report.json').write_text(json.dumps(report,indent=1)+'\n')


if __name__=='__main__':main()
