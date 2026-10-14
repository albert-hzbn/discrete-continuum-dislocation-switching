"""Curved-line source/force convergence and coupled 3D loop evolution."""

import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bmvr import _resource as resource
from bmvr.continuum3d import Cube,CoupledCrystal,fcc_slip_systems
from bmvr.loops3d import Loop,LoopMechanics,circular_loop,coupled_state,coupled_step,save_checkpoint


def dump(path,report,arrays):
    path.mkdir(parents=True,exist_ok=True)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_before_io_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    report['environment']={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                            'platform':platform.platform(),'threads':1}
    started=perf_counter()
    np.savez_compressed(path/'fields.npz',**arrays)
    report['timings']['array_io_s']=perf_counter()-started
    (path/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(path.name,report.get('checks',{}),flush=True)
    if not all(report.get('checks',{}).values()):
        raise RuntimeError('A declared check failed; the report was saved.')


def convergence(folder):
    start=perf_counter()
    loop=circular_loop(radius=3,count=16,core=.8,core_tension=.01)
    rows=[]
    for n in (13,17,25,33,49):
        model=LoopMechanics(Cube(16,n),quadrature_order=10)
        result=model.evaluate([loop])
        rows.append({'n':n,'h':16/n,'energy':result['energy'],'forces':result['forces'][0].tolist(),
                     'source_divergence':result['divergence_relative'],'curl_error':result['curl_relative'],
                     'equilibrium_rms':result['equilibrium_rms'],'timings':result['timings']})
    reference=rows[-1]
    for row in rows:
        row['relative_energy_error']=abs(row['energy']-reference['energy'])/abs(reference['energy'])
        row['relative_force_error']=float(np.linalg.norm(np.array(row['forces'])-reference['forces'])/np.linalg.norm(reference['forces']))
    polygon=[]
    model=LoopMechanics(Cube(16,25),quadrature_order=8)
    for count in (8,12,16,24,32,64,96):
        item=circular_loop(radius=3,count=count,core=.8,core_tension=.01)
        result=model.evaluate([item],need_forces=False)
        polygon.append({'nodes':count,'energy':result['energy'],'area':float(np.linalg.norm(item.area)),
                        'length':float(item.lengths.sum())})
    for row in polygon:
        row['relative_energy_error']=abs(row['energy']-polygon[-1]['energy'])/polygon[-1]['energy']
        row['relative_area_error']=abs(row['area']-np.pi*9)/(np.pi*9)
    quadrature=[]
    triangle=circular_loop(radius=3,count=3,core=.8,burgers=(.2,.05,.07))
    for order in (2,4,6,8,12,16):
        result=LoopMechanics(Cube(16,25),quadrature_order=order).evaluate([triangle])
        quadrature.append({'order':order,'forces':result['forces'][0].tolist()})
    for row in quadrature:
        row['relative_force_error']=float(np.linalg.norm(np.array(row['forces'])-quadrature[-1]['forces'])/np.linalg.norm(quadrature[-1]['forces']))
    report={'field_grid':rows,'polygon':polygon,'quadrature':quadrature,
             'parameters':loop.to_dict(),'timings':{'study_s':perf_counter()-start},
             'checks':{'charge':max(r['source_divergence'] for r in rows)<1e-10,
                       'curl':max(r['curl_error'] for r in rows)<1e-10,
                       'equilibrium':max(r['equilibrium_rms'] for r in rows)<1e-10,
                       'grid_refinement':all(a['relative_force_error']>b['relative_force_error'] for a,b in zip(rows[:-2],rows[1:-1])),
                       'polygon_refinement':all(a['relative_energy_error']>b['relative_energy_error'] for a,b in zip(polygon[:-2],polygon[1:-1])),
                       'force_quadrature':quadrature[-2]['relative_force_error']<1e-5},
             'notes':['Fixed Gaussian core a=0.8. Grid, polygon, and force quadrature are refined separately.',
                      'Field reference n=49, polygon reference 96 nodes, quadrature reference order 16.',
                      'Direct source/force evaluation is a verification method, not scalable DDD.']}
    dump(folder,report,{'loop_nodes':loop.nodes})


def coupled(folder,pinned=False,n=25,end=.4,dt=.02):
    wall=perf_counter()
    grid=Cube(16,n)
    x,y,z=grid.coordinates()
    slip_direction=np.array([1.,-1.,0.])/np.sqrt(2)
    plane_normal=np.ones(3)/np.sqrt(3)
    rotation=np.array([slip_direction,np.cross(plane_normal,slip_direction),plane_normal])
    crystal=CoupledCrystal(grid,schmid=fcc_slip_systems(rotation)[2],
                           mean_strain=np.array([[0,.04,-.08],[.04,0,0],[-.08,0,0]]))
    mechanics=LoopMechanics(grid,quadrature_order=8)
    fields=(np.tanh((4-np.sqrt(x*x+y*y+z*z))/np.sqrt(2)),
            .2+.02*np.exp(-((x-3)**2+(y-1)**2+z*z)/8),np.zeros((12,)+grid.shape))
    loops=[circular_loop(radius=3,count=12,center=(.5,0,0),core=.8,glide_normal=np.array([0.,0.,1.]))]
    current=coupled_state(crystal,fields,loops,mechanics)
    initial=current['energy']
    inventory=grid.integrate(fields[1])
    initialization=perf_counter()-wall
    t,steps,trials,solver,diagnostics=0.,0,0,0.,0.
    times,energy,mass,nodes,eq,div,components,minimum_dt=[],[],[],[],[],[],[],dt
    component_names=list(current['components'])

    def record():
        times.append(t); energy.append(current['energy']); mass.append(grid.integrate(fields[1]))
        nodes.append(loops[0].nodes.copy()); eq.append(current['equilibrium_rms'])
        div.append(current['source_divergence_relative']); components.append(list(current['components'].values()))

    record()
    while t<end-1e-12:
        start=perf_counter()
        fields,loops,current,used,attempts=coupled_step(crystal,fields,loops,mechanics,min(dt,end-t),current,pinned=pinned)
        solver+=perf_counter()-start
        t+=used; steps+=1; trials+=attempts; minimum_dt=min(minimum_dt,used)
        start=perf_counter(); record(); diagnostics+=perf_counter()-start
    start=perf_counter()
    displacement=float(np.max(np.linalg.norm(nodes[-1]-nodes[0],axis=1)))
    drift=float(np.max(np.abs(np.array(mass)-inventory)))
    metrics={'steps':steps,'trial_evaluations':trials,'minimum_dt':minimum_dt,'initial_energy':initial,'final_energy':energy[-1],
             'max_energy_increase':float(max(0.,max(np.diff(energy)))),'inventory_drift':drift,
             'maximum_node_displacement':displacement,'maximum_equilibrium_rms':max(eq),'source_divergence_max':max(div),
             'slip_rms':float(np.sqrt(np.mean(fields[2]**2))),
             'primary_field_dofs':sum(field.size for field in fields),'line_coordinate_dofs':loops[0].nodes.size}
    checks={'energy':metrics['max_energy_increase']<=1e-10*max(1,abs(initial)),
            'inventory':drift<=1e-10*max(1,inventory),'mechanical_equilibrium':max(eq)<1e-10,
            'source_charge':max(div)<1e-10,'glide_plane':bool(np.max(np.abs(nodes[-1][:,2]))<1e-12),
            'motion':displacement==0 if pinned else displacement>1e-5,
            'slip_active':metrics['slip_rms']>1e-5}
    diagnostics+=perf_counter()-start
    total=perf_counter()-wall
    report={'model':'3D phase/population/FCC slip with a moving Gaussian-core dislocation loop',
            'grid':{'n':n,'length':16.,'h':grid.h},'end':end,'dt':dt,'pinned':pinned,
            'parameters':asdict(crystal.p),'mean_strain':crystal.mean_strain.tolist(),
            'crystal_rotation':rotation.tolist(),'schmid':crystal.schmid.tolist(),'final_loops':[loop.to_dict() for loop in loops],
            'metrics':metrics,'checks':checks,
            'timings':{'initialization_s':initialization,'solver_s':solver,'diagnostics_s':diagnostics,
                       'unclassified_s':max(0.,total-initialization-solver-diagnostics),'experiment_s':total,
                       'compilation_s':0.,'adaptation_s':0.},
            'limits':['Uniform 3D mechanics and field grid; no adaptive speedup claim.',
                      'The explicit loop and continuum slips represent distinct populations.',
                      'Loop motion and field feedback are active; interface reactions and network junctions are not yet included.']}
    dump(folder,report,{'phase':fields[0],'population':fields[1],'slip':fields[2], 'stress':current['stress'],
                        'strain':current['strain'],'time':np.array(times),'energy':np.array(energy),
                        'inventory':np.array(mass),'nodes':np.array(nodes),'components':np.array(components),
                        'component_names':np.array(component_names)})
    start=perf_counter()
    save_checkpoint(folder/'checkpoint.npz',crystal,fields,loops,mechanics,t,dt,steps,pinned=pinned)
    report['timings']['checkpoint_io_s']=perf_counter()-start
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))


def collect(folder):
    moving=json.loads((folder/'coupled-moving/report.json').read_text())
    pinned=json.loads((folder/'coupled-pinned/report.json').read_text())
    with np.load(folder/'coupled-moving/fields.npz') as a,np.load(folder/'coupled-pinned/fields.npz') as b:
        differences={key:float(np.sqrt(np.mean((a[key]-b[key])**2))) for key in ('phase','population','slip','stress')}
    summary={'moving':moving['metrics'],'pinned':pinned['metrics'],'ablation_field_rms_differences':differences,
             'checks':{'all_feedback_blocks_change':all(value>1e-10 for value in differences.values())},
             'notes':'Pinned-coordinate ablation holds fields, parameters, and initial state fixed; not a performance comparison at equal constraints.'}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('feedback differences',differences,flush=True)
    if (folder/'time-moving/fields.npz').exists() and (folder/'time-pinned/fields.npz').exists():
        time_result={}
        with np.load(folder/'coupled-moving/fields.npz') as a,np.load(folder/'coupled-pinned/fields.npz') as b,\
                np.load(folder/'time-moving/fields.npz') as c,np.load(folder/'time-pinned/fields.npz') as d:
            for key in ('phase','population','slip','stress'):
                coarse,fine=a[key]-b[key],c[key]-d[key]
                relative=float(np.linalg.norm(coarse-fine)/np.linalg.norm(fine))
                time_result[key]={'coarse_rms':float(np.sqrt(np.mean(coarse**2))),
                                  'fine_rms':float(np.sqrt(np.mean(fine**2))), 'relative_change':relative,
                                  'within_predeclared_10_percent':relative<=.1}
            time_result['node_position_linf_change']=float(np.max(np.abs(a['nodes'][-1]-c['nodes'][-1])))
        (folder/'ablation-time-convergence.json').write_text(json.dumps(time_result,indent=2,allow_nan=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/curved-loops-fcc-2026-09-12')
    parser.add_argument('--worker',choices=('convergence','coupled-moving','coupled-pinned'))
    parser.add_argument('--dt',type=float,default=.02)
    parser.add_argument('--end',type=float,default=.4)
    parser.add_argument('--collect',action='store_true')
    args=parser.parse_args()
    if args.dt<=0 or args.end<=0:
        parser.error('dt and end must be positive')
    if args.worker:
        if args.worker=='convergence': convergence(args.output)
        else: coupled(args.output,pinned=args.worker=='coupled-pinned',dt=args.dt,end=args.end)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.collect:
        for part in ('convergence','coupled-moving','coupled-pinned'):
            subprocess.run([sys.executable,__file__,'--worker',part,'--dt',str(args.dt),'--end',str(args.end),'--output',str(args.output/part)],check=True,cwd=ROOT)
    collect(args.output)


if __name__=='__main__': main()
