"""One-encounter spatial graph reactions with live phase/population feedback."""

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
from bmvr.continuum3d import Cube,CoupledCrystal,Parameters3D
from bmvr.graphs3d import GraphMechanics,encounter_graphs,graph_state,graph_step,frozen_reaction_family,ensemble_trajectory,save_graph_checkpoint


def setup(interface_scale=1.,mobility=1.):
    grid=Cube(12,17)
    parameters=Parameters3D(W=interface_scale,gradient=(interface_scale,interface_scale,1.4*interface_scale),
                            phase_misfit=.03,binding=.1,yield_stress=10.,phase_mobility=mobility)
    # Local crystallographic motion belongs to the graph; unrotated bulk FCC slip is disabled in this encounter.
    crystal=CoupledCrystal(grid,parameters,mean_strain=np.array([[0,0,-.08],[0,0,0],[-.08,0,0]]))
    mechanics=GraphMechanics(grid,quadrature_order=8)
    x,_,_=grid.coordinates()
    phase=np.tanh(np.sin(2*np.pi*x/12)*12/(2*np.pi*np.sqrt(2)))
    return crystal,mechanics,(phase,np.full(grid.shape,.2),np.zeros((12,)+grid.shape))


def write(folder,report,arrays):
    folder.mkdir(parents=True,exist_ok=True)
    report['checks']={key:bool(value) for key,value in report['checks'].items()}
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_before_io_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    report['environment']={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                            'platform':platform.platform(),'threads':1}
    t=perf_counter();np.savez_compressed(folder/'fields.npz',**arrays);report['timings']['array_io_s']=perf_counter()-t
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(folder.name,report['checks'],flush=True)
    if not all(report['checks'].values()):raise RuntimeError('A declared check failed; see the saved report.')


def evolve(crystal,mechanics,fields,graph,end=.15,dt=.01):
    state=graph_state(crystal,fields,graph,mechanics)
    start_nodes=graph.nodes.copy();mass=crystal.grid.integrate(fields[1])
    times=[0.];energy=[state['energy']];nodes=[graph.nodes.copy()];inventory=[mass];equilibrium=[state['equilibrium_rms']]
    start=perf_counter();time=0.;trials=0
    while time<end-1e-12:
        fields,graph,state,used,attempts=graph_step(crystal,fields,graph,mechanics,min(dt,end-time),state)
        time+=used;trials+=attempts
        times.append(time);energy.append(state['energy']);nodes.append(graph.nodes.copy())
        inventory.append(crystal.grid.integrate(fields[1]));equilibrium.append(state['equilibrium_rms'])
    residual_samples=state['sampled_phase'].reshape(len(graph.edges),-1)[graph.kinds==1]
    contact_mask=np.linalg.norm(graph.projectors@np.array([1.,0.,0.]),axis=1)<1e-10
    contact_samples=mechanics.sample_phase(fields[0],graph.nodes[contact_mask],graph.core)
    metrics={'energy_initial':energy[0],'energy_final':energy[-1],'max_energy_increase':float(max(0.,max(np.diff(energy)))),
             'inventory_drift':float(max(np.abs(np.array(inventory)-mass))),
             'maximum_equilibrium_rms':float(max(equilibrium)),'junction_balance':graph.balance_error,
             'source_divergence':state['source_divergence'],'source_curl':state['source_curl'],
             'node_displacement':float(np.max(np.linalg.norm(graph.nodes-start_nodes,axis=1))),
             'constraint_error':float(np.max(np.abs(np.einsum('nij,nj->ni',np.eye(3)-graph.projectors,graph.nodes-start_nodes)))),
             'residual_phase_abs_max':float(np.max(np.abs(residual_samples))) if residual_samples.size else None,
             'contact_phase_abs_max':float(np.max(np.abs(contact_samples))),
             'steps':len(times)-1,'trials':trials}
    arrays={'time':np.array(times),'energy':np.array(energy),'nodes':np.array(nodes),'inventory':np.array(inventory),
            'phase':fields[0],'population':fields[1],'slip':fields[2],'stress':state['stress'],
            'edges':graph.edges,'burgers':graph.burgers,'kinds':graph.kinds}
    return fields,graph,state,metrics,arrays,perf_counter()-start


def branch_run(folder,branch='transmitted',interface_scale=1.,mobility=1.,angle=25.):
    wall=perf_counter();crystal,mechanics,fields=setup(interface_scale,mobility)
    incident=encounter_graphs(angle)['incident'];initial_apex=incident.nodes[1,0]
    fields,incident,before,approach_metrics,approach,approach_s=evolve(crystal,mechanics,fields,incident,end=.1)
    family=encounter_graphs(angle,base_graph=incident)
    rng=np.random.default_rng(20260912);selection=None;requested_branch=branch
    if branch=='sampled':
        names=list(family)
        ee=np.array([graph_state(crystal,fields,g,mechanics)['energy'] for g in family.values()])
        kinetics=frozen_reaction_family(ee,thermal_energy=.05)
        index=int(rng.choice(4,p=kinetics['first_event']));branch=names[index]
        selection={'seed':20260912,'candidate_energies':ee.tolist(),'first_event_probabilities':kinetics['first_event'].tolist(),
                   'selected_branch':branch,'thermal_energy':.05,'barriers':[.02]*3,
                   'scope':'Conditional first event at the frozen encounter snapshot; no claim of continuous global reaction kinetics.'}
    graph=family[branch]
    at_event=graph_state(crystal,fields,graph,mechanics)
    event_jump=at_event['energy']-before['energy']
    initial_fields=[f.copy() for f in fields]
    initial_graph=graph.to_dict()
    fields,graph,state,metrics,arrays,solver_s=evolve(crystal,mechanics,fields,graph)
    delta_fields={key:float(np.sqrt(np.mean((field-original)**2))) for key,field,original in zip(('phase','population','slip'),fields,initial_fields)}
    no_graph=crystal.evaluate(*fields)
    metrics['graph_stress_peak']=float(np.max(np.abs(state['stress']-no_graph['stress'])))
    metrics['approach_apex_x_change']=float(incident.nodes[1,0]-initial_apex)
    metrics['field_changes']=delta_fields
    if len(graph.nodes)==4:
        metrics['post_event_apex_x_change']=float(graph.nodes[1,0]-np.array(initial_graph['nodes'])[1,0])
    checks={'charge':max(metrics['junction_balance'],metrics['source_divergence'],metrics['source_curl'])<1e-10,
            'continuous_energy':metrics['max_energy_increase']<=1e-10*max(1,abs(metrics['energy_initial'])),
            'inventory':metrics['inventory_drift']<1e-8,'equilibrium':metrics['maximum_equilibrium_rms']<1e-10,
            'planes':metrics['constraint_error']<1e-10,'spatial_motion':metrics['node_displacement']>1e-6,
            'live_interface':delta_fields['phase']>1e-7,'approach':metrics['approach_apex_x_change']>0,
            'local_plane_validity':metrics['contact_phase_abs_max']<.2 and (metrics['residual_phase_abs_max'] is None or metrics['residual_phase_abs_max']<.2)}
    elapsed=perf_counter()-wall
    report={'branch':branch,'requested_branch':requested_branch,'stochastic_selection':selection,'parameters':asdict(crystal.p),'angle_degrees':angle,'grid':{'n':17,'length':12.,'h':12/17},
             'interface_coefficients':{'mobile_barrier':.2,'residual_binding':.4},'mean_strain':crystal.mean_strain.tolist(),
             'initial_graph':initial_graph,'final_graph':graph.to_dict(),'metrics':metrics,'approach_metrics':approach_metrics,
             'event_configuration_energy_jump':event_jump,'event_heat_to_bath':-event_jump,
             'checks':checks,'timings':{'approach_solver_and_diagnostics_s':approach_s,'branch_solver_and_diagnostics_s':solver_s,
                                       'setup_event_and_diagnostics_s':max(0.,elapsed-approach_s-solver_s),
                                       'experiment_s':elapsed,'compilation_s':0.,'adaptation_s':0.},
             'limits':['Finite-embryo one-encounter closure, not calibrated collision or general reaction-network kinetics.',
                       'Thermal configuration-energy jumps are separate from dissipative continuous evolution.',
                       'A local reference interface plane constrains contact/residual nodes; diffuse phase evolves.',
                       'Bulk crystal slip disabled; outgoing/incident crystallographic directions belong to the graph.']}
    arrays.update({'approach_time':approach['time'],'approach_energy':approach['energy'],'approach_nodes':approach['nodes'],
                   'event_phase':initial_fields[0],'event_population':initial_fields[1]})
    write(folder,report,arrays)
    start=perf_counter()
    save_graph_checkpoint(folder/'checkpoint.npz',crystal,fields,graph,mechanics,.25,.01,approach_metrics['steps']+metrics['steps'],rng,
                          reaction_context={'event_done':True,'branch':branch,'selection':selection})
    report['timings']['checkpoint_io_s']=perf_counter()-start
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))


def kinetic_study(folder):
    wall=perf_counter();crystal,mechanics,fields=setup()
    names=list(encounter_graphs())
    energies=np.array([graph_state(crystal,fields,g,mechanics)['energy'] for g in encounter_graphs().values()])
    theta=.05;family=frozen_reaction_family(energies,thermal_energy=theta,gate=.8)
    times=np.linspace(0,30,101);probability,free=ensemble_trajectory(energies,family,times,theta)
    rng=np.random.default_rng(20260912);samples=rng.choice(4,20000,p=family['first_event']);empirical=np.bincount(samples,minlength=4)/len(samples)
    error=np.abs(empirical-family['first_event']);se=np.sqrt(family['first_event']*(1-family['first_event'])/len(samples))
    flux=family['stationary'][:,None]*family['rates']
    angles=np.arange(0.,76.,15.);sweep=[];residual=[]
    for angle in angles:
        graphs=encounter_graphs(angle)
        ee=np.array([graph_state(crystal,fields,g,mechanics)['energy'] for g in graphs.values()])
        sweep.append(ee)
        residual.append(np.linalg.norm(graphs['incident'].burgers[0]-graphs['transmitted'].burgers[0]))
    metrics={'detailed_balance_error':float(np.max(np.abs(flux-flux.T))),
              'probability_sum_error':float(np.max(np.abs(probability.sum(axis=1)-1))),
              'max_ensemble_free_energy_increase':float(max(0.,max(np.diff(free))))}
    checks={'detailed_balance':metrics['detailed_balance_error']<1e-12,
            'probability':metrics['probability_sum_error']<1e-12 and probability.min()>=-1e-12,
            'ensemble_energy':metrics['max_ensemble_free_energy_increase']<1e-10,
            'sampling':bool(np.all(error<=5*se+1/len(samples))),
            'residual_geometry':bool(np.max(np.abs(np.array(residual)-.4*np.sin(np.deg2rad(angles)/2)))<1e-12)}
    report={'state_names':names,'energies':energies.tolist(),'thermal_energy':theta,'gate':.8,
             'barriers':[.02]*3,'seed':20260912,'first_event_probabilities':family['first_event'].tolist(),
             'sampled_probabilities':empirical.tolist(),'metrics':metrics,'checks':checks,'timings':{'study_s':perf_counter()-wall},
             'limits':['Detailed balance is verified for the frozen spatial candidate family only.',
                       'Finite embryo sizes and synthetic barriers prevent material outcome predictions.']}
    write(folder,report,{'time':times,'probability':probability,'ensemble_free_energy':free,'samples':samples,
                         'angles':angles,'state_energies':np.array(sweep),'residual_burgers_norm':np.array(residual)})


def collect(folder):
    baseline=folder/'transmitted'
    comparisons=[]
    with np.load(baseline/'fields.npz') as base:
        for name in ('energy-low','energy-high','mobility-low','mobility-high'):
            path=folder/name
            if not (path/'fields.npz').exists():continue
            with np.load(path/'fields.npz') as data:
                differences={key:float(np.sqrt(np.mean((data[key]-base[key])**2))) for key in ('phase','population','nodes','stress')}
            r=json.loads((path/'report.json').read_text())
            comparisons.append({'run':name,'W':r['parameters']['W'],'gradient':r['parameters']['gradient'],
                                'phase_mobility':r['parameters']['phase_mobility'],'rms_differences':differences})
    (folder/'sensitivity.json').write_text(json.dumps(comparisons,indent=2))
    feedback=[]
    with np.load(folder/'incident/fields.npz') as base:
        for branch in ('transmitted','absorbed','reflected','sampled'):
            path=folder/branch
            if not (path/'fields.npz').exists():continue
            with np.load(path/'fields.npz') as data:
                differences={key:float(np.sqrt(np.mean((data[key]-base[key])**2))) for key in ('phase','population','stress')}
            feedback.append({'branch':branch,'rms_difference_from_incident_control':differences})
    (folder/'branch-feedback.json').write_text(json.dumps(feedback,indent=2))
    print('sensitivity runs',len(comparisons),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/interface-graphs-2026-09-12')
    parser.add_argument('--worker',choices=('incident','transmitted','absorbed','reflected','sampled','kinetics'))
    parser.add_argument('--interface-scale',type=float,default=1.)
    parser.add_argument('--mobility',type=float,default=1.)
    parser.add_argument('--collect',action='store_true')
    args=parser.parse_args()
    if args.worker:
        if args.worker=='kinetics':kinetic_study(args.output)
        else:branch_run(args.output,args.worker,args.interface_scale,args.mobility)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.collect:
        jobs=[(name,name,1.,1.) for name in ('incident','transmitted','absorbed','reflected','sampled','kinetics')]
        jobs += [('energy-low','transmitted',.5,1.),('energy-high','transmitted',2.,1.),
                 ('mobility-low','transmitted',1.,.5),('mobility-high','transmitted',1.,2.)]
        for name,worker,scale,mobility in jobs:
            subprocess.run([sys.executable,__file__,'--worker',worker,'--interface-scale',str(scale),'--mobility',str(mobility),
                             '--output',str(args.output/name)],check=True,cwd=ROOT)
    collect(args.output)


if __name__=='__main__':main()
