"""Section 49: sustained-reflection criterion, cross-system reflection and repeated interface encounters."""

import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'benchmarks'))
from bmvr.graphs3d import (encounter_graphs,cross_reflection_graph,graph_state,graph_step,frozen_reaction_family,
                           save_graph_checkpoint,load_graph_checkpoint)
from interface_graphs import setup,evolve

DECLARED=(0.,25.,60.)
POST_HOC=(75.,)
SECOND_SYSTEM=60.
SEED=20260926
DT=.01


def approach():
    crystal,mechanics,fields=setup()
    incident=encounter_graphs()['incident']
    fields,incident,state,metrics,_,_=evolve(crystal,mechanics,fields,incident,end=.1)
    return crystal,mechanics,fields,incident,state


def branch_window(angle,end=.6):
    crystal,mechanics,fields,incident,before=approach()
    graph=cross_reflection_graph(incident,angle)
    jump=graph_state(crystal,fields,graph,mechanics)['energy']-before['energy']
    fields,graph,state,metrics,arrays,seconds=evolve(crystal,mechanics,fields,graph,end=end,dt=DT)
    distance=-arrays['nodes'][:,1,0]
    checks={'charge':max(metrics['junction_balance'],metrics['source_divergence'],metrics['source_curl'])<1e-10,
            'continuous_energy':metrics['max_energy_increase']<=1e-10*max(1,abs(metrics['energy_initial'])),
            'inventory':metrics['inventory_drift']<1e-8,'equilibrium':metrics['maximum_equilibrium_rms']<1e-10,
            'planes':metrics['constraint_error']<1e-10}
    sustained=bool(np.all(np.diff(distance)>0) and distance[-1]>distance[0])
    return dict(angle=angle,event_energy_jump=jump,event_heat_to_bath=-jump,apex_distance_start=float(distance[0]),
        apex_distance_end=float(distance[-1]),apex_distance_min=float(distance.min()),apex_distance_max=float(distance.max()),
        sustained=sustained,metrics=metrics,checks=checks,solver_seconds=seconds,
        history=dict(time=arrays['time'].tolist(),distance=distance.tolist()))


def incident_type(graph,bin):
    return len(graph.edges)==4 and not graph.kinds.any() and np.allclose(graph.burgers,bin)


def family_for(graph):
    family=encounter_graphs(base_graph=graph)
    family['cross_reflected']=cross_reflection_graph(graph,SECOND_SYSTEM)
    return family


def sequence(crystal,mechanics,fields,graph,rng,context,end,stop_at=None,checkpoint=None):
    """Evolve with repeated conditional first events; context carries time, events and ledger."""
    state=graph_state(crystal,fields,graph,mechanics);bin=np.array(context['bin'])
    while context['time']<end-1e-12 and len(context['events'])<5:
        if stop_at is not None and context['time']>=stop_at-1e-12:break
        apex=graph.nodes[1,0] if incident_type(graph,bin) else None
        if apex is not None and apex>=context['contact_x']-1e-12 and context['time']-context['last_event']>=.1-1e-12:
            family=family_for(graph);names=list(family)
            energies=np.array([graph_state(crystal,fields,g,mechanics)['energy'] for g in family.values()])
            kinetics=frozen_reaction_family(energies,thermal_energy=.05)
            forced=context.get('forced')
            index=names.index(forced) if forced else int(rng.choice(len(names),p=kinetics['first_event']));name=names[index]
            jump=float(energies[index]-energies[0]);graph=family[name];state=graph_state(crystal,fields,graph,mechanics)
            context['events'].append(dict(time=context['time'],branch=name,energy_jump=jump,heat_to_bath=-jump,
                probabilities=dict(zip(names,kinetics['first_event'].tolist())),charge=graph.balance_error))
            context['event_jump_sum']+=jump;context['last_event']=context['time']
            if name!='reflected':context['terminal']=name
            continue
        if context['terminal'] is not None and context['time']-context['last_event']>=.6-1e-12:break
        old=state['energy']
        fields,graph,state,used,_=graph_step(crystal,fields,graph,mechanics,min(DT,end-context['time']),state)
        context['time']+=used;context['continuous_change']+=state['energy']-old
        context['max_continuous_increase']=max(context['max_continuous_increase'],state['energy']-old)
        context['apex_x'].append([context['time'],float(graph.nodes[1,0]) if len(graph.nodes)==4 else None])
        if checkpoint and len(context['events'])==1 and not context.get('saved') and context['time']-context['last_event']>=.5-1e-12:
            context['saved']=True
            save_graph_checkpoint(checkpoint,crystal,fields,graph,mechanics,context['time'],DT,len(context['apex_x']),rng,
                                  reaction_context=context)
    context['final_energy']=state['energy']
    return fields,graph,state,context


def run_sequence(folder,end=20.,seed=SEED,forced=None,restart_check=True):
    crystal,mechanics,fields,incident,before=approach()
    rng=np.random.default_rng(seed)
    context=dict(time=.1,last_event=-1.,events=[],bin=incident.burgers[0].tolist(),contact_x=float(incident.nodes[1,0]),
                 event_jump_sum=0.,continuous_change=0.,max_continuous_increase=0.,apex_x=[],terminal=None,initial_energy=before['energy'],forced=forced)
    start=perf_counter();tag='' if forced is None else '-'+forced
    fields,graph,state,context=sequence(crystal,mechanics,fields,incident,rng,context,end,
        checkpoint=folder/f'sequence{tag}-checkpoint.npz' if restart_check else None)
    seconds=perf_counter()-start
    # Bitwise restart from the checkpoint saved between events.
    restart=None
    if restart_check and (folder/f'sequence{tag}-checkpoint.npz').exists():
        rc,rf,rg,rm,rr,clock=load_graph_checkpoint(folder/f'sequence{tag}-checkpoint.npz')
        rctx=clock['reaction_context']
        f2,g2,s2,c2=sequence(rc,rm,rf,rg,rr,rctx,end)
        restart=dict(events_equal=c2['events']==context['events'],nodes_equal=bool(np.array_equal(g2.nodes,graph.nodes)),
                     fields_equal=all(bool(np.array_equal(a,b)) for a,b in zip(f2,fields)),energy_equal=s2['energy']==state['energy'])
    ledger=context['final_energy']-context['initial_energy']-context['event_jump_sum']-context['continuous_change']
    checks={'charge':all(e['charge']<1e-10 for e in context['events']) and graph.balance_error<1e-10,
            'continuous_energy':context['max_continuous_increase']<=1e-10*max(1,abs(context['initial_energy'])),
            'ledger':abs(ledger)<1e-9*max(1,abs(context['initial_energy'])),
            'restart':(restart is not None and all(restart.values())) if restart_check else True}
    return dict(seed=seed,forced=forced,second_system_degrees=SECOND_SYSTEM,end=end,events=context['events'],event_count=len(context['events']),
        terminal_branch=context['terminal'],stopped_at=context['time'],ledger_residual=ledger,
        total_heat_to_bath=-context['event_jump_sum'],continuous_energy_change=context['continuous_change'],
        restart=restart,checks=checks,solver_seconds=seconds,apex_x=context['apex_x'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);start=perf_counter()
    branches=[branch_window(a) for a in DECLARED]+[dict(branch_window(a),post_hoc=True) for a in POST_HOC]
    predictions={'0':'returns','25':'returns','60':'departs'}
    for b in branches:
        key=str(int(b['angle']))
        if key in predictions:b['prediction']=predictions[key];b['prediction_holds']=(b['sustained'] if predictions[key]=='departs' else not b['sustained'])
    seq=run_sequence(out)
    # Post hoc (after the declared seeded sequence ended at its first event): driver verification and a seed ensemble.
    forced=run_sequence(out,forced='reflected')
    ensemble=[run_sequence(out,seed=SEED+k,restart_check=False) for k in range(1,33)]
    counts=[e['event_count'] for e in ensemble]
    report_ensemble=dict(seeds=[SEED+k for k in range(1,33)],event_counts=counts,terminal=[e['terminal_branch'] for e in ensemble],
        fraction_multi_event=float(np.mean(np.array(counts)>1)),all_checks=all(all(e['checks'].values()) for e in ensemble),
        max_ledger_residual=max(abs(e['ledger_residual']) for e in ensemble))
    report=dict(post_hoc_forced_recoil=forced,post_hoc_seed_ensemble=report_ensemble,scope='Section 49 synthetic nondimensional closure; no material calibration. Frozen-family conditional first events at each encounter.',
        branches=branches,sequence=seq,all_checks=all(all(b['checks'].values()) for b in branches) and all(seq['checks'].values())
            and all(forced['checks'].values()) and report_ensemble['all_checks'],
        study_seconds=perf_counter()-start,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),threads=1))
    (out/'report.json').write_text(json.dumps(report,indent=1,allow_nan=False,default=lambda o:o.item()))
    for b in branches:print(b['angle'],'sustained',b['sustained'],'start',round(b['apex_distance_start'],5),'end',round(b['apex_distance_end'],5),
                            'prediction',b.get('prediction'),b.get('prediction_holds'),'checks',all(b['checks'].values()),flush=True)
    print('sequence events',[(round(e['time'],3),e['branch']) for e in seq['events']],'checks',seq['checks'],flush=True)
    print('forced recoil events',[(round(e['time'],3),e['branch']) for e in forced['events']],'checks',forced['checks'],flush=True)
    print('ensemble counts',counts,'multi fraction',report_ensemble['fraction_multi_event'],'checks',report_ensemble['all_checks'],flush=True)


if __name__=='__main__':main()
