"""Section 54: junction formation, zipping/unzipping and dissolution with full elastic mechanics."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bmvr.continuum3d import Cube
from bmvr.graphs3d import GraphMechanics,junction_pair,dissolve_junction,junction_length,line_state,line_step,DislocationGraph

L_MIN=.1
PREDICTIONS=dict(favorable='lengthens',unfavorable='dissolves',neutral=None)


def trajectory(case,mechanics,end,dt=.01):
    pair=junction_pair(case);crossed,graph=pair['crossed'],pair['junction']
    before=line_state(crossed,mechanics)['energy'];state=line_state(graph,mechanics)
    events=[dict(time=0.,kind='zip',energy_jump=state['energy']-before,heat_to_bath=before-state['energy'])]
    history=[[0.,junction_length(graph),state['energy']]];continuous=0.;worst=0.;time=0.;start=perf_counter()
    while time<end-1e-12:
        if len(graph.edges)==9 and junction_length(graph)<L_MIN:
            new=dissolve_junction(graph,crossed);new_state=line_state(new,mechanics)
            events.append(dict(time=time,kind='dissolve',energy_jump=new_state['energy']-state['energy'],heat_to_bath=state['energy']-new_state['energy']))
            graph,state=new,new_state;continue
        old=state['energy'];graph,state,used=line_step(graph,min(dt,end-time),mechanics,state)
        time+=used;continuous+=state['energy']-old;worst=max(worst,state['energy']-old)
        history.append([time,junction_length(graph),state['energy']])
    history=np.array(history);ledger=state['energy']-before-sum(e['energy_jump'] for e in events)-continuous
    t=pair['t'];line_nodes=graph.nodes[6:] if len(graph.edges)==9 else np.zeros((0,3))
    off=float(np.max(abs(line_nodes-np.outer(line_nodes@t,t)))) if len(line_nodes) else 0.
    checks=dict(charge=graph.balance_error<1e-10,continuous_energy=worst<=1e-10*max(1,abs(before)),ledger=abs(ledger)<1e-10,line_constraint=off<1e-12)
    return dict(case=case,b1_dot_b2=float(pair['b1']@pair['b2']),events=events,initial_length=float(history[0,1]),final_length=float(history[-1,1]),
        max_length=float(history[:,1].max()),min_length=float(history[:,1].min()),dissolved=any(e['kind']=='dissolve' for e in events),
        ledger_residual=float(ledger),checks=checks,solver_seconds=perf_counter()-start,history=history.tolist(),final_graph=graph.to_dict())


def outcome(row):
    if row['case']=='favorable':return row['final_length']>row['initial_length']
    if row['case']=='unfavorable':return row['dissolved']
    return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);start=perf_counter();mechanics=GraphMechanics(Cube(12,17))
    declared=[trajectory(case,mechanics,2.) for case in ('favorable','unfavorable','neutral')]
    for row in declared:row.update(prediction=PREDICTIONS[row['case']],prediction_holds=outcome(row))
    # Post hoc (after the declared window): extended windows to observe completion of unzipping or zipping.
    extended=[dict(trajectory(case,mechanics,20.,dt=.05),post_hoc=True) for case in ('favorable','unfavorable','neutral')]
    # Restart: rebuild the favorable junction from its dictionary and continue one step; compare bitwise.
    pair=junction_pair('favorable');graph=pair['junction']
    for _ in range(5):graph,_,_=line_step(graph,.01,mechanics)
    copy=DislocationGraph(**{k:(np.array(v) if isinstance(v,list) else v) for k,v in json.loads(json.dumps(graph.to_dict())).items()})
    a,sa,_=line_step(graph,.01,mechanics);b,sb,_=line_step(copy,.01,mechanics)
    restart=bool(np.array_equal(a.nodes,b.nodes) and sa['energy']==sb['energy'])
    report=dict(scope='Section 54 junction formation/dissolution; synthetic nondimensional FCC-like geometry, no calibration.',
        l_min=L_MIN,declared=declared,post_hoc_extended=extended,restart_bitwise=restart,
        all_checks=all(all(r['checks'].values()) for r in declared+extended) and restart,
        line_tension_equilibrium_length_favorable=float(2*(1-np.sqrt(.140625/.75))),
        study_seconds=perf_counter()-start,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),threads=1))
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item())+'\n')
    for r in declared+extended:
        print(r['case'],'post hoc' if r.get('post_hoc') else 'declared','jump',round(r['events'][0]['energy_jump'],6),'L',round(r['initial_length'],4),'->',round(r['final_length'],4),
              'dissolved',r['dissolved'],'prediction',r.get('prediction'),r.get('prediction_holds'),'checks',all(r['checks'].values()),flush=True)
    print('restart',restart,flush=True)


if __name__=='__main__':main()
