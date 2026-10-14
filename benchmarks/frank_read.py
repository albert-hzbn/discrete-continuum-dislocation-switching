"""Section 60: Frank-Read multiplication in the line-tension approximation, with declared gates."""
import argparse,json,platform
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.frank_read import FrankRead,signed_area

ROOT=Path(__file__).resolve().parents[1]
DT=2e-4


def run(stress,t_end,stop_after=None,snapshots=(),require_descent=False):
    fr=FrankRead(stress=stress);s=fr.initial();E=fr.energy(s);Q=0.;t=0.;k=0;events=[];history=[];increase=0.;shots={}
    A,B=s['line'][0].copy(),s['line'][-1].copy();next_record=0.;start=perf_counter()
    while t<t_end-1e-12:
        s,E1,used=fr.step(s,min(DT,t_end-t),E);increase=max(increase,E1-E);E=E1;t+=used;k+=1
        s,removed=fr.remesh(s);Q-=removed;E=fr.energy(s)
        if k%10==0:
            res=fr.reconnect(s,require_descent=require_descent)
            if res is not None:
                new,info=res;En=fr.energy(new);dF=En-E;before=E+Q;Q-=dF
                events.append(dict(time=t,delta_F=dF,ledger_error=abs(En+Q-before)/max(abs(before),1.),loops=len(new['loops']),
                    pinned=bool(np.array_equal(new['line'][0],A) and np.array_equal(new['line'][-1],B)),loop_nodes=len(new['loops'][-1]),
                    loop_area=signed_area(new['loops'][-1]),**info))
                s,E=new,En
        if t>=next_record-1e-12:
            history.append(dict(time=t,energy=E,loops=len(s['loops']),line_nodes=len(s['line']),line_area=signed_area(s['line']),
                loop_radii=[float(np.sqrt(max(signed_area(l),0)/np.pi)) for l in s['loops']]));next_record+=.5
        for ts in snapshots:
            if ts not in shots and t>=ts:shots[ts]=dict(line=s['line'].tolist(),loops=[l.tolist() for l in s['loops']])
        if stop_after and len(events)>=stop_after and t>events[-1]['time']+1:break
    return dict(stress=stress,critical_stress=fr.critical_stress,end_time=t,events=events,max_energy_increase=increase,history=history,
        final=dict(line=s['line'].tolist(),loops=[l.tolist() for l in s['loops']]),snapshots={f'{k:g}':v for k,v in shots.items()},seconds=perf_counter()-start)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--post-hoc-descent',action='store_true');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    import sys
    if '--post-hoc-descent' in sys.argv:
        # Post hoc (Section 60 outcome): reconnection deferred while it would raise F.
        ph=dict(supercritical=run(1.2,30.,stop_after=2,snapshots=(1.,3.,5.,8.,12.),require_descent=True),above_critical=run(1.05,30.,stop_after=1,require_descent=True))
        (out/'post-hoc-descent.json').write_text(json.dumps(ph,indent=1,default=lambda o:o.item())+'\n')
        print('post hoc',[(round(e['time'],2),round(e['delta_F'],4)) for e in ph['supercritical']['events']],[(round(e['time'],2),round(e['delta_F'],4)) for e in ph['above_critical']['events']],flush=True);return
    sub=run(.8,20.);line=np.array(sub['final']['line']);R=1/.8;center=np.array([0.,-np.sqrt(R*R-1)])
    arc_error=float(np.max(np.abs(np.linalg.norm(line-center,axis=1)-R))/R)
    sup=run(1.2,30.,stop_after=2,snapshots=(1.,3.,5.,8.,12.))
    below=run(.95,30.);above=run(1.05,30.,stop_after=1)
    tol=lambda r:r['max_energy_increase']<=1e-12*max(1.,abs(r['history'][0]['energy']))
    gates=dict(F1_subcritical_arc=arc_error<=.005 and tol(sub) and not sub['events'],
        F2_multiplication=len(sup['events'])>=2 and all(e['pinned'] for e in sup['events']),
        F3_events=bool(sup['events']) and all(e['ledger_error']<=1e-12 and e['delta_F']<=0 for r in (sup,above) for e in r['events']) and all(tol(r) for r in (sup,below,above)),
        F4_critical=not below['events'] and len(above['events'])>=1)
    report=dict(scope='Section 60 Frank-Read source, line-tension approximation; declared gates.',dt=DT,subcritical=dict(sub,arc_relative_error=arc_error),
        supercritical=sup,below_critical=below,above_critical=above,gates=gates,environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()))
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item())+'\n')
    print(gates,'arc',arc_error,'events',[(round(e['time'],2),round(e['delta_F'],4)) for e in sup['events']],'above',[(round(e['time'],2)) for e in above['events']],flush=True)


if __name__=='__main__':main()
