"""Section 64: GLL graded notch with element deletion, restarted at t=0.2 and run to t=0.4; declared gates."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform
from pathlib import Path
from time import perf_counter
import numpy as np
from bmvr.nodal_damage import NodalThermalFracture2D
from graded_notch import compare,label

SOURCE=Path('results/lobatto-notch-adaptive-2026-09-26')
MESHES=(64,96,128);DT=.00025;START=.2;END=.4;CHECKPOINTS=(.25,.3,.35)
FIELDS=('c','eta','T','slip','damage')


MODE={'event':'delete','prefix':'deletion-notch'}
TRANSITIONS=[]  # Section 71 transition events, filled by advance() and stamped with times in run()  # Section 65: 'release' splits nodes along the crack line instead


def name(n,t):return f"{MODE['prefix']}-{n}-0.00025-"+('final' if abs(t-END)<1e-12 else label(t))


def advance(model,state,substep):
    """One DT with the event after each accepted (sub)step. substep (Section 65.4, post hoc): on a failed step retry as
    2 halves, then 4 quarters. Returns the new state, the last step diagnostics, the merged event and the pieces used."""
    tries=(1,2,4) if substep else (1,)
    for i,pieces in enumerate(tries):
        s,merged=state,None
        try:
            for _ in range(pieces):
                s,d=model.step(s,DT/pieces)
                s,event=model.delete_broken(s) if MODE['event']=='delete' else model.split_broken_line(s)
                if MODE.get('transition'):
                    s,tr=model.discretize_released(s,MODE.get('band'))
                    if tr:TRANSITIONS.append(dict(cells=len(tr['cells']),energy_moved=tr['energy_moved'],min_moved=tr['min_moved'],ledger_error=tr['ledger_error']))
                if event:merged=event if merged is None else {**event,'nodes':merged.get('nodes',0)+event.get('nodes',0),'cells':list(merged.get('cells',[]))+list(event.get('cells',[])),'ledger_error':max(merged['ledger_error'],event['ledger_error'])}
            return s,d,merged,pieces
        except (ValueError,RuntimeError):
            if i==len(tries)-1:raise


def advance_eroding(model,held,substep,t,erosions,limit=8):
    """Section 68: when the orientation checks refuse a step and every refusing cell has a fully broken edge, erode those
    cells at the last accepted state (energy to heat) and retry; any other failure propagates. held[0] always holds the
    last accepted state, including any erosion already applied to it."""
    for _ in range(limit+1):
        model.mechanics.refused_cells=None
        try:return advance(model,held[0],substep)
        except ValueError:
            if not MODE.get('erode'):raise
            held[0],event=model.erode_refused(held[0])
            if event is None:raise
            erosions.append(dict(time=t,cells=event['cells'],energy_moved=event['energy_moved'],ledger_error=event['ledger_error']))
    raise ValueError('Erosion limit reached within one step.')


def run(n,out,source=None,subdivide=0,smoothing=0.,substep=False):
    # Post hoc (Section 64 outcome): `source` restarts from a declared run's last accepted state with the subdivided
    # orientation certificate of depth `subdivide`; the dynamics and the deletion rule are unchanged.
    path=SOURCE/f'lobatto-notch-{n}-0.00025-t0.2.npz' if source is None else source
    model,state,clock=NodalThermalFracture2D.load(path);start_time=clock['time'];model.mechanics.subdivide=subdivide
    if MODE.get('elastic'):model.elastic_discrete=True  # Section 71.6
    if smoothing:
        from dataclasses import replace as _replace
        model.mechanics.p=_replace(model.mechanics.p,split_smoothing=smoothing)
    if source is None:assert abs(start_time-START)<1e-12
    first_step=round((start_time-START)/DT)
    steps=round((END-START)/DT);rows=[dict(time=start_time,**model.rate(state)['diagnostics'])];events=[];erosions=[];failure=None;solve=0.;min_inc=0.
    for k in range(first_step+1,steps+1):
        t=START+k*DT;start=perf_counter();prev=state.damage
        held=[state]
        try:
            state,d,event,pieces=advance_eroding(model,held,substep,t-DT,erosions)
            if pieces>1:d={**d,'substeps':pieces}
        except (ValueError,RuntimeError) as error:
            state=held[0];model.save(out/f"{MODE['prefix']}-{n}-0.00025-last-accepted.npz",state,t-DT,k-1);failure=dict(error=str(error),step=k,last_accepted_time=t-DT);break
        solve+=perf_counter()-start;min_inc=min(min_inc,float(np.min(state.damage-prev)));rows.append(dict(time=t,**d))
        if event:events.append(dict(time=t,**{a:b for a,b in event.items() if a!='cells'},count=len(event.get('cells',[])) or event.get('nodes',0)))
        for tr in TRANSITIONS:tr.setdefault('time',t)
        for tc in CHECKPOINTS+(END,):
            if abs(t-tc)<1e-12:model.save(out/(name(n,tc)+'.npz'),state,t,k)
        if k%40==0:print(n,f't={t:.4f}',f"deleted={int((~model.mechanics.active).sum()) if getattr(model.mechanics,'active',None) is not None else 0}",f"split={len(getattr(model,'split',{}))}",f"iters={d['midpoint_iterations']}",flush=True)
    first=rows[0]
    scale=max(first['elastic_energy']+first['chemical_energy']+first['gradient_energy']+2*abs(first['partition_energy'])+first['fracture_energy']+first['damage_binding_energy'],1e-12)
    checks=dict(energy_drift=max(abs(r['total_energy']-first['total_energy']) for r in rows)/scale<1e-4,positive_T=min(r['min_T'] for r in rows)>0,
        composition_inventory=max(abs(r['mass']-first['mass']) for r in rows)<1e-10,damage_bounds=min(r['min_damage'] for r in rows)>=0 and max(r['max_damage'] for r in rows)<=1,
        damage_irreversibility=min_inc>=-1e-13,entropy_production=min(r['entropy_production'] for r in rows)>=-1e-14,
        midpoint_converged=max(r['midpoint_residual'] for r in rows[1:])<=1e-12,deletion_ledger=all(e['ledger_error']<=1e-10 for e in events+erosions+TRANSITIONS))
    result=dict(n=n,start_time=start_time,subdivide=subdivide,split_smoothing=smoothing,failure=failure,checks=checks,events=events,erosions=erosions,transitions=TRANSITIONS,discrete_cells=int(getattr(model,'discrete',np.zeros(1,bool)).sum()),eroded_cells=sum(len(e['cells']) for e in erosions),deleted_cells=int(sum(e['count'] for e in events)),energy_to_heat=float(sum(e.get('energy_moved',0.) for e in events)),
        relative_energy_drift=max(abs(r['total_energy']-first['total_energy']) for r in rows)/scale,solver_seconds=solve,
        rows=[dict(time=r['time'],total_energy=r['total_energy'],damage_integral=r['damage_integral'],min_J=r['min_J'],iterations=r.get('midpoint_iterations'),substeps=r.get('substeps',1)) for r in rows])
    (out/f"{MODE['prefix']}-{n}.json").write_text(json.dumps(result,indent=1,default=lambda o:o.item())+'\n')
    print(n,'done',failure,checks,flush=True)


def assemble(out,subdivide=0):
    # Post hoc checkpoints written before the certificate depth was persisted are read with this class-level default.
    from bmvr.finite_elasticity import FiniteElasticity2D
    FiniteElasticity2D.subdivide=subdivide
    r={n:json.loads((out/f"{MODE['prefix']}-{n}.json").read_text()) for n in MESHES}
    reached=lambda n,t:r[n]['failure'] is None or t<=r[n]['failure']['last_accepted_time']+1e-12
    pairs={}
    for a,b in ((64,96),(96,128)):
        pairs[f'{a}_{b}']={('final' if abs(t-END)<1e-12 else label(t)):compare(out/(name(a,t)+'.npz'),out/(name(b,t)+'.npz'),NodalThermalFracture2D)
            for t in CHECKPOINTS+(END,) if reached(a,t) and reached(b,t)}
    final=lambda k:pairs[k].get('final')
    gates=dict(E1_ledger_and_checks=all(all(x['checks'].values()) for x in r.values()),E2_reach=all(x['failure'] is None for x in r.values()),
        E3_t04=bool(final('64_96') and final('96_128') and all(v<.05 for v in final('64_96').values()) and all(v<.05 for v in final('96_128').values())
            and all(final('96_128')[f]<final('64_96')[f] for f in ('slip','damage'))))
    report=dict(scope='Section 64 element deletion on the GLL graded notch, restarted at t=0.2; declared gates.',pairs=pairs,gates=gates,
        runs={n:{k:v for k,v in x.items() if k!='rows'} for n,x in r.items()},environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),threads=1))
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item())+'\n');print(gates,json.dumps(pairs)[:1500],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--run',type=int)
    parser.add_argument('--source',type=Path);parser.add_argument('--subdivide',type=int,default=0);parser.add_argument('--smoothing',type=float,default=0.);parser.add_argument('--release',action='store_true')
    parser.add_argument('--substep',action='store_true',help='Section 65.4 post hoc: retry a failed step as halves, then quarters')
    parser.add_argument('--erode',action='store_true',help='Section 68: node release plus reactive erosion of refusing broken-edge cells')
    parser.add_argument('--transition',action='store_true',help='Section 71: release plus the smeared-to-discrete transition (erosion kept as a logged safety net)')
    parser.add_argument('--band',type=float,help='Section 71.4: also transition fully formed cells (all damage nodes >= band) in released columns')
    parser.add_argument('--elastic-flank',action='store_true',help='Section 71.6: discrete-crack cells carry no plastic flow');args=parser.parse_args()
    if args.release:MODE.update(event='release',prefix='release-notch')
    if args.erode:MODE.update(event='release',prefix='erode-notch',erode=True)  # Section 68: release plus reactive erosion
    if args.transition:MODE.update(event='release',prefix='transition-notch',erode=True,transition=True)  # Section 71
    if args.band is not None:MODE.update(event='release',prefix='transition-band-notch',erode=True,transition=True,band=args.band)  # Section 71.4
    if args.elastic_flank:MODE.update(prefix='transition-elastic-notch',elastic=True)  # Section 71.6, with --band
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    run(args.run,out,args.source,args.subdivide,args.smoothing,args.substep) if args.run else assemble(out,args.subdivide)


if __name__=='__main__':main()
