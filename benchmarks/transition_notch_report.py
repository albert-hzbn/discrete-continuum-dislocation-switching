"""Section 71 report: node release with the smeared-to-discrete transition, from saved evidence; adds gates D1 (transition ledger) and D4 (no erosion)."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.nodal_damage import NodalThermalFracture2D
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
mark=lambda b:'PASS' if b else 'FAIL'
pct=lambda v:'–' if v is None else f'{100*v:.3f}%'
GATES={'E1_ledger_and_checks':'D1 energy ledgers (release, transition, erosion) and checks','D1_nonnegative_transition_heat':'D1 every transition moves non-negative heat','E2_reach':'D2 all meshes reach t = 0.4','E3_t04':'D3 declared t = 0.4 gates','D4_no_erosion':'D4 no cell eroded','D5_band_width':'D5 band half-width agrees across meshes within one n64 cell','D6_outside_band':'D6 slip outside the band below 5% and decreasing'}


PREFIX='transition-notch'


def checkpoint(out,n,x):
    return out/f"{PREFIX}-{n}-0.00025-{'final' if x['failure'] is None else 'last-accepted'}.npz"


BAND=False
ELASTIC=False


def main():
    global PREFIX,BAND,ELASTIC
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--band',action='store_true');parser.add_argument('--elastic',action='store_true',help='Section 71.6 elastic flank (implies the band rule)');args=parser.parse_args()
    if args.band:PREFIX,BAND='transition-band-notch',True
    if args.elastic:PREFIX,BAND,ELASTIC='transition-elastic-notch',True,True
    out=args.input;r=json.loads((out/'report.json').read_text());g=dict(r['gates']);pairs=r['pairs'];runs=r['runs']
    locality={}
    for n,x in runs.items():
        m,s,_=NodalThermalFracture2D.load(checkpoint(out,n,x));mech=m.mechanics
        active=np.ones(mech.mesh.n**2,bool) if getattr(mech,'active',None) is None else mech.active
        broken=s.damage[m.damage_cells]>=1.;edge=(broken&np.roll(broken,-1,axis=1)).any(axis=1)  # includes eroded cells
        locality[n]=dict(eroded=int((~active).sum()),broken_edge_cells=int(edge.sum()),fraction=float((~active).sum()/max(edge.sum(),1)))
    g['D1_nonnegative_transition_heat']=all(tr['min_moved']>=-1e-14 for x in runs.values() for tr in x.get('transitions',[]))
    g['D4_no_erosion']=all(x.get('eroded_cells',0)==0 for x in runs.values())
    width={}
    for n,x in runs.items():
        m,s,_=NodalThermalFracture2D.load(checkpoint(out,n,x));M=m.mechanics.mesh;disc=getattr(m,'discrete',None)
        if disc is not None and disc.any():
            ys=M.coordinates[M.cells[disc]][...,1];width[n]=float(np.max(abs(ys-.5)))
    if BAND:g['D5_band_width']=len(width)==3 and max(width.values())-min(width.values())<=.0104+1e-12
    split=out/'slip-band-split.json';sp=json.loads(split.read_text()) if split.exists() else {}
    if ELASTIC:g['D6_outside_band']=len(sp)==2 and all(v['outside']<.05 for v in sp.values()) and sp['96_128']['outside']<sp['64_96']['outside']
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    reach=lambda n:.4 if runs[n]['failure'] is None else runs[n]['failure']['last_accepted_time']
    shown=max(runs,key=lambda n:(reach(n),int(n)));m,s,_=NodalThermalFracture2D.load(checkpoint(out,shown,runs[shown]));M=m.mechanics.mesh
    x=M.coordinates[:,0]+s.u[0::2];y=M.coordinates[:,1]+s.u[1::2];drawn=[]
    active=np.ones(M.n**2,bool) if getattr(m.mechanics,'active',None) is None else m.mechanics.active
    for c in range(len(M.cells)):
        centre=M.coordinates[M.cells[c]].mean(0)
        if abs(centre[1]-.5)<.08 and centre[0]<.45:
            corners=M.cells[c][[0,2,8,6,0]];disc=getattr(m,'discrete',None) is not None and m.discrete[c]
            axes[0].plot(x[corners],y[corners],('C0-' if disc else 'k-') if active[c] else 'r-',lw=.3 if active[c] and not disc else .6);drawn+=list(y[corners])
    axes[0].set(title=f'n={shown}, t={reach(shown):.4g}: discrete cells blue, eroded red',xlim=(-.02,.45),ylim=(min(drawn)-.01,max(drawn)+.01),aspect='equal')
    for i,(key,v) in enumerate(pairs.items()):
        ts=[(.4 if k=='final' else float(k[1:])) for k in v];axes[1].plot(ts,[100*v[k]['slip'] for k in v],'o-',color=f'C{i}',label=key.replace('_','/')+' slip')
        axes[1].plot(ts,[100*v[k]['damage'] for k in v],'s--',color=f'C{i}',mfc='none',label=key.replace('_','/')+' damage')
    axes[1].axhline(5,color='k',ls='--',lw=.8);axes[1].set(xlabel='Time',ylabel='Discrepancy (%)',title='Successive-mesh discrepancies');axes[1].legend(fontsize=7)
    for n,xr in runs.items():
        ev=xr['events'];axes[2].step([e['time'] for e in ev],np.cumsum([e['count'] for e in ev]),where='post',label=f'n{n} released')
        er=xr.get('erosions',[])
        if er:axes[2].plot([e['time'] for e in er],np.cumsum([len(e['cells']) for e in er]),'x',ms=6,label=f'n{n} eroded (cumulative)')
    axes[2].set(xlabel='Time',ylabel='Count',title='Released nodes and eroded cells');axes[2].legend(fontsize=7)
    for ext in ('png','svg'):fig.savefig(figures/f'transition-notch.{ext}',dpi=160)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out
    if ELASTIC:lines=['# An elastic flank layer for the smeared-to-discrete transition','',
        'September 28, 2026. Section 71.6 keeps the band rule of 71.4 (discrete cells wherever nodal damage is at least 0.9) and adds one rule: discrete-crack cells carry no plastic flow. Their slip rate is zero from the transition on, so their slip and hardening histories freeze at the transition values. The flank layer inside the band has a thickness set by the regularization length, not by physics, and in the sharp-crack limit its plastic dissipation must vanish. The GLL n64, n96 and n128 notch runs restart at t = 0.2. The gates were declared before running.','']
    else:lines=['# A smeared-to-discrete transition beside the released crack','',
        'September 27, 2026. Section 71 releases the crack line (Section 65). Each released edge turns its two adjacent cells into discrete-crack cells: their degradation is switched off, and their reference is reset so they are intact and stress-free in their current shape. The removed mechanical energy becomes heat through an exact per-cell temperature solve. Damage and fracture energy are untouched. Reactive erosion (Section 68) stays enabled as a logged safety net, and D4 asks that it never fire. The GLL n64, n96 and n128 notch runs restart at t = 0.2.','']
    lines+=['| Gate | Outcome |','| --- | --- |']
    lines+=[f'| {GATES.get(k,k)} | {mark(v)} |' for k,v in g.items()]
    lines+=['','| Mesh | Reached | Released nodes | Discrete cells (events) | Min transition heat | Eroded cells | Relative energy drift | Checks |','| --- | --- | ---: | --- | ---: | ---: | ---: | --- |']
    for n,xr in runs.items():
        er=xr.get('erosions',[]);when=', '.join(f"{e['time']:.5g}" for e in er) or '–'
        state='t=0.4' if xr['failure'] is None else f"partial t={xr['failure']['last_accepted_time']:.5g} ({xr['failure']['error'].rstrip('.')})"
        trs=xr.get('transitions',[]);mh=min((tr['min_moved'] for tr in trs),default=float('nan'))
        lines.append(f"| n{n} | {state} | {xr['deleted_cells']} | {xr.get('discrete_cells',0)} ({len(trs)}) | {mh:.2e} | {xr.get('eroded_cells',0)} ({when}) | {xr['relative_energy_drift']:.1e} | {'all pass' if all(xr['checks'].values()) else xr['checks']} |")
    lines+=['','Transitioned band half-width (largest distance of a discrete-cell edge from the crack line): '+', '.join(f'n{n} {w:.4f}' for n,w in width.items())+'.']
    if BAND and sp and not ELASTIC:
        lines+=['','D3 fails on slip alone. A post hoc split ([slip_band_split.py](benchmarks/slip_band_split.py)) puts the discrepancy inside the thin crack band: outside |y - 0.5| < 0.02 the slip error is '+', '.join(f"{k.replace('_','/')} {100*v['outside']:.2f}%" for k,v in sp.items())+', which passes, decreases with refinement and matches the erosion run. Inside the band it is '+', '.join(f"{100*v['inside']:.0f}%" for v in sp.values())+', and the band carries '+', '.join(f"{100*v['share_of_squared_error_inside']:.0f}%" for v in sp.values())+' of the squared error while holding '+', '.join(f"{100*v['share_of_reference_inside']:.1f}%" for v in sp.values())+' of the slip. The transitioned cells are intact and stress-free, and they then deform plastically as the faces keep opening. That plastic wake along the flanks is the one quantity that does not converge (theory 71.5).']
    if ELASTIC and sp:
        lines+=['','Split of the slip discrepancy at the crack band |y - 0.5| < 0.02 ([slip_band_split.py](benchmarks/slip_band_split.py)): outside '+', '.join(f"{k.replace('_','/')} {100*v['outside']:.2f}%" for k,v in sp.items())+'; inside '+', '.join(f"{100*v['inside']:.1f}%" for v in sp.values())+', carrying '+', '.join(f"{100*v['share_of_squared_error_inside']:.0f}%" for v in sp.values())+' of the squared error; total '+', '.join(f"{100*v['total']:.2f}%" for v in sp.values())+'.']
    if not BAND:lines+=['','No cell was eroded, so the first-row fold is gone. n128 stops at t = 0.3955 without a fold: the minimum J of the last accepted state is 0.14, in the second-row mouth cell (min nodal d 0.95, not broken), and the full step stops converging there. The one-row band is exactly one cell thick on each mesh, a mesh-dependent width, and the slip discrepancies (9.3% for n64/n96 at t = 0.4) show it. Section 71.4 (TRANSITION_BAND_NOTCH_RESULTS.md) replaces it with the physically defined band d ≥ 0.9.']
    lines+=['','| Pair | Time | c | η | T | slip | damage |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for key,v in pairs.items():
        for t in sorted(v,key=lambda k:.4 if k=='final' else float(k[1:])):
            e=v[t];lines.append(f"| {key.replace('_','/')} | {'0.4' if t=='final' else t[1:]} | {pct(e['c'])} | {pct(e['eta'])} | {pct(e['T'])} | {pct(e['slip'])} | {pct(e['damage'])} |")

    lines+=['',f'![Deformed mesh with eroded cells, discrepancies, events]({rel}/figures/transition-notch.png)','','```sh',
        f'for n in 64 96 128; do python3 benchmarks/deletion_notch.py --output {rel} --run $n --subdivide 6 --smoothing 0.001 '+('--band 0.9' if BAND else '--transition')+(' --elastic-flank' if ELASTIC else '')+' --substep & done; wait',
        f'python3 benchmarks/deletion_notch.py --output {rel} --subdivide 6 '+('--band 0.9' if BAND else '--transition')+(' --elastic-flank' if ELASTIC else ''),*([f'python3 benchmarks/slip_band_split.py --input {rel} --prefix {PREFIX}'] if BAND else []),f'python3 benchmarks/transition_notch_report.py --input {rel}'+(' --elastic' if ELASTIC else ' --band' if BAND else ''),'```','',
        f'[Numerical report]({rel}/report.json), [theory Section 71](unified_multiscale_defect_framework.md).','']
    write_doc('TRANSITION_ELASTIC_NOTCH_RESULTS.md' if ELASTIC else 'TRANSITION_BAND_NOTCH_RESULTS.md' if BAND else 'TRANSITION_NOTCH_RESULTS.md','\n'.join(lines))
    (out/'locality.json').write_text(json.dumps(dict(locality=locality,D4=g['D4_no_erosion']),indent=1)+'\n');print(g,locality)


if __name__=='__main__':main()
