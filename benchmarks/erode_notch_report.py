"""Section 68 report: node release with reactive erosion, from saved evidence; adds the declared locality gate X4."""
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
GATES={'E1_ledger_and_checks':'X1 energy ledger and checks','E2_reach':'X2 all meshes reach t = 0.4','E3_t04':'X3 declared t = 0.4 gates'}


def checkpoint(out,n,x):
    return out/f"erode-notch-{n}-0.00025-{'final' if x['failure'] is None else 'last-accepted'}.npz"


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());g=dict(r['gates']);pairs=r['pairs'];runs=r['runs']
    locality={}
    for n,x in runs.items():
        m,s,_=NodalThermalFracture2D.load(checkpoint(out,n,x));mech=m.mechanics
        active=np.ones(mech.mesh.n**2,bool) if getattr(mech,'active',None) is None else mech.active
        broken=s.damage[m.damage_cells]>=1.;edge=(broken&np.roll(broken,-1,axis=1)).any(axis=1)  # includes eroded cells
        locality[n]=dict(eroded=int((~active).sum()),broken_edge_cells=int(edge.sum()),fraction=float((~active).sum()/max(edge.sum(),1)))
    g['X4_locality']=all(v['fraction']<=.10 for v in locality.values())
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    reach=lambda n:.4 if runs[n]['failure'] is None else runs[n]['failure']['last_accepted_time']
    shown=max(runs,key=lambda n:(reach(n),int(n)));m,s,_=NodalThermalFracture2D.load(checkpoint(out,shown,runs[shown]));M=m.mechanics.mesh
    x=M.coordinates[:,0]+s.u[0::2];y=M.coordinates[:,1]+s.u[1::2];drawn=[]
    active=np.ones(M.n**2,bool) if getattr(m.mechanics,'active',None) is None else m.mechanics.active
    for c in range(len(M.cells)):
        centre=M.coordinates[M.cells[c]].mean(0)
        if abs(centre[1]-.5)<.08 and centre[0]<.45:
            corners=M.cells[c][[0,2,8,6,0]];axes[0].plot(x[corners],y[corners],'k-' if active[c] else 'r-',lw=.3 if active[c] else .8);drawn+=list(y[corners])
    axes[0].set(title=f'Deformed mesh near the crack, t={reach(shown):.4g}, n={shown} (eroded cells red)',xlim=(-.02,.45),ylim=(min(drawn)-.01,max(drawn)+.01),aspect='equal')
    for i,(key,v) in enumerate(pairs.items()):
        ts=[(.4 if k=='final' else float(k[1:])) for k in v];axes[1].plot(ts,[100*v[k]['slip'] for k in v],'o-',color=f'C{i}',label=key.replace('_','/')+' slip')
        axes[1].plot(ts,[100*v[k]['damage'] for k in v],'s--',color=f'C{i}',mfc='none',label=key.replace('_','/')+' damage')
    axes[1].axhline(5,color='k',ls='--',lw=.8);axes[1].set(xlabel='Time',ylabel='Discrepancy (%)',title='Successive-mesh discrepancies');axes[1].legend(fontsize=7)
    for n,xr in runs.items():
        ev=xr['events'];axes[2].step([e['time'] for e in ev],np.cumsum([e['count'] for e in ev]),where='post',label=f'n{n} released')
        er=xr.get('erosions',[])
        if er:axes[2].plot([e['time'] for e in er],np.cumsum([len(e['cells']) for e in er]),'x',ms=6,label=f'n{n} eroded (cumulative)')
    axes[2].set(xlabel='Time',ylabel='Count',title='Released nodes and eroded cells');axes[2].legend(fontsize=7)
    for ext in ('png','svg'):fig.savefig(figures/f'erode-notch.{ext}',dpi=160)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out
    lines=['# Node release with reactive erosion for the separated notch','',
        'September 27, 2026. Section 68 keeps the Section 65 displacement discontinuity along the crack line. A cell is removed only when the rigorous orientation certificate refuses a step because of it, and only if the cell already has a fully broken edge. Its mechanical internal energy becomes heat in place (the Section 64 ledger), and the step is retried; a midpoint failure is retried with halved steps. The GLL n64, n96 and n128 notch runs restart at t = 0.2.','',
        '| Gate | Outcome |','| --- | --- |']+[f'| {GATES.get(k,k)} | {mark(v)} |' for k,v in g.items()]
    lines+=['','| Mesh | Reached | Released nodes | Eroded cells (times) | Eroded / broken-edge cells | Relative energy drift | Checks |','| --- | --- | ---: | --- | ---: | ---: | --- |']
    for n,xr in runs.items():
        er=xr.get('erosions',[]);when=', '.join(f"{e['time']:.5g}" for e in er) or '–'
        state='t=0.4' if xr['failure'] is None else f"partial t={xr['failure']['last_accepted_time']:.5g} ({xr['failure']['error'].rstrip('.')})"
        lines.append(f"| n{n} | {state} | {xr['deleted_cells']} | {xr.get('eroded_cells',0)} ({when}) | {locality[n]['eroded']}/{locality[n]['broken_edge_cells']} = {100*locality[n]['fraction']:.1f}% | {xr['relative_energy_drift']:.1e} | {'all pass' if all(xr['checks'].values()) else xr['checks']} |")
    lines+=['','| Pair | Time | c | η | T | slip | damage |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for key,v in pairs.items():
        for t in sorted(v,key=lambda k:.4 if k=='final' else float(k[1:])):
            e=v[t];lines.append(f"| {key.replace('_','/')} | {'0.4' if t=='final' else t[1:]} | {pct(e['c'])} | {pct(e['eta'])} | {pct(e['T'])} | {pct(e['slip'])} | {pct(e['damage'])} |")
    lines+=['','The declared t = 0.4 mesh gates (X3) pass: every field is below 5%, and the finer pair beats the coarser one in slip and damage. The locality gate X4 fails at n128. The eroded fraction grows with refinement, and so does the eroded zone at the crack mouth: it reaches x ≈ 0.004, 0.018 and 0.041 on n64, n96 and n128, plus a few cells at a hinge of the opening near x ≈ 0.24–0.29. The compared fields converge because eroded cells carried almost no stress, but the eroded set itself does not converge (theory 68.3). In the figure, eroded cells look stretched because nodes that belong only to eroded cells carry no stiffness and drift at constant velocity, which is energy-neutral.']
    lines+=['',f'![Deformed mesh with eroded cells, discrepancies, events]({rel}/figures/erode-notch.png)','','```sh',
        f'for n in 64 96 128; do python3 benchmarks/deletion_notch.py --output {rel} --run $n --subdivide 6 --smoothing 0.001 --erode --substep & done; wait',
        f'python3 benchmarks/deletion_notch.py --output {rel} --subdivide 6 --erode',f'python3 benchmarks/erode_notch_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 68](unified_multiscale_defect_framework.md).','']
    write_doc('EROSION_NOTCH_RESULTS.md','\n'.join(lines))
    (out/'locality.json').write_text(json.dumps(dict(locality=locality,X4=g['X4_locality']),indent=1)+'\n');print(g,locality)


if __name__=='__main__':main()
