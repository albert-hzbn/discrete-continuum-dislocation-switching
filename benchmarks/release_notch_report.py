"""Section 65 report: node release along the notch crack line, from saved evidence."""
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
GATES={'E1_ledger_and_checks':'R1 energy ledger and checks','E2_reach':'R2 all meshes reach t = 0.4','E3_t04':'R3 declared t = 0.4 gates'}  # the shared driver's names
pct=lambda v:'–' if v is None else f'{100*v:.3f}%'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());g=r['gates'];pairs=r['pairs'];runs=r['runs']
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    reach=lambda n:.4 if runs[n]['failure'] is None else runs[n]['failure']['last_accepted_time']
    shown=max(runs,key=lambda n:(reach(n),int(n)))
    if shown:
        n=int(shown);m,s,_=NodalThermalFracture2D.load(out/f"release-notch-{n}-0.00025-{'final' if runs[shown]['failure'] is None else 'last-accepted'}.npz");M=m.mechanics.mesh
        x=M.coordinates[:,0]+s.u[0::2];y=M.coordinates[:,1]+s.u[1::2];drawn=[]
        for c in range(len(M.cells)):
            corners=M.cells[c][[0,2,8,6,0]] if M.cells.shape[1]==9 else M.cells[c]
            if abs(M.coordinates[M.cells[c]].mean(0)[1]-.5)<.08 and M.coordinates[M.cells[c]].mean(0)[0]<.45:axes[0].plot(x[corners],y[corners],'k-',lw=.3);drawn+=list(y[corners])
        split=np.array(list(getattr(m,'split',{}).values()),dtype=int)
        if len(split):axes[0].plot(x[split],y[split],'r.',ms=2,label='released (upper face)')
        axes[0].set(title=f'Deformed mesh near the crack, t={reach(shown):.4g}, n={n}',xlim=(-.02,.45),ylim=(min(drawn)-.01,max(drawn)+.01),aspect='equal');axes[0].legend(fontsize=7)
    for i,(key,v) in enumerate(pairs.items()):
        ts=[(.4 if k=='final' else float(k[1:])) for k in v];axes[1].plot(ts,[100*v[k]['slip'] for k in v],'o-',color=f'C{i}',label=key.replace('_','/')+' slip')
        axes[1].plot(ts,[100*v[k]['damage'] for k in v],'s--',color=f'C{i}',mfc='none',label=key.replace('_','/')+' damage')
    axes[1].axhline(5,color='k',ls='--',lw=.8);axes[1].set(xlabel='Time',ylabel='Discrepancy (%)',title='Successive-mesh discrepancies with node release');axes[1].legend(fontsize=7)
    for n,x in runs.items():
        ev=x['events'];axes[2].step([e['time'] for e in ev],np.cumsum([e['count'] for e in ev]),where='post',label=f'n{n}')
    axes[2].set(xlabel='Time',ylabel='Released nodes',title='Cumulative node release');axes[2].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'release-notch.{ext}',dpi=160)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out
    lines=['# Node release: a displacement discontinuity for the separated notch','',
        'September 27, 2026. Section 65 replaces cell deletion by node release along the known crack line y = 0.5. Behind the fully broken front, the line nodes are duplicated and the cells above the line are reconnected to the copies, so the two crack faces separate as a genuine displacement discontinuity. Nothing is deleted. Each event conserves the total energy exactly, because a copy starts at its original\'s position and velocity with the lumped mass of its cells. The GLL n64, n96 and n128 notch runs restart at t = 0.2 with the subdivided certificate and the C² split of Section 64.','',
        '| Gate | Outcome |','| --- | --- |']+[f'| {GATES.get(k,k)} | {mark(v)} |' for k,v in g.items()]
    lines+=['','| Mesh | Reached | Released nodes | Relative energy drift | Checks |','| --- | --- | ---: | ---: | --- |']
    for n,x in runs.items():
        reach='t=0.4' if x['failure'] is None else f"partial t={x['failure']['last_accepted_time']:.5g} ({x['failure']['error'].rstrip('.')})"
        lines.append(f"| n{n} | {reach} | {x['deleted_cells']} | {x['relative_energy_drift']:.1e} | {'all pass' if all(x['checks'].values()) else x['checks']} |")
    lines+=['','| Pair | Time | c | η | T | slip | damage |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for key,v in pairs.items():
        for t in sorted(v,key=lambda k:.4 if k=='final' else float(k[1:])):
            e=v[t];lines.append(f"| {key.replace('_','/')} | {'0.4' if t=='final' else t[1:]} | {pct(e['c'])} | {pct(e['eta'])} | {pct(e['T'])} | {pct(e['slip'])} | {pct(e['damage'])} |")
    diagnostics=sorted(out.glob('*-fold-diagnostic.json'))
    if diagnostics:
        lines+=['','Each refused step was replayed from its last accepted state ([diagnostic script](benchmarks/release_fold_diagnostic.py)): a deeper certificate, dense sampling of the exact determinant in every flagged cell, and a certificate-bypassing replay of the step.','',
            '| Checkpoint | Retry at depth 12 | Last accepted min J | Flagged cells (centre, dense min J) | Certificate-bypassed step | dt/2, dt/4 | Genuine fold |','| --- | --- | ---: | --- | --- | --- | --- |']
        for f in diagnostics:
            x=json.loads(f.read_text());flagged='; '.join(f"({c['centre'][0]:.4f}, {c['centre'][1]:.4f}) {c['dense_min_J']:.4f}" for c in x.get('flagged',[]))
            b=x.get('bypassed_step',{});bypass=b.get('error') or f"min J {b['dense_min_J']:.4f}";small=', '.join(x.get('smaller_steps',{}).values()) or '–'
            lines.append(f"| t={x['time']:.5g}, {x['released']} released nodes | {x['retries'].get('12','–')} | {x.get('last_accepted_dense_min_J',{}).get('value',float('nan')):.3f} | {flagged or '–'} | {bypass} | {small} | {x.get('genuine_fold',False)} |")
    post=out/'post-hoc-substep'
    if (post/'release-notch-128.json').exists():
        x=json.loads((post/'release-notch-128.json').read_text());f=x['failure'];halved=sum(1 for r in x['rows'] if r.get('substeps',1)>1)
        dg=next(iter(sorted(post.glob('*-fold-diagnostic.json'))),None);ex=next(iter(sorted(post.glob('*-extension-depth*.json'))),None)
        lines+=['','## Post hoc n128 continuation (theory 65.4 and 65.5)','',
            f"The n128 run continues from t = {x['start_time']:.4g} with the same model and rules; a failed step is retried as halves, then quarters. {halved} steps needed halves. "+(f"It stops at t = {f['last_accepted_time']:.5g} ({f['error'].rstrip('.')}); all trajectory checks {'pass' if all(x['checks'].values()) else 'do not pass'}, with {x['deleted_cells']} further nodes released." if f else 'It reaches t = 0.4.')]
        if dg:
            d=json.loads(dg.read_text());lines.append(f"The refusal retried at depth 12: {d['retries'].get('12')}. Last accepted dense min J: {d['last_accepted_dense_min_J']['value']:.4f} in the cell centred at ({d['last_accepted_dense_min_J']['centre'][0]:.4f}, {d['last_accepted_dense_min_J']['centre'][1]:.4f}).")
        if ex:
            e=json.loads(ex.read_text());end=e['end'] or {}
            lines.append(f"A depth-{e['depth']} extension takes {len(e['rows'])} more step(s) (last min J {e['rows'][-1]['min_J']:.4f}) and is then refused on the step from t = {end.get('time',float('nan')):.5g}. The certificate-bypassed replay gives min J {end.get('bypassed_step',{}).get('dense_min_J',float('nan')):.4f}, so the fold is {'genuine' if end.get('genuine_fold') else 'not confirmed'}.")
        lines+=['','```sh',f'python3 benchmarks/deletion_notch.py --output {rel}/post-hoc-substep --run 128 --source {rel}/release-notch-128-0.00025-last-accepted.npz --subdivide 6 --smoothing 0.001 --release --substep',
            f'python3 benchmarks/release_fold_diagnostic.py {rel}/post-hoc-substep/release-notch-128-0.00025-last-accepted.npz',f'python3 benchmarks/release_mouth_extension.py {rel}/post-hoc-substep/release-notch-128-0.00025-last-accepted.npz','```']
    lines+=['',f'![Deformed mesh with released faces, discrepancies, releases]({rel}/figures/release-notch.png)','','```sh',
        f'for n in 64 96 128; do python3 benchmarks/deletion_notch.py --output {rel} --run $n --subdivide 6 --smoothing 0.001 --release & done; wait',
        f'python3 benchmarks/deletion_notch.py --output {rel} --subdivide 6 --release',f'python3 benchmarks/release_notch_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 65](unified_multiscale_defect_framework.md).','']
    write_doc('RELEASE_NOTCH_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
