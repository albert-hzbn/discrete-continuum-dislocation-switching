"""Section 64 report: figures and markdown from saved element-deletion notch evidence."""
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());g=r['gates'];pairs=r['pairs'];runs=r['runs']
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    done=[n for n in ('128','96','64') if runs[n]['failure'] is None]
    if done:
        n=int(done[0]);path=out/f'deletion-notch-{n}-0.00025-final.npz';m,s,_=NodalThermalFracture2D.load(path);M=m.mechanics.mesh;xe,ye=M.edges
        axes[0].pcolormesh(xe,ye,s.damage.reshape(n+1,n+1).T,shading='gouraud',cmap='magma',vmin=0,vmax=1)
        dead=~m.mechanics.active if getattr(m.mechanics,'active',None) is not None else np.zeros(n*n,bool)
        xc=M.coordinates[M.cells].mean(1)[dead];axes[0].plot(xc[:,0],xc[:,1],'c.',ms=1.5,label='deleted cells')
        axes[0].set(title=f'Damage and deleted cells at t=0.4, n={n}',xlim=(0,.6),ylim=(.3,.7),aspect='equal');axes[0].legend(fontsize=7,loc='upper right')
    for key in pairs:
        ts=[(.4 if k=='final' else float(k[1:])) for k in pairs[key]];vals=[100*pairs[key][k]['slip'] for k in pairs[key]]
        axes[1].plot(ts,vals,'o-',label=key.replace('_','/'))
    php=out/'post-hoc-c2/report.json'
    if not php.exists():php=out/'post-hoc-subdivided/report.json'
    if php.exists():
        for i,(key,v) in enumerate(json.loads(php.read_text())['pairs'].items()):
            ts=[(.4 if k=='final' else float(k[1:])) for k in v];axes[1].plot(ts,[100*v[k]['slip'] for k in v],'s--',color=f'C{i}',mfc='none',label=key.replace('_','/')+' (post hoc)')
    axes[1].axhline(5,color='k',ls='--',lw=.8);axes[1].set(xlabel='Time',ylabel='Slip discrepancy (%)',title='Successive-mesh slip discrepancy with deletion');axes[1].legend(fontsize=8)
    for n,x in runs.items():
        ev=x['events'];axes[2].step([e['time'] for e in ev],np.cumsum([e['count'] for e in ev]),where='post',label=f'n{n}')
    axes[2].set(xlabel='Time',ylabel='Deleted cells',title='Cumulative deletions');axes[2].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'deletion-notch.{ext}',dpi=160)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out
    times=sorted({k for v in pairs.values() for k in v},key=lambda k:.4 if k=='final' else float(k[1:]))
    lines=['# Element deletion for the separated notch','',
        'September 26, 2026. Section 64 adds element deletion, the simplest discontinuity kinematics, to the finite thermochemical fracture model, then re-poses the declared Section 47.4 t = 0.4 gates. A cell is deleted once one of its edges is fully broken (both damage nodes at d = 1). Its mechanics vanish while its chemistry, phase and heat remain, and its elastic and thermoelastic energy becomes heat in the same cell, so the total energy is unchanged at every event (tested to 1e-13). The GLL n64, n96 and n128 trajectories restart from their t = 0.2 checkpoints, before any edge saturates.','',
        '| Gate | Outcome |','| --- | --- |']+[f'| {k} | {mark(v)} |' for k,v in g.items()]
    lines+=['','| Mesh | Reached | Deleted cells | Energy to heat | Relative energy drift | Checks |','| --- | --- | ---: | ---: | ---: | --- |']
    for n,x in runs.items():
        reach='t=0.4' if x['failure'] is None else f"partial t={x['failure']['last_accepted_time']:.5g} ({x['failure']['error'].rstrip('.')})"
        lines.append(f"| n{n} | {reach} | {x['deleted_cells']} | {x['energy_to_heat']:.2e} | {x['relative_energy_drift']:.1e} | {'all pass' if all(x['checks'].values()) else x['checks']} |")
    lines+=['','| Pair | Time | c | η | T | slip | damage |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for key,v in pairs.items():
        for t in times:
            if t in v:e=v[t];lines.append(f"| {key.replace('_','/')} | {'0.4' if t=='final' else t[1:]} | {pct(e['c'])} | {pct(e['eta'])} | {pct(e['T'])} | {pct(e['slip'])} | {pct(e['damage'])} |")
    ph=out/'post-hoc-subdivided/report.json'
    if ph.exists():
        q=json.loads(ph.read_text());pp=q['pairs']
        lines+=['','## Post hoc: subdivided orientation certificate','',
            "Both declared refusals (n96 at t = 0.2945, n128 at t = 0.27825) were conservative: the refused active cells, next to the deleted band at the free edge, had exact minimum determinants of 0.80 (n96) and 0.71 (n128). The plain Bernstein certificate bounds J by the minimum of its 16 coefficients. The subdivided certificate splits a patch whose coefficients are not all positive by de Casteljau at the midpoint, to depth 6. Sub-patch coefficients remain lower bounds of J and converge to its minimum, so the check stays rigorous (tested on random patches). n96 and n128 restart from their last accepted states with this certificate; the dynamics and the deletion rule are unchanged.",'',
            '| Gate (post hoc) | Outcome |','| --- | --- |']+[f'| {k} | {mark(v)} |' for k,v in q['gates'].items()]
        lines+=['','| Mesh | Reached | Deleted cells | Checks |','| --- | --- | ---: | --- |']
        for n,x in q['runs'].items():
            reach='t=0.4' if x['failure'] is None else f"partial t={x['failure']['last_accepted_time']:.5g} ({x['failure']['error'].rstrip('.')})"
            lines.append(f"| n{n} | {reach} | {x['deleted_cells']} | {'all pass' if all(x['checks'].values()) else x['checks']} |")
        lines+=['','| Pair | Time | c | η | T | slip | damage |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for key,v in pp.items():
            for t in sorted(v,key=lambda k:.4 if k=='final' else float(k[1:])):
                e=v[t];lines.append(f"| {key.replace('_','/')} | {'0.4' if t=='final' else t[1:]} | {pct(e['c'])} | {pct(e['eta'])} | {pct(e['T'])} | {pct(e['slip'])} | {pct(e['damage'])} |")
    c2=out/'post-hoc-c2/report.json'
    if c2.exists():
        q=json.loads(c2.read_text());pp=q['pairs']
        lines+=['','## Post hoc: C² energy split','',
            "The subdivided-certificate n128 run stopped at t = 0.30375 because the midpoint temperature equation had no solution. In a partially damaged cell, thermal expansion moved J_e across 1, where the Section 43 volumetric/deviatoric energy is only C¹: its compressive branch adds curvature 2μ/3 + λ for J_e < 1, so the heat capacity and thermoelastic coupling jump, and the temperature map jumps across its fixed point ([scan](results/deletion-notch-2026-09-26/post-hoc-subdivided/midpoint-stall-128.json)). A C² split multiplies the compressive energy by a quintic smoothstep in ln J_e over a band ε = 1e-3. Its derivatives are exact, it is unchanged outside the band, and its heat capacity is continuous; with it the stalled step converges in 14 iterations. All three meshes rerun from t = 0.2 with deletion, the subdivided certificate and the C² split.",'',
            '| Gate (post hoc, C²) | Outcome |','| --- | --- |']+[f'| {k} | {mark(v)} |' for k,v in q['gates'].items()]
        lines+=['','| Mesh | Reached | Deleted cells | Checks |','| --- | --- | ---: | --- |']
        for n,x in q['runs'].items():
            reach='t=0.4' if x['failure'] is None else f"partial t={x['failure']['last_accepted_time']:.5g} ({x['failure']['error'].rstrip('.')})"
            lines.append(f"| n{n} | {reach} | {x['deleted_cells']} | {'all pass' if all(x['checks'].values()) else x['checks']} |")
        lines+=['','| Pair | Time | c | η | T | slip | damage |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for key,v in pp.items():
            for t in sorted(v,key=lambda k:.4 if k=='final' else float(k[1:])):
                e=v[t];lines.append(f"| {key.replace('_','/')} | {'0.4' if t=='final' else t[1:]} | {pct(e['c'])} | {pct(e['eta'])} | {pct(e['T'])} | {pct(e['slip'])} | {pct(e['damage'])} |")
    lines+=['',f'![Damage with deleted cells, slip discrepancy, deletions]({rel}/figures/deletion-notch.png)','','```sh',
        f'for n in 64 96 128; do python3 benchmarks/deletion_notch.py --output {rel} --run $n & done; wait',f'python3 benchmarks/deletion_notch.py --output {rel}',f'python3 benchmarks/deletion_notch_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 64](unified_multiscale_defect_framework.md).','']
    write_doc('DELETION_NOTCH_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
