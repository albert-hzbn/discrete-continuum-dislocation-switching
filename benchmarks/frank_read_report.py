"""Section 60 report: figures and markdown from saved Frank-Read evidence."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
mark=lambda b:'PASS' if b else 'FAIL'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());ph=json.loads((out/'post-hoc-descent.json').read_text());g=r['gates']
    sub,sup,below,above=r['subcritical'],r['supercritical'],r['below_critical'],r['above_critical']
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    line=np.array(sub['final']['line']);R=1/.8;th=np.linspace(0,np.pi,200);c=-np.sqrt(R*R-1)
    axes[0].plot(line[:,0],line[:,1],'o',ms=3,label='relaxed line, τ=0.8');axes[0].plot(R*np.cos(th),c+R*np.sin(th),'k--',lw=.8,label='arc R=Γ/(τb)=1.25')
    axes[0].plot([-1,1],[0,0],'ks',ms=6);axes[0].set(aspect='equal',title=f"Subcritical equilibrium (max deviation {sub['arc_relative_error']:.1e} R)",xlim=(-1.5,1.5),ylim=(-.3,1.));axes[0].legend(fontsize=8)
    src=ph['supercritical'];cmap=plt.get_cmap('viridis');keys=sorted(src['snapshots'],key=float)
    for i,k in enumerate(keys):
        col=cmap(i/max(1,len(keys)-1));sn=src['snapshots'][k];l=np.array(sn['line']);axes[1].plot(l[:,0],l[:,1],color=col,lw=1,label=f't={k}')
        for lp in sn['loops']:lp=np.array(lp+[lp[0]]);axes[1].plot(lp[:,0],lp[:,1],color=col,lw=.8,ls='--')
    fl=src['final'];l=np.array(fl['line']);axes[1].plot(l[:,0],l[:,1],'r',lw=1.2,label=f"t={src['end_time']:.1f} (final)")
    for lp in fl['loops']:lp=np.array(lp+[lp[0]]);axes[1].plot(lp[:,0],lp[:,1],'r--',lw=.8)
    axes[1].plot([-1,1],[0,0],'ks',ms=6);axes[1].set(aspect='equal',title='Frank–Read source, τ=1.2 (post hoc rule): line and emitted loops');axes[1].legend(fontsize=7,loc='upper right')
    for run,label in ((below,'τ=0.95'),(above,'τ=1.05'),(sup,'τ=1.2')):
        axes[2].step([h['time'] for h in run['history']],[h['loops'] for h in run['history']],where='post',label=label)
    axes[2].set(xlabel='Time',ylabel='Emitted loops',title='Multiplication above τ_c = 2Γ/(bL) = 1 (declared rule)');axes[2].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'frank-read.{ext}',dpi=160)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out
    ev=lambda run:', '.join(f"t={e['time']:.2f} (ΔF={e['delta_F']:+.3f})" for e in run['events']) or 'none'
    lines=['# Dislocation multiplication: a Frank–Read source','',
        'September 26, 2026. Section 60 implements dislocation multiplication (brief section 11) through the canonical Frank–Read mechanism (Frank and Read, 1950) in the line-tension approximation of discrete dislocation dynamics. A pinned line bows under the resolved shear. Above τ_c = 2Γ/(bL) it has no equilibrium: it wraps around the pins and reconnects behind the source by annihilating antiparallel segments, emitting a closed loop and restoring the source. Every force is the exact gradient of the polygonal energy (tested), remeshing never raises the energy, and each reconnection conserves the Burgers carriers and books its energy change as heat. Self-stress beyond line tension is neglected.','',
        '## Gates (declared rule)','','| Gate | Outcome | Evidence |','| --- | --- | --- |',
        f"| F1 subcritical arc | {mark(g['F1_subcritical_arc'])} | τ=0.8: nodes within {sub['arc_relative_error']:.1e} R of the circle R = 1.25 through the pins; no event; energy never increased |",
        f"| F2 multiplication | {mark(g['F2_multiplication'])} | τ=1.2: events at {ev(sup)}; the line re-pins at A and B after each |",
        f"| F3 events | {mark(g['F3_events'])} | ledger errors ≤ {max([e['ledger_error'] for x in (sup,above) for e in x['events']] or [0]):.1e}; requires every ΔF ≤ 0 |",
        f"| F4 critical stress | {mark(g['F4_critical'])} | τ=0.95: {ev(below)}; τ=1.05: {ev(above)} |",'',
        f"Post hoc rule, labelled as such. A reconnection is deferred while it would raise F. With it, τ = 1.2 emits at {ev(ph['supercritical'])}, and τ = 1.05 emits at {ev(ph['above_critical'])}. The declared rule accepted the closest antiparallel pair regardless of energy; when the pair is staggered, the swapped connections lose more swept area than the line length they remove. Deferring such events until the arms are closer keeps every event dissipative.",'',
        f'![Subcritical arc, loop emission and multiplication by stress]({rel}/figures/frank-read.png)','',
        '## Scope','','Line tension only: no elastic self-stress or loop–source interaction, so the emitted loops expand without back-stress, and a source in a real crystal operates slightly differently from τ_c = 2Γ/(bL). The capture distance is a closure. The full elastic loop mechanics of Section 19 covers closed loops but not pinned open lines. The critical stress, arc shape and emission cycle are the classical line-tension results, reproduced here as the multiplication rule of theory 7.','',
        '```sh',f'python3 benchmarks/frank_read.py --output {rel}',f'python3 benchmarks/frank_read.py --output {rel} --post-hoc-descent',f'python3 benchmarks/frank_read_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [post hoc]({rel}/post-hoc-descent.json), [theory Section 60](unified_multiscale_defect_framework.md).','']
    write_doc('FRANK_READ_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
