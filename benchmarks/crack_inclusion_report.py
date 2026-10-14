"""Section 62 report: figures and markdown from saved crack-inclusion evidence."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from crack_inclusion import RUNS,CENTER,RADIUS
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
mark=lambda b:'PASS' if b else 'FAIL'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());g=r['gates'];runs={k:json.loads((out/f'{k}.json').read_text()) for k in RUNS}
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,6,figsize=(26,4.4),layout='constrained')
    th=np.linspace(0,2*np.pi,100)
    for ax,k in zip(axes[:4],RUNS):
        snaps=np.load(out/f'{k}.npz');key=sorted(snaps.files,key=float)[-1];n=runs[k]['n'];q=snaps[key].reshape(n+1,n+1)
        ax.imshow(q.T,origin='lower',extent=(0,1,0,1),cmap='magma',vmin=0,vmax=1);ax.plot(CENTER[0]+RADIUS*np.cos(th),CENTER[1]+RADIUS*np.sin(th),'c--',lw=.8)
        ax.set(title=f"κ={runs[k]['kappa']:g}, n={n}, load {float(key):.3f}",xticks=[],yticks=[])
    for k in RUNS:
        rows=runs[k]['rows'];axes[4].plot([x['load'] for x in rows],[x['tip'] for x in rows],label=k);axes[5].plot([x['load'] for x in rows],[x['reaction'] for x in rows],label=k)
    axes[4].axhspan(CENTER[0]-RADIUS,CENTER[0]+RADIUS,color='c',alpha=.15);axes[4].set(xlabel='Load',ylabel='Rightmost x with d ≥ 0.8',title='Crack tip (band: inclusion)');axes[4].legend(fontsize=7)
    axes[5].set(xlabel='Load',ylabel='Reaction',title='Load–displacement response');axes[5].legend(fontsize=7)
    for ext in ('png','svg'):fig.savefig(figures/f'crack-inclusion.{ext}',dpi=150)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out;fin=r['final'];arr=r['arrest']
    lines=['# Crack arrest, deflection and branching at a tough inclusion','',
        'September 26, 2026. Section 62 exercises crack initiation, arrest, deflection and branching (brief section 11) in a spatial computation. A notch grows under mode-I displacement loading in the Section 36 plane-strain AT2 model, toward a circular inclusion whose toughness is κ times the matrix value. Elasticity is homogeneous, so only the toughness varies. The path, whether it penetrates, arrests, deflects or branches, is left entirely to the variational solver.','',
        '| Run | Final tip x | Mean damage inside inclusion | Branches at x = 0.6 | Arrest interval (fraction of arrival load) | Checks |','| --- | ---: | ---: | ---: | ---: | --- |']
    for k in RUNS:
        f=fin[k];a=arr[k]
        interval='–' if a is None else '%.2f'%a['interval_fraction']
        lines.append(f"| {k} | {f['tip']:.3f} | {f['inside_mean']:.3f} | {f['branches_at_center']} | {interval} | {'all pass' if all(runs[k]['checks'].values()) else runs[k]['checks']} |")
    lines+=['','| Gate | Outcome as evaluated |','| --- | --- |']+[f"| {k} | {mark(v)} |" for k,v in g.items()]
    lines+=['',"**B3 is a false pass, reclassified as failed.** The fields show that at κ = 4 the notch crack arrested at the inclusion and never went around it. The specimen then failed by a second crack that initiated along the loaded edge (top at n64, bottom at n96; the side is a roundoff choice under symmetric loading). The B3 metric, the rightmost damage ≥ 0.8 anywhere, was met by that edge crack. What the declared runs demonstrate is penetration below the threshold (κ ≤ 1.25, consistent with the π/2 lower bound), arrest above it (a 14% load interval, identical at n64 and n96), and failure transferring to a new initiation site. Deflection and branching around the inclusion were not observed."]
    ph=out/'post-hoc-tough-edges'
    if ph.exists() and any(ph.glob('*.json')):
        lines+=['','Post hoc, labelled as such. The loaded-edge strips (y < 0.06 and y > 0.94) are toughened tenfold so that failure cannot move to the grips:','','| Run | Final tip x | Mean damage inside inclusion | Branches at x = 0.6 | Crossed the inclusion on the centreline | Broken at x = 0.6, y range |','| --- | ---: | ---: | ---: | --- | --- |']
        for f in sorted(ph.glob('*.json')):
            x=json.loads(f.read_text());last=x['rows'][-1];snaps=np.load(ph/f'{x["name"]}.npz');key=sorted(snaps.files,key=float)[-1];n=x['n'];q=snaps[key].reshape(n+1,n+1)
            col=q[int(round(.6*n)),:];y=np.linspace(0,1,n+1)[col>=.8];yr='none' if not len(y) else f'{y.min():.3f}–{y.max():.3f}'
            lines.append(f"| {x['name']} | {last['tip']:.3f} | {last['inside_mean']:.3f} | {last['branches_at_center']} | {'yes' if any(r['centerline_through_inclusion'] for r in x['rows']) else 'no'} | {yr} |")
            if x['failure']:lines.append(f"| {x['name']} stopped | {x['failure']['message'][:80]} | | | | |")
        lines+=['',"With the grips protected, κ = 2 still penetrates the inclusion, while κ = 4 deflects: the crack turns, runs around the inclusion on one side and continues past it. The deflection threshold therefore lies in (2, 4], above the sharp-interface lower bound π/2, as the mode mixity of a deflected path requires. The deflection is single-sided. Two branches would cost twice the surface energy, and the solver leaves the symmetric path. Symmetric branching was not observed in any run.",'',
            f'![Post hoc tough-edge runs: penetration at κ=2, deflection at κ=4]({rel}/post-hoc-tough-edges/figures/tough-edges.png)']
    lines+=['',f'![Final damage fields, crack tip and reaction]({rel}/figures/crack-inclusion.png)','','```sh',
        f'for r in k1-n64 k1.25-n64 k4-n64 k4-n96; do python3 benchmarks/crack_inclusion.py --output {rel} --run $r & done; wait',f'python3 benchmarks/crack_inclusion.py --output {rel}',f'python3 benchmarks/crack_inclusion_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 62](unified_multiscale_defect_framework.md).','']
    write_doc('CRACK_INCLUSION_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
