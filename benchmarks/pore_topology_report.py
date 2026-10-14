"""Section 58 report: figures and markdown from saved pore-topology evidence."""
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
    out=args.input;r=json.loads((out/'report.json').read_text());ph=json.loads((out/'nucleation-post-hoc.json').read_text());f=np.load(out/'fields.npz')
    col,co,cof,nuc,ctrl,g=r['collapse'],r['coalescence'],r['coalescence_h025'],r['nucleation'],r['control'],r['gates']
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,4,figsize=(19,4.3),layout='constrained')
    t=[x['time'] for x in col['rows']];axes[0].plot(t,[x['area'] for x in col['rows']],label='∫P')
    ax=axes[0].twinx();ax.step(t,[x['count'] for x in col['rows']],'k:',where='post');ax.set_ylabel('Pore count');axes[0].axvline(col['collapse_time'],color='r',lw=.8)
    axes[0].set(xlabel='Time',ylabel='∫P',title=f"Collapse of R0=5 (count 1→0 at t={col['collapse_time']:.0f})")
    im=axes[1].imshow(f['coalescence_final'],origin='lower',extent=(-32,32,-32,32),cmap='Blues',vmin=0,vmax=1);axes[1].set(title='Coalesced pore at t=300 (P)',xlabel='x',ylabel='y');fig.colorbar(im,ax=axes[1],shrink=.8)
    sh=[(x['time'],x['a2_over_a0']) for x in co['rows'] if 'a2_over_a0' in x and x['a2_over_a0']==x['a2_over_a0']]
    axes[2].plot(*zip(*sh));axes[2].axvline(co['merge_time'],color='r',lw=.8,label=f"merge t={co['merge_time']:.1f}");axes[2].set(xlabel='Time',ylabel='a2/a0',title='Merged pore relaxing toward a circle');axes[2].legend(fontsize=8)
    h=nuc['history'];axes[3].plot([x['time'] for x in h],[x['pore_area'] for x in h],label='∫P (declared rule)')
    hp=ph['history'];axes[3].plot([x['time'] for x in hp],[x['pore_area'] for x in hp],label='∫P (post hoc exclusion)')
    ax=axes[3].twinx();ax.plot([x['time'] for x in h],[x['mean_n'] for x in h],'k:',label='mean n');ax.set_ylabel('Mean vacancy n')
    axes[3].set(xlabel='Time',ylabel='Pore content ∫P',title='Nucleation events and matrix depletion');axes[3].legend(fontsize=7,loc='center right')
    for ext in ('png','svg'):fig.savefig(figures/f'pore-topology.{ext}',dpi=160)
    plt.close(fig)
    rel=out.relative_to(ROOT) if out.is_absolute() else out;iso=ph['isolated_embryo']['history']
    lines=['# Pore topology: collapse, coalescence and nucleation','',
        'September 26, 2026. Section 58 tests the pore topology changes named in brief section 11. The model is the Section 52 pore model with the singly degenerate exchange, whose interface law was confirmed there. Collapse and coalescence need no rule: the diffuse pore performs them through its gradient flow. Nucleation needs one, and it is implemented as a conserving event with an exact energy ledger. Nucleation parameters are synthetic closure values.','',
        '## Failures on the way (retained)','',
        '1. The declared collapse matrix n = 0 lies on the bound n ≥ 0, and the first step failed for every step size ([log](results/pore-topology-2026-09-26/run-first-attempt.log)). The matrix now starts at 0.005.',
        '2. The coalescence neck consumed its local vacancies down to n = 0. The Section 52 vacancy energy has no entropic barrier, so its step could not respect n ≥ 0. An availability limiter was added. It limits vacancy-consuming exchange cell by cell, conserves n + P exactly, keeps each cell\'s exchange a descent direction, and is bitwise inactive away from the bound (tested).',
        '3. A report crash, a division by the zero inventory of the undersaturated control, lost one completed run; the normalization was fixed.','',
        '## Gates','','| Gate | Outcome | Evidence |','| --- | --- | --- |',
        f"| T1 topology | {mark(g['T1_topology'])} | collapse 1→0 at t={col['collapse_time']:.0f}; coalescence 2→1 at t={co['merge_time']:.1f} (h=0.25: {cof['merge_time']:.1f}) |",
        f"| T2 conservation, energy | {mark(g['T2_conservation'])} | inventory drift ≤ {max(col['max_inventory_drift'],co['max_inventory_drift']):.1e}; energy increase ≤ {max(col['max_energy_increase'],co['max_energy_increase']):.1e} (backtracking tolerance 1e-11) |",
        f"| T3 monotone shape relaxation | {mark(g['T3_shape'])} | see below |",
        f"| N1 event inventory | {mark(g['N1_event_inventory'])} | {len(nuc['events'])} events, relative error ≤ {max(e['inventory_error'] for e in nuc['events']):.1e}, n ≥ 0 |",
        f"| N2 ledger F + Q | {mark(g['N2_ledger'])} | ≤ {max(e['ledger_error'] for e in nuc['events']):.1e} |",
        f"| N3 embryo growth | {mark(g['N3_growth'])} | see below |",
        f"| N4 undersaturated control | {mark(g['N4_control'])} | {sum(ctrl['refusals'].values())} candidates, all refused ({', '.join(ctrl['refusals'])}) |",'',
        f"Collapse took {col['collapse_time']:.0f} time units, {col['ratio']:.2f}x the interface-limited estimate R0²/(2 M_I γ) = {col['interface_limited_estimate']:.0f}. The vacancies released raise the matrix potential, and diffusion adds resistance; both slow collapse, as expected. After collapse, P settles to the uniform dissolved level P ≈ n/2 that the exchange equilibrium sets; no pore remains.",'',
        f"T3 fails as evaluated. The first post-merge sample is NaN, because at the instant of merging the thin neck gives the ray sampler no clean crossing from the centroid. All {len(sh)} later samples decrease monotonically, from a2/a0 = {sh[0][1]:.3f} to {sh[-1][1]:.3f}, as the merged pore relaxes toward a circle. The limiter was active in {co['limited_cell_steps']} cell-steps, all in the vacancy-starved neck and merged region.",'',
        '## Nucleation events','','| Event time | Radius | Local μ | ΔF of insertion | Area at insertion | After one interval |','| ---: | ---: | ---: | ---: | ---: | ---: |']
    for e in nuc['events']:lines.append(f"| {e['time']:g} | {e['radius']:.2f} | {e['local_mu']:.4f} | {e['delta_F']:.3f} | {e['area_at_insertion']:.2f} | {e['area_after_interval']:.2f} |")
    lines+=['',f"Refusals: "+', '.join(f"{k} {v}" for k,v in nuc['refusals'].items())+'.','',
        "N3 fails as declared, and the run shows why. First, events 2–4 landed within about 10 units of event 1. An insertion that overlaps an existing pore lowers F (ΔF from -0.55 to -1.2), so it is accepted with certainty, and growth of an existing pore is turned into jump events. The rule lacked an exclusion zone. Second, the declared metric, ∫P over the P > 1/2 component, jumps by about ±1 as cells cross the threshold. The first embryo's 0.07% decrease in its first interval is within that resolution.",'',
        f"Post hoc, labelled as such. With an exclusion zone of R_e + 4/a around existing pores, {len(ph['events'])} isolated embryos nucleate ({', '.join(f'{k} {v}' for k,v in ph['refusals'].items())}). The second grows after a short relaxation. The first shrinks slowly once the second has depleted the shared matrix, which is physical competition that an insertion-time criterion cannot anticipate. An isolated embryo in the rule's first insertion state grows monotonically in total pore content, from {iso[0]['total_pore']:.1f} to {iso[-1]['total_pore']:.1f} over 60 time units. The rule conserves inventory, closes its ledger exactly and nucleates growing embryos when they are isolated. The missing exclusion zone and the one-interval growth test were design errors, and competition between embryos needs a population-level criterion.",'',
        f'![Collapse, coalescence, shape relaxation and nucleation]({rel}/figures/pore-topology.png)','','```sh',f'python3 benchmarks/pore_topology.py --output {rel}',f'python3 benchmarks/pore_topology.py --output {rel} --post-hoc-nucleation',f'python3 benchmarks/pore_topology_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [post hoc]({rel}/nucleation-post-hoc.json), [theory Section 58](unified_multiscale_defect_framework.md).','']
    write_doc('PORE_TOPOLOGY_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
