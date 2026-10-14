"""Section 49 report: figure and markdown from saved reflection evidence."""
import argparse,json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());figures=out/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,4.4),layout='constrained')
    for b in r['branches']:
        h=b['history'];label=f"theta_R={b['angle']:g} deg"+(' (post hoc)' if b.get('post_hoc') else '')
        axes[0].plot(h['time'],np.array(h['distance'])-h['distance'][0],label=label)
    axes[0].axhline(0,color='0.5',lw=.8)
    axes[0].set(xlabel='Time after event',ylabel='Apex distance change from event',title='Reflected apex: departure (>0) or return (<0)');axes[0].legend(fontsize=8)
    f=r['post_hoc_forced_recoil'];t=np.array([p[0] for p in f['apex_x'] if p[1] is not None]);x=np.array([p[1] for p in f['apex_x'] if p[1] is not None])
    axes[1].plot(t,-x,color='tab:blue',lw=1)
    for e in f['events']:axes[1].axvline(e['time'],color='tab:red',ls=':',lw=1)
    axes[1].set(xlabel='Time',ylabel='Apex distance from interface',title='Forced same-system recoil: repeated encounters (red)')
    for ext in ('png','svg'):fig.savefig(figures/f'reflection.{ext}',dpi=170)
    plt.close(fig)
    tests=int(ET.parse(out/'test-results.xml').getroot().find('testsuite').attrib['tests']) if (out/'test-results.xml').exists() else None
    s=r['sequence'];e=r['post_hoc_seed_ensemble'];rel=out.relative_to(ROOT) if out.is_absolute() else out
    counts=Counter(e['event_counts']);terminal=Counter(str(x) for x in e['terminal'])
    lines=['# Sustained reflection, cross-system reflection and repeated encounters','',
        f"September 26, 2026.{' **'+str(tests)+' tests pass.**' if tests else ''} Section 49 explains the Section 20 reflection limitation, adds a cross-system reflection channel and replaces the one-encounter restriction with a repeated-encounter driver. All quantities are the synthetic nondimensional Section 20 closure; nothing here is calibrated to a material, and no calibration dataset exists in this project.",'',
        '## Why the original reflection returned','',
        'The original reflected branch keeps the incident Burgers vector, glide plane and line sense. Its line tension pulls the apex toward the chord between the two interface contacts, and the uniform applied stress exerts the same glide force that drove the approach. Only the decaying interface barrier opposes them. The recoil therefore must return; this is a property of the channel, not a numerical failure (theory 49.1). Sustained reflection needs a mobile line whose applied glide force points away from the interface.','',
        '## Cross-system reflection','',
        'A second incident-grain system rotated about y by theta_R carries b_R=Q_R b_in and leaves the residual b_in−b_R on the interface, exactly balancing both contact junctions. For the applied shear, the resolved stress is tau_in cos 2theta_R. The applied part of the implemented apex force, isolated as the difference of the full graph force with and without the applied strain, matches this law to 1e-9 relative at 0, 25, 40, 50, 60 and 75 degrees (automated test). Above 45 degrees the applied force reverses.','',
        '| theta_R | Declared prediction | Event energy jump | Apex distance at event → after 0.6 | Outcome | Checks |','| ---: | --- | ---: | --- | --- | --- |']
    for b in r['branches']:
        pred=b.get('prediction','— (post hoc)');hold='' if 'prediction_holds' not in b else (' — prediction holds' if b['prediction_holds'] else ' — **prediction fails**')
        lines.append(f"| {b['angle']:g}° | {pred} | {b['event_energy_jump']:+.5f} | {b['apex_distance_start']:.5f} → {b['apex_distance_end']:.5f} | {'departs' if b['sustained'] else 'returns'}{hold} | {'pass' if all(b['checks'].values()) else 'FAIL'} |")
    lines+=['',"The declared 60° prediction failed. Its applied force is reversed (−0.016 along the new glide direction, half the incident magnitude), but line tension and the interface/self-field terms of this embryo still outweigh it, so the apex returns. Only the post hoc 75° run departs, and only by 0.00025 over 0.6 time units, a marginal departure. The Frank–Read-type condition |tau_R| b R ≳ Gamma of theory 49.2 is the operative criterion: reversing the resolved shear is necessary but not sufficient. Sustained reflection is therefore demonstrated only marginally, for a steep second system, in this synthetic closure.",'',
        '## Repeated encounters','',
        f"The driver builds a five-member frozen family at every encounter (incident, transmitted, absorbed, same-system recoil, and cross-system reflection on a declared 60° second system) and samples one conditional first event. It records each event's energy jump and bath heat. The declared seeded sequence (seed {s['seed']}) drew cross-system reflection at its first encounter (t=0.1, probability {s['events'][0]['probabilities']['cross_reflected']:.3f}). That branch is terminal, so the declared run exercised only one event. Its ledger residual is {s['ledger_residual']:.1e}, and a restart from a checkpoint saved between events reproduces events, nodes, fields and energy bitwise.",'',
        f"Post hoc, after that outcome: a forced same-system recoil sequence meets the interface again at t={r['post_hoc_forced_recoil']['events'][1]['time']:.2f}, consistent with the ~16-unit return time estimated from the Section 20 return speed. Its two event jumps (+{r['post_hoc_forced_recoil']['events'][0]['energy_jump']:.5f}, +{r['post_hoc_forced_recoil']['events'][1]['energy_jump']:.5f}) are paid by the bath, the ledger residual is exactly zero, and the restart is bitwise. Across 32 further seeds, {counts.get(2,0)} sequences ({100*e['fraction_multi_event']:.1f}%) have two encounters, consistent with the 0.169 recoil probability per encounter. Terminal branches: {', '.join(f'{k} {v}' for k,v in terminal.items())} (None means still recoiling at t=20). Every ensemble run passes charge, continuous-energy and ledger checks (maximum residual {e['max_ledger_residual']:.1e}).",'',
        f'![Reflection branches and repeated encounters]({rel}/figures/reflection.png)','',
        '## Scope','',"Repeated encounters are sequences of frozen-family conditional first events. They do not prove detailed balance for an indefinitely reacting ensemble. Only incident-type graphs re-encounter the interface; a returning cross-reflected line would need its own reaction family, which is not constructed. Embryo sizes, barriers, the second system's orientation and all coefficients are synthetic. A calibrated repeated-collision law needs slip-system, barrier and embryo data that this project does not have.",'',
        '```sh',f'python3 benchmarks/reflection_sequence.py --output {rel}',f'python3 benchmarks/reflection_sequence_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 49](unified_multiscale_defect_framework.md).','']
    write_doc('REFLECTION_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
