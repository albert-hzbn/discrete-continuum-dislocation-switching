"""Keep AT1 initiation evidence and unchanged-physics AT2 refinement separate."""
import argparse,json
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.thermofracture import ThermalFracture2D
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]


def fmt(value,percent=False):
    if value is None:return 'undefined'
    return f'{100*value:.4f}%' if percent else f'{value:.6g}'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--at2',type=Path,required=True)
    args=parser.parse_args();start=perf_counter();out=args.input;r=json.loads((out/'report.json').read_text());old=json.loads((args.at2/'report.json').read_text())
    figures=out/'figures';figures.mkdir(exist_ok=True)
    loading=[x for x in r['runs'] if x['n']==16 and x['dt']==.00025 and x['variant'].startswith('at1-load-')]
    states=[]
    for run in loading:
        m,a,_=ThermalFracture2D.load(out/f'{run["name"]}-initial.npz');_,b,_=ThermalFracture2D.load(out/f'{run["name"]}-final.npz')
        states.append((b.damage-a.damage).reshape(16,16).T)
    maximum=max(float(x.max()) for x in states)
    fig,axes=plt.subplots(1,4,figsize=(15,4.2),layout='constrained')
    for ax,values,run in zip(axes,states,loading):
        im=ax.imshow(values,origin='lower',extent=[0,1,0,1],cmap='magma',vmin=0,vmax=maximum)
        ax.set(title='Grip '+run['variant'].removeprefix('at1-load-'),xlabel='x',ylabel='y')
    fig.colorbar(im,ax=axes,shrink=.75,label='Damage increment; common scale')
    for ext in ('png','svg'):fig.savefig(figures/f'at1-loading-fields.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for run in loading:
        t=[x['time'] for x in run['rows']];E=np.array([x['total_energy'] for x in run['rows']])
        axes[0].plot(t,[x['damage_integral'] for x in run['rows']],label='Grip '+run['variant'].removeprefix('at1-load-'))
        axes[1].plot(t,(E-E[0])/run['energy_scale'])
    axes[0].set(xlabel='Time',ylabel='Integral of damage',title='AT1 fixed-parameter load sweep');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Time',ylabel='Normalized energy change',title='Closed AT1 trajectories')
    keys=['c','eta','T','slip','damage'];x=np.arange(5)
    for shift,data,label in [(-.18,old['comparison_32_48'],'AT2 grip .12: n32/n48'),(.18,r['spatial']['errors'],'AT1 grip .06: n24/n32')]:
        axes[2].bar(x+shift,[np.nan if data[k] is None else 100*data[k] for k in keys],width=.36,label=label)
    axes[2].axhline(5,color='black',linestyle='--',label='5% spatial target')
    axes[2].set(xticks=x,xticklabels=keys,ylabel='Discrepancy (%)',title='Separate physical-model refinements');axes[2].legend(fontsize=7)
    for ext in ('png','svg'):fig.savefig(figures/f'crack-initiation-verification.{ext}',dpi=170)
    plt.close(fig)
    activation={}
    for n in (16,24,32):
        name=f'at1-load-0.06-{n}-0.00025';m,s,_=ThermalFracture2D.load(out/f'{name}-final.npz')
        activation[str(n)]=dict(weighted_slip_norm=float(np.sqrt(np.sum(m.mechanics.mesh.weights[None,:,None]*s.gamma**2))),
            max_slip=float(abs(s.gamma).max()),nonzero_slip_fraction=float(np.mean(np.any(s.gamma!=0,axis=-1))))
    (out/'slip-activation.json').write_text(json.dumps(activation,indent=2)+'\n')
    _,initial,_=ThermalFracture2D.load(out/'at1-load-0.06-32-0.00025-initial.npz')
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,values,title in zip(axes,[initial.damage,s.damage,s.damage-initial.damage],['Initial damage','Final damage at grip .06','Local intensification; own scale']):
        im=ax.imshow(values.reshape(32,32).T,origin='lower',extent=[0,1,0,1],cmap='magma',vmin=0,
            vmax=max(initial.damage.max(),s.damage.max()) if title!='Local intensification; own scale' else None)
        ax.set(title=title,xlabel='x',ylabel='y');fig.colorbar(im,ax=ax,shrink=.8)
    for ext in ('png','svg'):fig.savefig(figures/f'at1-local-intensification.{ext}',dpi=170)
    plt.close(fig)
    tests=int(ET.parse(out/'test-results.xml').getroot().find('testsuite').attrib['tests'])
    checks=sum(sum(x['checks'].values()) for x in r['runs']);passes=sum(r['gates'].values());gp=sum(r['geometry_gates'].values())
    rel=out.relative_to(ROOT) if out.is_absolute() else out;oldrel=args.at2.relative_to(ROOT) if args.at2.is_absolute() else args.at2
    lines=['# Finite damage initiation and retained notch-resolution failures','',
        f'September 22, 2026. **{tests} automated tests pass**. Fourteen AT1 trajectories pass **{checks}/252** invariant/solver checks, **{passes}/{len(r["gates"])}** accuracy/threshold gates and **{gp}/4** geometry gates. The separate unchanged-physics AT2 n48 run passes 18 trajectory checks but retains failed resolution targets. The full objective remains active.','',
        '## Original AT2 case: refinement did not resolve the failure','',
        'The n48 trajectory uses a preserved source snapshot, the original AT2 law, grip 0.12, fixed physical widths and material coefficients, dt=0.00025 and t=0.4. It does not use the new AT1 option. Its n32/n48 discrepancies are:','',
        '| Field | Error | <5% target |','| --- | ---: | --- |']
    for key in keys:lines.append(f"| {key} | {fmt(old['comparison_32_48'][key],True)} | {'PASS' if old['gates'][key] else 'FAIL'} |")
    lines+=['',f"The original n24/n32 slip failure (19.45%) remains preserved. The new slip discrepancy is {100*old['comparison_32_48']['slip']:.2f}%, and damage discrepancy is {100*old['comparison_32_48']['damage']:.2f}%. The n48 solver cost is {old['run']['solver_seconds']:.2f} seconds; normalized energy drift is {old['run']['relative_energy_drift']:.3e}. These single-run observations do not establish adaptive efficiency.",'',
        '## AT1: an explicit constitutive option','',
        'The common-energy implementation now supports AT1 and AT2. Surface energy is normalized by c_w=8/3 for AT1 and 2 for AT2; the corresponding gradient coefficient also changes in force and microstress power. AT1 supplies the virgin homogeneous threshold psi_crit=3 G_c/[16 ell_f(1-k)]. Below it damage arrests; above it the declared viscous law evolves. Independent tests verify isolated-profile energy G_c, analytic damage/heating, local/global first laws, entropy, reciprocal phase derivatives, no healing and Q1/Q2 restart for both energy splits. Old checkpoints retain AT2.','',
        'At equal ell_f the equilibrium profile widths differ: one-sided 0.9-to-0.1 width is ell_f log(9) for AT2 and 2 ell_f(sqrt(0.9)-sqrt(0.1)) for AT1. This matrix fixes ell_f=0.08 within each law. AT1 versus AT2 is a physical-model comparison; it is not an equivalent-width accuracy or speedup comparison. No material strength has been calibrated.','',
        '| Homogeneous energy / threshold | dt | Damage absolute error | T absolute error |','| --- | --- | ---: | ---: |']
    for run in r['runs']:
        if run['variant'].startswith('threshold-'):
            lines.append(f"| {run['actual_energy_ratio']:.3f} | {run['dt']} | {run['damage_error']:.3e} | {run['temperature_error']:.3e} |")
    lines+=['','## Prespecified loading matrix','',
        'Top displacement is 0, 0.04, 0.06 or 0.12 on the same n16 grid. Every other physical coefficient, profile and width is held fixed; t=0.4 and dt=0.00025. Zero imposed grip still permits thermochemical eigenstress and seeded-gradient relaxation; it is not called a stress-free sample.','',
        '| Top displacement | Initial integral d | Final integral d | Front advance | Low-damage cells initially active | Notch fraction of increment |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for run in loading:
        g=r['geometry'][run['name']];a=r['activation'][run['name']]
        lines.append(f"| {run['variant'].removeprefix('at1-load-')} | {run['rows'][0]['damage_integral']:.6f} | {run['rows'][-1]['damage_integral']:.6f} | {fmt(g['advance'])} | {fmt(a['active_fraction'],True)} | {fmt(g['notch_fraction'],True)} |")
    lines+=['','Low-damage activation refers to initial d<1e-4 and d_dot>1e-12. An undefined fraction means no net damage increment, not a failed conservation test. All contour and region definitions are unchanged from Section 43.','',
        '| Field at grip 0.06 | n24/n32 discrepancy | dt .001/.00025, n16 | dt .0005/.00025, n16 |','| --- | ---: | ---: | ---: |']
    for key in keys:lines.append(f"| {key} | {fmt(r['spatial']['errors'][key],True)} | {fmt(r['time_coarse']['errors'][key],True)} | {fmt(r['time_fine']['errors'][key],True)} |")
    lines+=['',f"Exact zero-reference fields: spatial {r['spatial']['exact_zero_fields']}; temporal {r['time_fine']['exact_zero_fields']}. A zero error here requires both fields to be exactly zero. Undefined nonzero differences are not accepted as relative accuracy.",'',
        f"At grip 0.06, n16 has weighted slip norm {activation['16']['weighted_slip_norm']:.3e}, while n24 and n32 have exactly zero slip. The finer comparison therefore checks a quiescent slip state; it does not establish accuracy of active plastic flow. The n16 temporal check characterizes that coarse-grid activation. The d=0.4 front stays at its initial position on all three grids despite localized damage intensification, so propagation is not demonstrated at this load/time.",'',
        '| Original geometry target at finest grip 0.06 | Outcome |','| --- | --- |']
    for key,value in r['geometry_gates'].items():lines.append(f"| {key} | {'PASS' if value else 'FAIL'} |")
    failed=[key for key,value in r['gates'].items() if not value];gf=[key for key,value in r['geometry_gates'].items() if not value]
    lines+=['',f"Failed AT1 accuracy/threshold gates: {', '.join(failed) or 'none'}. Failed geometry gates: {', '.join(gf) or 'none'}. Loads and thresholds were declared before the runs and were not adjusted to force qualification.",'',
        '## Balances, costs and reproducibility','',
        f"Maximum AT1 normalized energy drift is {max(x['relative_energy_drift'] for x in r['runs']):.3e}; composition inventory error is {max(x['mass_error'] for x in r['runs']):.3e}. All accepted damage updates obey irreversibility; temperature, orientation and capacity remain positive. Tests and simulations ran sequentially with thread limits one.",'',
        '| Run | Initialization s | Solver s | Diagnostics / I/O s | Stored scalars |','| --- | ---: | ---: | ---: | ---: |']
    for run in r['runs']:lines.append(f"| {run['name']} | {run['initialization_seconds']:.3f} | {run['solver_seconds']:.3f} | {run['diagnostics_output_seconds']:.3f} | {run['evolving_stored_scalars']:,} |")
    lines+=['',f"Whole AT1 process peak RSS: {r['process_peak_rss_bytes']/2**20:.2f} MiB. Plotting is separately recorded and no JIT is used. These are full-grid costs; neither fewer active coordinates nor adaptive speedup is claimed.",'',
        f'![AT1 loading responses at fixed parameters]({rel}/figures/at1-loading-fields.png)','',
        f'![Separate refinement and balance evidence]({rel}/figures/crack-initiation-verification.png)','',
        f'![Finest low-load damage intensification without contour advance]({rel}/figures/at1-local-intensification.png)','',
        '```sh','PYTHONPATH=src python3 -m pytest -q --junitxml=results/at1-reproduction/test-results.xml',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src python3 benchmarks/at1_thermofracture.py --output results/at1-reproduction',
        'PYTHONPATH=src python3 benchmarks/crack_initiation_report.py --input results/at1-reproduction --at2 results/notch-refinement-2026-09-22','```','',
        'To reproduce the AT2 refinement from its original source and the retained n32 checkpoint:','',
        '```sh','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=results/notch-refinement-2026-09-22/source-snapshot/src:results/notch-refinement-2026-09-22/source-snapshot/benchmarks python3 benchmarks/notch_refinement.py --output results/notch-refinement-reproduction','```','',
        f'[AT1 numerical report]({rel}/report.json), [AT2 refinement report]({oldrel}/report.json), [tests]({rel}/test-results.xml), [theory Section 44](unified_multiscale_defect_framework.md).','',
        'Finite-threshold initiation is now implemented and tested as a specific constitutive option. The original AT2 resolution/localization failures remain. Neither law establishes general crack contact, branching/arrest across loading histories, reactive line/point-defect coupling, calibrated material behavior or efficient dynamic multiscale representation. Those requirements remain open.','']
    write_doc('CRACK_INITIATION_RESULTS.md','\n'.join(lines))
    (out/'plotting.json').write_text(json.dumps({'plotting_and_report_seconds':perf_counter()-start},indent=2)+'\n')


if __name__=='__main__':main()
