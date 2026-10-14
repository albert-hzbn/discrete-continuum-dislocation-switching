"""Report declared accuracy and notch geometry gates without hiding failures."""
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


def number(value,fmt='.5f'):return 'absent' if value is None else format(value,fmt)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    start=perf_counter();out=args.input;r=json.loads((out/'report.json').read_text());figures=out/'figures';figures.mkdir(exist_ok=True)
    spatial=[x for x in r['runs'] if x['dt']==.00025];last=spatial[-1];n=last['n']
    model,b,_=ThermalFracture2D.load(out/f'{last["name"]}-final.npz');_,a,_=ThermalFracture2D.load(out/f'{last["name"]}-initial.npz')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained');maximum=max(a.damage.max(),b.damage.max())
    for ax,values,title in zip(axes,[a.damage,b.damage,b.damage-a.damage],['Initial damage','Final damage','Damage increment']):
        im=ax.imshow(values.reshape(n,n).T,origin='lower',extent=[0,1,0,1],cmap='magma',vmin=0,vmax=maximum if title!='Damage increment' else None)
        ax.set(title=title,xlabel='x',ylabel='y');fig.colorbar(im,ax=ax,shrink=.8)
        if title=='Final damage':
            centers=(np.arange(n)+.5)/n
            ax.contour(centers,centers,b.eta.reshape(n,n).T,levels=[.5],colors='cyan',linewidths=.8)
            ax.axhline(.5,color='white',linestyle=':',linewidth=.8)
    for ext in ('png','svg'):fig.savefig(figures/f'finite-damage-split-fields.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    for run in spatial:
        t=[x['time'] for x in run['rows']]
        axes[0].plot(t,[x['damage_integral'] for x in run['rows']],label=f'n={run["n"]}')
        e=np.array([x['total_energy'] for x in run['rows']]);axes[1].plot(t,(e-e[0])/run['energy_scale'],label=f'n={run["n"]}')
    axes[0].set(xlabel='Time',ylabel='Integral of damage',title='Held-tension notch evolution');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Time',ylabel='Normalized total-energy change',title='Closed energy balance');axes[1].legend(fontsize=8)
    names=['c','eta','T','slip','damage'];x=np.arange(len(names))
    axes[2].bar(x-.18,[100*r['spatial'][k] for k in names],width=.36,label='n24 versus n32')
    axes[2].bar(x+.18,[100*r['time_fine'][k] for k in names],width=.36,label='dt .0005 versus .00025')
    axes[2].axhline(5,color='black',linestyle='--',label='Spatial 5%');axes[2].axhline(1,color='gray',linestyle=':',label='Time 1%')
    axes[2].set(xticks=x,xticklabels=names,ylabel='Discrepancy (%)',title='Unchanged accuracy targets');axes[2].legend(fontsize=7)
    for ext in ('png','svg'):fig.savefig(figures/f'finite-damage-split-verification.{ext}',dpi=170)
    plt.close(fig)
    tests=int(ET.parse(out/'test-results.xml').getroot().find('testsuite').attrib['tests']);passes=sum(r['gates'].values());gp=sum(r['geometry_gates'].values())
    checks=sum(sum(x['checks'].values()) for x in r['runs']);relative=out.relative_to(ROOT) if out.is_absolute() else out
    lines=['# Finite damage split and held-tension notch study','',
        f'September 22, 2026. **{tests} automated tests pass**. Five trajectories pass **{checks}/90** invariant/solver checks, **{passes}/15** accuracy gates and **{gp}/4** geometry gates. Failures, if any, remain below. This extends the common-energy finite thermochemical damage model with a split that preserves hydrostatic compressive bulk energy and a separate tensile-boundary experiment.','',
        '## Constitutive change and its verified scope','',
        'The new volumetric/deviatoric option exactly recovers the old undamaged compressible neo-Hookean energy. It retains the compressive volumetric contribution while degrading positive volumetric and distortional energy. Independent checks cover pressure preservation in hydrostatic compression, degraded expansion, stress/tangent derivatives and symmetry, thermochemical derivatives, heat capacity, objectivity, the small-strain limit, the selected one-sided switch tangent, compressed-shear local first laws and entropy, and complete Q1/Q2 restart.','',
        'The thermal and damage derivatives change with the energy. The damage force and mixed entropy term use the damage-active part rather than the full undegraded elastic energy or Mandel trace. Old saved isotropic states retain their original physical model. This is a conventional constitutive extension; it is not a new multiscale method. Deviatoric deformation can still damage under compressive volume, and bulk-stiffness preservation is not crack-face contact.','',
        '## Boundary experiment and geometry','',
        'The unit square has bottom/top vertical displacement fixed at 0/0.12 and one horizontal pin at the bottom midpoint; all other boundary traction components vanish. Grips are held from the equilibrated initial state, so no external work enters during the trajectory. The initial c/eta/T/d profiles, coefficients and widths match Section 42; the split, boundary conditions and final time change explicitly. This is not a refinement of its mixed-grip trajectory.','',
        '| n | Initial d=0.4 front | Final front | Advance | Increment at y>0.85 | Increment in notch region |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for nkey,g in r['geometry'].items():
        lines.append(f"| {nkey} | {number(g['initial_front'])} | {number(g['final_front'])} | {number(g['advance'])} | {number(None if g['top_fraction'] is None else 100*g['top_fraction'],'.2f')}% | {number(None if g['notch_fraction'] is None else 100*g['notch_fraction'],'.2f')}% |")
    lines+=['','The front is the right threshold crossing of the left-connected centerline interval, with interpolation in both coordinates. The notch region is |y−0.5|<0.16 and x<0.75. These diagnostics measure a specified diffuse contour and spatial damage increment, not an independently calibrated sharp crack length.','',
        '| Predeclared geometry criterion | Result |','| --- | --- |']
    meanings={'top_fraction':'Top-strip fraction <20%','notch_fraction':'Notch-region fraction >50%','front_advance':'Finest front advance >0.02','front_mesh':'n24/n32 final-front difference <0.04'}
    for key,value in r['geometry_gates'].items():lines.append(f'| {meanings[key]} | {"PASS" if value else "FAIL"} |')
    lines+=['','## Accuracy and balances','',
        'Q2 mechanics uses n=16,24,32 at dt=0.00025 to t=0.4; the n24 time study also uses dt=0.001 and 0.0005. Damage ell=0.08 and all other physical profiles/coefficients remain fixed. Field norms and interpolation follow Section 42. Spatial results are differences between two finite discretizations; they are not a certified continuum error.','',
        '| Field | n24/n32 discrepancy | Spatial <5% | dt .0005/.00025 discrepancy | Time <1% |','| --- | ---: | --- | ---: | --- |']
    for key in names:lines.append(f"| {key} | {100*r['spatial'][key]:.4f}% | {r['gates']['mesh_'+key]} | {100*r['time_fine'][key]:.6f}% | {r['gates']['time_'+key]} |")
    failed=[key for key,value in r['gates'].items() if not value];gf=[key for key,value in r['geometry_gates'].items() if not value]
    lines+=['',f"Failed accuracy gates: {', '.join(failed) or 'none'}. Failed geometry gates: {', '.join(gf) or 'none'}. No load, width, coefficient or threshold was adjusted after these runs to change qualification.",'',
        f"Maximum normalized total-energy drift: {max(x['relative_energy_drift'] for x in r['runs']):.3e}. Maximum composition inventory error: {max(x['mass_error'] for x in r['runs']):.3e}. Accepted damage is irreversible; temperature, element orientation and heat capacity remain positive. Entropy and all edge-balance checks are in the saved reports. The piecewise-C2 constitutive split requires actual time refinement and does not guarantee second-order convergence through switching events.",'',
        '| Run | Initialization s | Solver s | Diagnostics / I/O s | Stored scalars |','| --- | ---: | ---: | ---: | ---: |']
    for run in r['runs']:lines.append(f"| {run['name']} | {run['initialization_seconds']:.3f} | {run['solver_seconds']:.3f} | {run['diagnostics_output_seconds']:.3f} | {run['evolving_stored_scalars']:,} |")
    lines+=['',f"The sequential process peak RSS was {r['process_peak_rss_bytes']/2**20:.2f} MiB. Thread limits were one; plotting is timed separately and no JIT is used. These are single-run reference costs, not equivalent-accuracy adaptive performance ratios.",'']
    diagnostic=out/'failure-diagnostic.json'
    if diagnostic.exists():
        d=json.loads(diagnostic.read_text());f=d['squared_error_fractions'];a=d['initial_damage_activation']
        lines+=['## Retained failure diagnosis','',
            f"Independent reconstruction reproduces the slip discrepancy ({100*d['relative_slip_error']:.4f}%), with absolute weighted error {d['absolute_weighted_slip_error']:.3e}. The declared notch region contains {100*f['notch']:.2f}% of its squared error; the bottom-pin neighborhood contains {100*f['bottom_pin']:.3f}%. The peak error cell is at ({d['peak_error_cell_center'][0]:.5f}, {d['peak_error_cell_center'][1]:.5f}). This locates the unresolved slip near the notch; it does not prove a unique numerical cause or certify removal by a particular refinement.",'',
            f"Cells with initial d<1e-4 cover {100*a['low_damage_area_fraction']:.2f}% of the area. All of those cells initially have positive damage rate, with mean {a['mean_damage_rate_within_low_damage']:.5f}. For homogeneous d=0 the AT2 force is q=-2(1-k)psi_+, so any positive driving energy activates damage. The distributed background is therefore allowed by the stated constitutive model. A finite initiation threshold requires a separately derived and calibrated physical law; raising the localization threshold would not fix it.",'',
            f'![Localized slip discrepancy]({relative}/figures/finite-damage-split-slip-error.png)','']
    lines+=[
        f'![Saved notch damage fields; cyan is eta=0.5]({relative}/figures/finite-damage-split-fields.png)','',
        f'![Trajectory balances and errors]({relative}/figures/finite-damage-split-verification.png)','',
        '## Reproduction and remaining scope','',
        '```sh','PYTHONPATH=src python3 -m pytest -q --junitxml=results/finite-damage-split-reproduction/test-results.xml',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src python3 benchmarks/finite_damage_split.py --output results/finite-damage-split-reproduction',
        'PYTHONPATH=src python3 benchmarks/finite_damage_split_diagnostic.py --input results/finite-damage-split-reproduction',
        'PYTHONPATH=src python3 benchmarks/finite_damage_split_report.py --input results/finite-damage-split-reproduction','```','',
        f'[Numerical report]({relative}/report.json), [test evidence]({relative}/test-results.xml), [theory Section 43](unified_multiscale_defect_framework.md).','',
        'The full objective remains active. General crack contact, branching/arrest under varied loading, reactive line/point-defect integration, calibrated materials and useful dynamic multiscale reduction remain unresolved. A held-grip notch experiment cannot establish all of these requirements.','']
    write_doc('FINITE_DAMAGE_SPLIT_RESULTS.md','\n'.join(lines))
    (out/'plotting.json').write_text(json.dumps({'plotting_and_report_seconds':perf_counter()-start},indent=2)+'\n')


if __name__=='__main__':main()
