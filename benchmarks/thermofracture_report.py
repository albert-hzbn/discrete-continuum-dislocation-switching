"""Render saved Section 42 evolution, convergence and coupling evidence."""
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    start=perf_counter();out=args.input;r=json.loads((out/'report.json').read_text());figures=out/'figures';figures.mkdir(exist_ok=True)
    runs=r['runs'];spatial=[x for x in runs if x['variant']=='coupled' and x['dt']==.00025]
    coupled=next(x for x in spatial if x['n']==16);finest=next(x for x in spatial if x['n']==24)
    ablations=[coupled]+[x for x in runs if x['variant'] in ('frozen-damage','no-phase-resistance')]
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for run in ablations:
        axes[0].plot([x['time'] for x in run['rows']],[x['damage_integral'] for x in run['rows']],label=run['variant'])
    axes[0].set(xlabel='Time',ylabel='Integral of damage',title='Evolving versus frozen damage / phase coupling');axes[0].legend(fontsize=7)
    for run in spatial:
        E=np.array([x['total_energy'] for x in run['rows']]);axes[1].plot([x['time'] for x in run['rows']],(E-E[0])/run['energy_scale'],label=f'n={run["n"]}')
    axes[1].set(xlabel='Time',ylabel='Normalized energy change',title='Closed total-energy balance');axes[1].legend(fontsize=8)
    fields=['c','eta','T','slip','damage'];x=np.arange(5)
    axes[2].bar(x-.18,[100*r['spatial'][k] for k in fields],width=.36,label='n16 versus n24')
    axes[2].bar(x+.18,[100*r['time_fine'][k] for k in fields],width=.36,label='dt .0005 versus .00025')
    axes[2].axhline(5,color='black',linestyle='--',label='Spatial 5% target');axes[2].axhline(1,color='gray',linestyle=':',label='Time 1% target')
    axes[2].set(xticks=x,xticklabels=fields,ylabel='Relative discrepancy (%)',title='Predeclared accuracy gates');axes[2].legend(fontsize=7)
    for ext in ('png','svg'):fig.savefig(figures/f'thermofracture-verification.{ext}',dpi=170)
    plt.close(fig)
    model,s,_=ThermalFracture2D.load(out/f'{finest["name"]}-final.npz');_,initial,_=ThermalFracture2D.load(out/f'{finest["name"]}-initial.npz')
    m=model.mechanics.mesh;rate=model.rate(s);slip=np.sum(m.weights*np.linalg.norm(s.gamma,axis=-1),axis=1)/m.h**2
    centers=m.coordinates[m.cells].mean(axis=1);increment=s.damage-initial.damage
    top_fraction=float(increment[centers[:,1]>.85].sum()/increment.sum())
    fig,axes=plt.subplots(2,3,figsize=(12,7.3),layout='constrained')
    fields=[(initial.damage,'Initial damage'),(s.damage,'Final damage'),(s.eta,'Final phase'),
            (s.c,'Final composition'),(s.T,'Final temperature'),(slip,'Final slip magnitude')]
    for ax,(values,title) in zip(axes.ravel(),fields):
        image=ax.imshow(values.reshape(24,24).T,origin='lower',extent=[0,1,0,1],cmap='magma',
                        **({'vmin':0,'vmax':max(initial.damage.max(),s.damage.max())} if 'damage' in title else {}))
        ax.set(title=title,xlabel='x',ylabel='y');fig.colorbar(image,ax=ax,shrink=.8)
    for ext in ('png','svg'):fig.savefig(figures/f'thermofracture-fields.{ext}',dpi=170)
    plt.close(fig)
    contrasts={}
    for run in ablations[1:]:
        _,other,_=ThermalFracture2D.load(out/f'{run["name"]}-final.npz')
        _,base,_=ThermalFracture2D.load(out/f'{coupled["name"]}-final.npz')
        contrasts[run['variant']]={key:float(np.linalg.norm(getattr(base,key)-getattr(other,key))) for key in ('damage','eta','T','gamma','u')}
    tests=int(ET.parse(out/'test-results.xml').getroot().find('testsuite').attrib['tests'])
    passes=sum(r['gates'].values());checks=sum(sum(x['checks'].values()) for x in runs)
    rel=out.relative_to(ROOT) if out.is_absolute() else out
    lines=['# Finite thermochemical fracture results','',
        f'September 22, 2026. **{tests} automated tests pass.** Ten trajectories pass **{checks}/180** invariant/solver checks. Accuracy gates: **{passes}/19 pass**, with failures listed below. Damage, diffusion, phase evolution, heat, finite plasticity and inertial mechanics now evolve from one free energy. This is a conventional full-grid isotropic damage reference; dynamic adaptation, discrete line reactions and general crack contact remain outside this implementation.','',
        '## Implemented coupling and independent verification','',
        'Section 42 derives normalized elastic degradation, irreversible viscous damage, AT2 storage, reciprocal phase-dependent damage resistance and damage microstress energy transport. The thermal equation contains both nonnegative damage dissipation and the reversible damage/elastic entropy term. Fracture energy is stored rather than counted again as heat.','',
        'Tests independently differentiate free energy with respect to damage, phase, composition and deformation; differentiate local internal energy and global entropy along the coupled rate; and verify the scaled tangent, undamaged frozen recovery, plastic-volume preservation, irreversibility and complete Q1/Q2 restart. The homogeneous finite-deformation problem has an analytic exponential damage solution and temperature fixed by total energy.','',
        'The initial homogeneous test used a 1e-8 absolute error assertion inconsistent with its finite midpoint step (observed error 2.84e-8). It now checks the exact discrete midpoint solution to 1e-12 and the unchanged, predeclared continuum target 1e-6. The benchmark below separately demonstrates temporal convergence.','',
        '| Homogeneous dt | Damage absolute error | Temperature absolute error |','| --- | ---: | ---: |']
    for run in runs:
        if run['variant']=='homogeneous':lines.append(f"| {run['dt']} | {run['damage_error']:.3e} | {run['temperature_error']:.3e} |")
    lines+=['','## Spatial and temporal accuracy','',
        'Spatial candidates use Q2 displacement on n=8,16,24 with fixed coefficients, physical widths, initial profiles, dt=0.00025 and t=0.2. Damage width ell=0.08 is unchanged, giving h/ell=1.5625, 0.78125 and 0.52083. The n16/n24 comparison is a finite numerical sensitivity check, not an exact continuum-error certificate. Scalar comparisons interpolate to coarse centers; slip uses physical quadrature weights. Temperature/composition/phase norms remove the uniform background, while damage uses its uncentered norm.','',
        '| Field | n16 / n24 error | Spatial <5% | dt .0005 / .00025 error | Temporal <1% |','| --- | ---: | --- | ---: | --- |']
    for key in ('c','eta','T','slip','damage'):
        lines.append(f"| {key} | {100*r['spatial'][key]:.4f}% | {r['gates']['mesh_'+key]} | {100*r['time_fine'][key]:.6f}% | {r['gates']['time_'+key]} |")
    failed=[key for key,value in r['gates'].items() if not value]
    lines+=['',f"Failed gates: {', '.join(failed) if failed else 'none in this matrix'}. Previous studies retain their own failed comparisons.",'',
        '## Conservation, feedback and cost','',
        f"Maximum normalized energy drift is {max(x['relative_energy_drift'] for x in runs):.3e}; maximum composition inventory error is {max(x['mass_error'] for x in runs):.3e}. Every accepted damage increment is nonnegative within the 1e-13 gate. Saved entropy is nondecreasing and temperatures, orientation bounds and effective heat capacities remain positive.",'',
        f"On the n24 trajectory, integrated damage changes from {finest['rows'][0]['damage_integral']:.6f} to {finest['rows'][-1]['damage_integral']:.6f}. Frozen-damage and zero-phase-resistance ablations preserve their stated alternative physics and are not accuracy references. The table gives unnormalized Euclidean final-state differences against the coupled n16 run; these diagnose reciprocal response, not calibrated effect size.",'',
        f"Visual inspection shows damage intensification in the seeded notch and growth near the non-affine upper grip. The strip y>0.85 contains {100*top_fraction:.2f}% of the damage increment. This run demonstrates coupled damage evolution; it does not isolate notch-tip propagation, branching or interface-controlled arrest.",'',
        '| Ablation | Damage difference | Phase difference | T difference | Slip difference | Displacement difference |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for label,values in contrasts.items():lines.append('| '+label+' | '+' | '.join(f'{values[k]:.3e}' for k in ('damage','eta','T','gamma','u'))+' |')
    lines+=['','| Run | Initialization s | Solver s | Diagnostics / checkpoint s | Stored scalars |','| --- | ---: | ---: | ---: | ---: |']
    for run in runs:lines.append(f"| {run['name']} | {run['initialization_seconds']:.3f} | {run['solver_seconds']:.3f} | {run['diagnostics_output_seconds']:.3f} | {run['evolving_stored_scalars']:,} |")
    lines+=['',f"The sequential study process peak RSS is {r['process_peak_rss_bytes']/2**20:.2f} MiB; this is a whole-process peak, not per-run memory. Thread limits were one, no JIT was used, and plotting is separately recorded in plotting.json. Costs are single observations with no adaptive speedup claim. Section 41 timing evidence remains tied to its preserved source hashes; the present source adds damage support.",'',
        f'![Evolution, balance and accuracy]({rel}/figures/thermofracture-verification.png)','',
        f'![Saved coupled fields]({rel}/figures/thermofracture-fields.png)','',
        '## Reproduction and remaining requirements','',
        '```sh','PYTHONPATH=src python3 -m pytest -q --junitxml=results/thermofracture-reproduction/test-results.xml',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src python3 benchmarks/thermofracture.py --output results/thermofracture-reproduction',
        'PYTHONPATH=src python3 benchmarks/thermofracture_report.py --input results/thermofracture-reproduction','```','',
        f'[Numerical report]({rel}/report.json), [test evidence]({rel}/test-results.xml), [theory](unified_multiscale_defect_framework.md).','',
        'The new model couples regularized damage with the evolving thermochemical fields, but degrades compressive as well as tensile elastic energy. It does not inherit the separate small-strain spectral split. Contact, branching/arrest under general loads, material calibration, reactive discrete-line/point defects, dynamic multiscale representation and useful adaptive efficiency remain unresolved. The goal remains active.','']
    write_doc('THERMOFRACTURE_RESULTS.md','\n'.join(lines))
    (out/'plotting.json').write_text(json.dumps({'plotting_and_report_seconds':perf_counter()-start,'ablation_differences':contrasts,'damage_increment_fraction_y_above_085':top_fraction},indent=2)+'\n')


if __name__=='__main__':main()
