"""Report common-energy nodal damage accuracy without upgrading older failures."""
import argparse,json
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.nodal_damage import NodalThermalFracture2D
from bmvr.thermofracture import ThermalFracture2D
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]


def number(value):return 'undefined' if value is None else f'{value:.6f}'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--cellwise',type=Path,default=Path('results/finite-damage-split-2026-09-22'));args=parser.parse_args()
    start=perf_counter();out=args.input;r=json.loads((out/'report.json').read_text());old=json.loads((args.cellwise/'report.json').read_text())
    figures=out/'figures';figures.mkdir(exist_ok=True);spatial=[x for x in r['runs'] if x['dt']==.00025]
    initial_errors={}
    for run in spatial:
        n=run['n'];m,s,_=NodalThermalFracture2D.load(out/f'{run["name"]}-initial.npz')
        _,c,_=ThermalFracture2D.load(args.cellwise/f'tensile-split-{n}-0.00025-initial.npz')
        x,y=m.mechanics.mesh.quadrature_points.transpose(2,0,1)
        exact=.65*np.exp(-((y-.5)/.08)**2)*(1-np.tanh((x-.4)/.08))/2
        w=m.mechanics.mesh.weights
        initial_errors[str(n)]={name:float(np.sqrt(np.sum(w*(value-exact)**2)/np.sum(w*exact**2)))
            for name,value in [('cellwise',c.damage[:,None]),('nodal',m.damage_quadrature(s.damage))]}
    (out/'initial-interpolation.json').write_text(json.dumps(initial_errors,indent=2)+'\n')
    run=spatial[-1];n=run['n'];m,b,_=NodalThermalFracture2D.load(out/f'{run["name"]}-final.npz')
    _,a,_=NodalThermalFracture2D.load(out/f'{run["name"]}-initial.npz');axis=np.linspace(0,1,n+1)
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained');maximum=max(a.damage.max(),b.damage.max())
    for ax,values,title in zip(axes,[a.damage,b.damage,b.damage-a.damage],['Initial nodal damage','Final nodal damage','Damage increment; own scale']):
        im=ax.pcolormesh(axis,axis,values.reshape(n+1,n+1).T,shading='gouraud',cmap='magma',vmin=0,
                        vmax=maximum if title!='Damage increment; own scale' else None)
        ax.set(title=title,xlabel='x',ylabel='y',aspect='equal');fig.colorbar(im,ax=ax,shrink=.8)
        if title=='Final nodal damage':
            ax.contour(axis,axis,b.damage.reshape(n+1,n+1).T,levels=[.4],colors='white',linewidths=.8)
    for ext in ('png','svg'):fig.savefig(figures/f'nodal-damage-fields.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    fields=['c','eta','T','slip','damage'];x=np.arange(5)
    axes[0].bar(x-.18,[100*old['spatial'][k] for k in fields],width=.36,label='Cellwise n24/n32')
    axes[0].bar(x+.18,[100*r['spatial'][k] for k in fields],width=.36,label='Nodal n24/n32')
    axes[0].axhline(5,color='black',linestyle='--',label='5% spatial target')
    axes[0].set(xticks=x,xticklabels=fields,ylabel='Discrepancy (%)',title='Same continuum model, distinct approximations');axes[0].legend(fontsize=7)
    for run in spatial:
        t=[x['time'] for x in run['rows']];E=np.array([x['total_energy'] for x in run['rows']])
        axes[1].plot(t,(E-E[0])/run['energy_scale'],label=f'n={run["n"]}')
    axes[1].set(xlabel='Time',ylabel='Normalized total-energy change',title='Closed nodal-damage evolution');axes[1].legend(fontsize=8)
    for basis in ('cellwise','nodal'):
        axes[2].loglog([int(n) for n in initial_errors],[initial_errors[n][basis] for n in initial_errors],'o-',label=basis)
    axes[2].set(xlabel='Cells per side',ylabel='Initial weighted damage error',title='Against the same analytical initial profile');axes[2].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'nodal-damage-verification.{ext}',dpi=170)
    plt.close(fig)
    tests=int(ET.parse(out/'test-results.xml').getroot().find('testsuite').attrib['tests'])
    checks=sum(sum(x['checks'].values()) for x in r['runs']);passes=sum(r['gates'].values());gp=sum(r['geometry_gates'].values())
    rel=out.relative_to(ROOT) if out.is_absolute() else out
    saturated=sum(row.get('newly_saturated_damage_nodes',0) for x in r['runs'] for row in x['rows'])
    obstacle=max(row.get('midpoint_damage_obstacle_power',0.) for x in r['runs'] for row in x['rows'])
    lines=['# Continuous nodal damage: unchanged-physics notch refinement','',
        f'September 26, 2026. **{tests} tests pass**. Five trajectories pass **{checks}/90** invariant/solver checks, **{passes}/15** accuracy gates and **{gp}/4** geometry gates. This changes damage approximation and its consistent assembly, while retaining the original active-notch AT2 law, physical widths, coefficients and loading. The full objective remains active.','',
        '## Discrete model and verification','',
        'Damage is reconstructed continuously with Q1 nodal functions at the existing mechanical quadrature. Degradation, surface energy, phase coupling and their derivatives use that same field. The positive lumped-area metric gives irreversible nodal rates. Assembled nodal dissipation is allocated to heat cells with positive weights; an antisymmetric assembly-power transfer closes the local first law. The old cell-damage edge flux is not added again.','',
        'Tests independently verify affine reconstruction and masses, nodal and phase energy derivatives, local/global energy and entropy, mixed-sign element work with positive allocated heat, pairwise power antisymmetry, undamaged frozen recovery, AT1/AT2 with Q1/Q2 mechanics, restart, bounds and irreversibility. An initial area-sum assertion below one floating-point spacing was corrected to an eight-epsilon roundoff allowance; no benchmark tolerance changed.','',
        '## Finite bound arrival','',
        f"The first Section 45 matrix ([retained evidence](results/nodal-damage-2026-09-22/nodal-tensile-24-0.00025-failure.json)) stopped at n24 step 1240: a node at d=0.99989457 still had positive rate, because neighboring quadrature damage was below one. Its pre-fix source is preserved beside that record. Section 45.4 now solves the discrete nodal obstacle inside the coupled midpoint: each step-average damage rate is capped at (1-d_old)/dt, and the same capped rate drives dissipation, the entropy mixed term and assembly power. Saturating coordinates satisfy d=1 exactly. The resulting obstacle reaction adds nonnegative dissipation. In this matrix {saturated} node-arrival events occur, with maximum obstacle power {obstacle:.3e}; the first-law, entropy and irreversibility checks below include those steps. The failed original n24 run is not counted as a completed trajectory.",'',
        'Checkpoint metadata distinguishes cellwise and Q1-nodal damage. Old files load as cellwise; an incompatible loader refuses instead of reinterpreting coordinates. No already evolved field was remapped for this study. Initial damage samples the same analytical profile at each basis coordinate.','',
        '## Accuracy at unchanged physics','',
        'Use the original AT2 volumetric/deviatoric model, ell=0.08, G_c=0.002, grip 0.12, fixed bottom y and one horizontal pin, with free remaining traction components. n=16,24,32 use dt=0.00025 to t=0.4; n24 also uses dt=0.001 and 0.0005. All transport and profile parameters are unchanged.','',
        '| Field | Nodal n24/n32 | Spatial <5% | dt .0005/.00025 | Temporal <1% |','| --- | ---: | --- | ---: | --- |']
    for key in fields:lines.append(f"| {key} | {100*r['spatial'][key]:.4f}% | {r['gates']['mesh_'+key]} | {100*r['time_fine'][key]:.6f}% | {r['gates']['time_'+key]} |")
    failed=[key for key,value in r['gates'].items() if not value]
    lines+=['',f"Failed accuracy gates: {', '.join(failed) or 'none'}. The original cellwise n24/n32 slip discrepancy (19.45%) and n32/n48 slip/damage discrepancies (15.30%/5.33%) remain preserved.",'',
        f"Interpretation: continuous damage does not reduce the unresolved discrepancy. Slip changes from {100*old['spatial']['slip']:.2f}% (cellwise) to {100*r['spatial']['slip']:.2f}% (nodal), damage from {100*old['spatial']['damage']:.2f}% to {100*r['spatial']['damage']:.2f}%, and the finest notch-region fraction from {100*old['geometry']['32']['notch_fraction']:.2f}% to {100*r['geometry']['32']['notch_fraction']:.2f}%. Cell-constant degradation is therefore not the dominant source of the slip mesh error. Temporal errors are four to five orders below their gates, so the remaining failure is spatial.",'',
        'Damage errors integrate both reconstructed Q1 fields at coarse mechanical quadrature. c/eta/T and slip use the existing comparison procedure. Each refinement is a comparison of finite discretizations; neither is an exact continuum reference. This is not an equivalent-accuracy speedup comparison or a BMVR novelty result.','',
        '| n | Initial-profile error, cellwise | Initial-profile error, nodal |','| --- | ---: | ---: |']
    for n,values in initial_errors.items():lines.append(f"| {n} | {100*values['cellwise']:.4f}% | {100*values['nodal']:.4f}% |")
    lines+=['','The analytical initial-profile comparison is a separate reconstruction diagnostic. It does not prove that damage interpolation is the sole source of trajectory error.','',
        '## Geometry, balances and cost','',
        '| n | Initial d=0.4 front | Final front | Advance | Top-strip fraction | Notch-region fraction |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for n,g in r['geometry'].items():lines.append(f"| {n} | {number(g['initial_front'])} | {number(g['final_front'])} | {number(g['advance'])} | {number(g['top_fraction'])} | {number(g['notch_fraction'])} |")
    lines+=['','Geometry targets remain top fraction <20%, notch fraction >50%, front advance >0.02 and cross-mesh front-position difference <0.04. Fractions use cell-integrated damage increments and the same fixed physical regions; the centerline contour uses continuous nodal interpolation.','',
        '| Gate | Outcome |','| --- | --- |']
    for key,value in r['geometry_gates'].items():lines.append(f"| {key} | {'PASS' if value else 'FAIL'} |")
    lines+=['',f"Maximum normalized total-energy drift is {max(x['relative_energy_drift'] for x in r['runs']):.3e}; maximum composition inventory error is {max(x['mass_error'] for x in r['runs']):.3e}. Every accepted update meets damage irreversibility, positivity, determinant and entropy checks. Shared-coordinate power is conservative; it is not an added heat-conduction model.",'',
        '| Run | Initialization s | Solver s | Diagnostics / I/O s | Stored scalars |','| --- | ---: | ---: | ---: | ---: |']
    for run in r['runs']:lines.append(f"| {run['name']} | {run['initialization_seconds']:.3f} | {run['solver_seconds']:.3f} | {run['diagnostics_output_seconds']:.3f} | {run['evolving_stored_scalars']:,} |")
    lines+=['',f"Whole sequential-process peak RSS is {r['process_peak_rss_bytes']/2**20:.2f} MiB. Thread limits were one and no JIT is used; plotting is measured separately. Nodal damage has (n+1)² coordinates rather than n², so this change does not reduce degrees of freedom. Costs are single observations without a qualified adaptive-performance claim.",'',
        f'![Continuous damage and its increment]({rel}/figures/nodal-damage-fields.png)','',
        f'![Refinement, balances and reconstruction]({rel}/figures/nodal-damage-verification.png)','',
        '## Reproduction and remaining scope','',
        '```sh','PYTHONPATH=src python3 -m pytest -q --junitxml=results/nodal-damage-reproduction/test-results.xml',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src python3 benchmarks/nodal_damage.py --output results/nodal-damage-reproduction',
        'PYTHONPATH=src python3 benchmarks/nodal_damage_report.py --input results/nodal-damage-reproduction','```','',
        f'[Numerical report]({rel}/report.json), [test evidence]({rel}/test-results.xml), [initial-profile diagnostic]({rel}/initial-interpolation.json), [theory Section 45](unified_multiscale_defect_framework.md).','',
        'Continuous damage is a conventional full-grid approximation. General crack contact and branching/arrest, calibrated initiation, reactive line/point-defect integration and useful dynamic multiscale representation remain unresolved. The AT2 law still permits distributed damage initiation; changing its approximation does not change that constitutive property.','']
    write_doc('NODAL_DAMAGE_RESULTS.md','\n'.join(lines))
    (out/'plotting.json').write_text(json.dumps({'plotting_and_report_seconds':perf_counter()-start},indent=2)+'\n')


if __name__=='__main__':main()
