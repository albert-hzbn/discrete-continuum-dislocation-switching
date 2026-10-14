"""Render saved Section 37 evidence without rerunning equilibria."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bmvr.finite_elasticity import FiniteElasticity2D


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args(); output = args.input; report = json.loads((output/'report.json').read_text())
    start = perf_counter(); figures = output/'figures'; figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    manufactured = report['manufactured']; n = np.array([r['n'] for r in manufactured])
    for key, label in (('displacement', 'Displacement / non-affine part'), ('gradient', 'Gradient / F − I'), ('piola', 'Piola stress')):
        axes[0].loglog(n, [r['errors'][key] for r in manufactured], 'o-', label=label)
    axes[0].set(xlabel='Elements per side', ylabel='Relative L2 error', title='Manufactured finite deformation')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    runs = report['loading']
    for run in runs[:3]:
        axes[1].plot([r['load'] for r in run['rows']], [r['reaction'] for r in run['rows']], '.-', label=f'n = {run["n"]}')
    axes[1].set(xlabel='Prescribed shear amplitude', ylabel='Conjugate grip reaction', title='Loading and reversible unloading')
    axes[1].legend(); axes[1].grid(alpha=.2)
    for run in (runs[1], runs[3]):
        steps = run['increments']+1; energy = np.array([r['energy'] for r in run['rows']]); work = np.array(run['work'])
        axes[2].plot([r['load'] for r in run['rows'][:steps]], (work-energy+energy[0])[:steps], '.-', label=f'{steps-1} increments')
    axes[2].set(xlabel='Prescribed shear amplitude', ylabel='Trapezoid work − free-energy change', title='Loading-work refinement, n = 24')
    axes[2].legend(); axes[2].grid(alpha=.2)
    for suffix in ('png', 'svg'): fig.savefig(figures/f'finite-verification.{suffix}', dpi=170)
    plt.close(fig)
    model, u, clock = FiniteElasticity2D.load(output/'baseline-48-10-peak.npz'); state = model.evaluate(u)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    n = model.mesh.n; extent = (0, 1, 0, 1)
    values = (np.linalg.norm(u.reshape(n+1, n+1, 2), axis=-1).T,
              state['local']['J'].mean(axis=1).reshape(n, n).T,
              state['local']['sigma'][..., 0, 1].mean(axis=1).reshape(n, n).T)
    for ax, value, title in zip(axes, values, ('Displacement magnitude (nodes)', 'Mean element J', 'Mean element Cauchy shear')):
        im = ax.imshow(value, origin='lower', extent=extent, cmap='viridis', interpolation='nearest')
        ax.set(xlabel='Reference x', ylabel='Reference y', title=title); fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(f'Prescribed heterogeneous fields, n = 48, load = {clock["load"]}; frozen Fp and temperature')
    for suffix in ('png', 'svg'): fig.savefig(figures/f'finite-fields.{suffix}', dpi=170)
    plt.close(fig)
    (output/'plotting.json').write_text(json.dumps({'plotting_seconds':perf_counter()-start, 'source':'report.json and baseline-48-10-peak.npz'}, indent=2)+'\n')


if __name__ == '__main__': main()
