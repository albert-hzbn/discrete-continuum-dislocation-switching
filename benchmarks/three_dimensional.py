"""Fresh-process 3D coupled verification and fixed-width convergence studies."""

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import numpy as np
import scipy
from scipy.sparse import csr_matrix
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bmvr import _resource as resource
from bmvr.continuum3d import Cube, CoupledCrystal, Parameters3D
from bmvr.adaptivity import overlap_matrix


def run(n, dt, end, output, frozen=False):
    wall_start = perf_counter()
    grid = Cube(16., n)
    x, y, z = grid.coordinates()
    phase = np.tanh((4. - np.sqrt(x*x + y*y + z*z)) / np.sqrt(2))
    population = .2 + .025 * np.exp(-((x-3)**2+(y-2)**2+(z-1)**2)/8)
    slip = np.zeros((12,) + grid.shape)
    params = Parameters3D(yield_stress=10. if frozen else .015)
    mean = np.array([[0, .06, 0], [.06, 0, 0], [0, 0, 0]])
    model = CoupledCrystal(grid, params, mean_strain=mean)
    initial_fields = (phase.copy(), population.copy(), slip.copy())
    fields = (phase, population, slip)
    current = model.evaluate(*fields)
    initial_inventory = grid.integrate(population)
    initial_energy = current['energy']
    initialization = perf_counter() - wall_start
    times, energy, components, inventories, plastic, equilibrium = [], [], [], [], [], []
    component_names = list(current['components'])
    solver, diagnostics, t, steps, trials, minimum_dt = 0., 0., 0., 0, 0, dt

    def record():
        times.append(t)
        energy.append(current['energy'])
        components.append(list(current['components'].values()))
        inventories.append(grid.integrate(fields[1]))
        plastic.append(float(np.sqrt(np.mean(fields[2] ** 2))))
        equilibrium.append(current['equilibrium_rms'])

    start = perf_counter()
    record()
    diagnostics += perf_counter() - start
    while t < end - 1e-12:
        start = perf_counter()
        fields, current, used, attempts = model.step(fields, min(dt, end-t), current)
        solver += perf_counter() - start
        t += used
        steps += 1
        trials += attempts
        minimum_dt = min(minimum_dt, used)
        start = perf_counter()
        record()
        diagnostics += perf_counter() - start
    start = perf_counter()
    pstrain = np.einsum('a...,aij->ij...', fields[2], model.schmid)
    trace_error = float(np.max(np.abs(np.einsum('ii...->...', pstrain))))
    drift = float(np.max(np.abs(np.array(inventories) - initial_inventory)))
    maximum_increase = float(max(0., np.max(np.diff(energy))))
    maximum_equilibrium = float(max(equilibrium))
    plastic_spatial_std = float(np.sqrt(np.mean(np.var(fields[2], axis=(1, 2, 3)))))
    metrics = {'inventory_drift': drift, 'max_energy_increase': maximum_increase,
               'equilibrium_rms_max': maximum_equilibrium, 'plastic_trace_error': trace_error,
               'initial_energy': initial_energy, 'final_energy': energy[-1],
               'slip_rms': plastic[-1], 'slip_spatial_std': plastic_spatial_std,
               'steps': steps, 'trial_evaluations': trials, 'minimum_dt': minimum_dt,
               'final_population_range': [float(fields[1].min()), float(fields[1].max())],
               'mean_stress': current['stress'].mean(axis=(2, 3, 4)).tolist(),
               'field_dofs': int(sum(f.size for f in fields)),
               'persistent_primary_field_bytes': int(sum(f.nbytes for f in fields))}
    checks = {'inventory': drift <= 1e-10 * max(1, initial_inventory),
              'energy': maximum_increase <= 1e-10 * max(1, abs(initial_energy)),
              'equilibrium': maximum_equilibrium <= 1e-10,
              'isochoric_slip': trace_error <= 1e-12,
              'population_bounds': bool(fields[1].min() > 0 and fields[1].max() < 1),
              'slip_response': bool(plastic[-1] == 0 if frozen else plastic[-1] > 1e-5 and plastic_spatial_std > 1e-6)}
    diagnostics += perf_counter() - start
    experiment = perf_counter() - wall_start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {'model': '3D periodic small-strain FCC crystal slip with phase and population fields',
              'grid': {'length': grid.length, 'n': n, 'h': grid.h, 'profile_scale': float(np.sqrt(2))},
              'parameters': asdict(params), 'mean_strain': mean.tolist(), 'dt_requested': dt, 'end': end,
              'frozen_slip': frozen, 'metrics': metrics, 'checks': checks,
              'peak_rss_before_io_bytes': int(peak if sys.platform == 'darwin' else peak * 1024),
              'timings': {'initialization_s': initialization, 'solver_s': solver, 'diagnostics_s': diagnostics,
                          'adaptation_s': 0., 'compilation_s': 0., 'plotting_s': 0.,
                          'experiment_s': experiment, 'unclassified_s': max(0., experiment-initialization-solver-diagnostics)},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'platform': platform.platform(), 'threads': 1},
              'limits': ['Uniform field-grid reference; no adaptive speedup is claimed.',
                         'Small strain, fixed orientation, homogeneous isotropic elasticity.',
                         'No explicit dislocation lines, topology reactions, or temperature evolution.']}
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    np.savez_compressed(output / 'fields.npz', phase=fields[0], population=fields[1], slip=fields[2],
                        initial_phase=initial_fields[0], initial_population=initial_fields[1],
                        stress=current['stress'], strain=current['strain'], plastic_strain=pstrain,
                        time=np.array(times), energy=np.array(energy), components=np.array(components),
                        component_names=np.array(component_names), inventory=np.array(inventories),
                        slip_rms=np.array(plastic), equilibrium=np.array(equilibrium))
    report['timings']['array_io_s'] = perf_counter() - start
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(output.name, sum(checks.values()), '/', len(checks), 'checks;', steps, 'steps;', round(solver, 3), 'solver s', flush=True)
    if not all(checks.values()):
        raise RuntimeError('A declared 3D numerical check failed; inspect the saved report.')


def volume_project(field, old_n, new_n, length):
    """Separable cell-overlap projection, including leading slip/tensor dimensions."""
    old = np.linspace(-length/2, length/2, old_n+1)
    new = np.linspace(-length/2, length/2, new_n+1)
    matrix = csr_matrix(overlap_matrix(old, new))
    for axis in (-3, -2, -1):
        moved = np.moveaxis(field, axis, 0)
        out = matrix @ moved.reshape(old_n, -1)
        field = np.moveaxis(out.reshape((new_n,) + moved.shape[1:]), 0, axis)
    return field


def compare(folder, coarse, reference):
    a = json.loads((folder/coarse/'report.json').read_text())
    b = json.loads((folder/reference/'report.json').read_text())
    errors = {}
    with np.load(folder/coarse/'fields.npz') as data, np.load(folder/reference/'fields.npz') as ref:
        for name, background, scale in [('phase', -1., .5), ('population', .2, 1.), ('slip', 0., 1.), ('stress', 0., 1.)]:
            exact = volume_project(ref[name], b['grid']['n'], a['grid']['n'], a['grid']['length'])
            delta = scale * (data[name] - exact)
            norm = np.sqrt(np.mean((scale * (exact-background)) ** 2))
            errors[name] = {'l1': float(np.mean(np.abs(delta))), 'l2': float(np.sqrt(np.mean(delta**2))),
                            'linf': float(np.max(np.abs(delta))), 'relative_l2': float(np.sqrt(np.mean(delta**2))/max(norm, 1e-30))}
    return {'run': coarse, 'reference': reference, 'h': a['grid']['h'], 'dt': a['dt_requested'],
            'solver_s': a['timings']['solver_s'], 'errors': errors,
            'relative_energy_error': abs(a['metrics']['final_energy']-b['metrics']['final_energy'])/abs(b['metrics']['final_energy'])}


def collect(folder):
    space = [compare(folder, f'space-{n}', 'space-49') for n in (17, 25, 33)]
    time = [compare(folder, f'time-{dt}', 'time-0.0025') for dt in (.04, .02, .01, .005)]
    orders = {}
    for label, rows, axis in [('space', space, 'h'), ('time', time, 'dt')]:
        orders[label] = {key: [float(np.log(a['errors'][key]['l2']/b['errors'][key]['l2']) / np.log(a[axis]/b[axis]))
                               for a,b in zip(rows[:-1],rows[1:])] for key in rows[0]['errors']}
    summary = {'space': space, 'time': time, 'observed_orders': orders,
               'monotone_error_reduction': {label: {key: all(a['errors'][key]['l2'] > b['errors'][key]['l2'] for a,b in zip(rows[:-1],rows[1:]))
                                                    for key in rows[0]['errors']} for label,rows in [('space',space),('time',time)]},
               'notes': ['Fixed physical width and domain. Spatial runs use identical dt=0.005.',
                         'Spatial comparison volume-projects reference cell values. Reference errors and initialization quadrature can inflate observed order.',
                         'Time convergence uses one n=25 grid, with dt=0.0025 reference.',
                         'No asymptotic order or adaptive speedup is inferred from these small studies.']}
    (folder/'convergence.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('monotone errors:', summary['monotone_error_reduction'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'results/three-dimensional-2026-09-12')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--n', type=int, default=25)
    parser.add_argument('--dt', type=float, default=.01)
    parser.add_argument('--end', type=float, default=.4)
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    if args.dt <= 0 or args.end <= 0:
        parser.error('dt and end must be positive')
    if args.worker:
        run(args.n, args.dt, args.end, args.output, args.frozen)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.collect:
        runs = [(f'space-{n}',n,.005,False) for n in (17,25,33,49)]
        runs += [(f'time-{dt}',25,dt,False) for dt in (.04,.02,.01,.005,.0025)]
        runs += [('frozen-slip',25,.005,True)]
        for name,n,dt,frozen in runs:
            command = [sys.executable,__file__,'--worker','--n',str(n),'--dt',str(dt),'--end',str(args.end),'--output',str(args.output/name)]
            if frozen:
                command.append('--frozen')
            subprocess.run(command, check=True, cwd=ROOT)
    collect(args.output)


if __name__ == '__main__':
    main()
