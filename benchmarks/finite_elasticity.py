"""Section 37 manufactured convergence and reversible finite-loading study."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
from bmvr import _resource as resource
from time import perf_counter
import numpy as np
import scipy
from bmvr.finite_elasticity import FiniteParameters, FiniteElasticity2D, constitutive


def manufactured(x, y):
    pi = np.pi; sx = np.sin(pi*x); cx = np.cos(pi*x)
    sy = np.sin(pi*y); cy = np.cos(pi*y); s2 = np.sin(2*pi*x); c2 = np.cos(2*pi*x)
    u = np.stack((.2*x+.06*sx*sy, -.1*y+.04*s2*sy), axis=-1)
    F = np.broadcast_to(np.eye(3), x.shape+(3, 3)).copy()
    F[..., 0, 0] += .2+.06*pi*cx*sy; F[..., 0, 1] = .06*pi*sx*cy
    F[..., 1, 0] = .08*pi*c2*sy; F[..., 1, 1] += -.1+.04*pi*s2*cy
    H = np.zeros(x.shape+(2, 2, 2))
    H[..., 0, 0, 0] = H[..., 0, 1, 1] = -.06*pi*pi*sx*sy
    H[..., 0, 0, 1] = H[..., 0, 1, 0] = .06*pi*pi*cx*cy
    H[..., 1, 0, 0] = -.16*pi*pi*s2*sy; H[..., 1, 1, 1] = -.04*pi*pi*s2*sy
    H[..., 1, 0, 1] = H[..., 1, 1, 0] = .08*pi*pi*c2*cy
    C = constitutive(F)['tangent'][..., :2, :2, :2, :2]
    body = -np.einsum('...ijkl,...klj->...i', C, H)
    return u, F, body


def storage_bytes(model):
    # Count the composed mesh's actual arrays, including unused fracture geometry data.
    return sum(v.nbytes for obj in (model, model.mesh) for v in vars(obj).values() if isinstance(v, np.ndarray))


def write_json(path, data):
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Unsupported JSON value: {type(value).__name__}')
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=scalar)+'\n')


def manufactured_run(n, output):
    start = perf_counter()
    model = FiniteElasticity2D(n, body=lambda x, y: manufactured(x, y)[2])
    m = model.mesh; fixed = model.boundary(); x, y = m.coordinates.T
    exact, _, _ = manufactured(x, y)
    initial = np.column_stack((.2*x, -.1*y)).ravel()
    setup = perf_counter()-start; start = perf_counter()
    u, state, status = model.equilibrate(initial, fixed, exact.ravel()[fixed])
    solve = perf_counter()-start; start = perf_counter()
    x, y = m.quadrature_points.transpose(2, 0, 1)
    exact, F, _ = manufactured(x, y)
    uq = np.einsum('ga,eai->egi', m.N, u.reshape(-1, 2)[m.cells])
    perturbation = exact-np.stack((.2*x, -.1*y), axis=-1)
    errors = dict(displacement=float(np.linalg.norm(uq-exact)/np.linalg.norm(perturbation)),
                  gradient=float(np.linalg.norm(state['F']-F)/np.linalg.norm(F-np.eye(3))),
                  piola=float(np.linalg.norm(state['local']['P']-constitutive(F)['P'])/np.linalg.norm(constitutive(F)['P'])))
    model.save(output/f'manufactured-{n}.npz', u)
    result = dict(n=n, errors=errors, status=status, min_corner_J=state['min_corner_J'],
                  initialization_seconds=setup, solver_seconds=solve, diagnostics_output_seconds=perf_counter()-start,
                  persistent_model_bytes=storage_bytes(model), displacement_bytes=u.nbytes,
                  unknowns=len(u), checks=dict(converged=status['relative_residual'] <= 1e-9,
                      admissible=state['min_corner_J'] > 0,
                      energy_search=bool(np.all(np.diff(status['potential']) <= 1e-12))))
    write_json(output/f'manufactured-{n}.json', result)
    print(f'manufactured n={n} errors={errors}', flush=True)
    return result


def heterogeneous(n, increments, variant, output):
    start = perf_counter()
    def G(x, y): return np.exp(-((x-.45)**2+(y-.5)**2)/(2*.15**2))
    def Fp(x, y):
        value = np.broadcast_to(np.eye(3), x.shape+(3, 3)).copy()
        value[..., 0, 1] = .08*np.exp(-((x-.6)**2+(y-.5)**2)/(2*.18**2))
        return value
    p = FiniteParameters(alpha_c=0, alpha_eta=0, alpha_T=0) if variant == 'no-growth' else FiniteParameters()
    model = FiniteElasticity2D(n, parameters=p, c=lambda x, y: .2+.1*G(x, y),
        eta=lambda x, y: .4*G(x, y), T=lambda x, y: 1+.2*G(x, y),
        Fp=None if variant == 'no-plastic-map' else Fp)
    m = model.mesh; x, y = m.coordinates.T; fixed = model.boundary()
    lifting = np.column_stack((y*np.sin(np.pi*x), np.zeros_like(x))).ravel()
    u = np.zeros_like(lifting); previous = 0.; rows = []; initial = None
    loads = np.r_[np.linspace(0, .25, increments+1), np.linspace(.25, 0, increments+1)[1:]]
    setup = perf_counter()-start; solver = 0.; diagnostics = 0.; max_search_increase = 0.
    label = f'{variant}-{n}-{increments}'
    for step, load in enumerate(loads):
        start = perf_counter()
        u, state, status = model.equilibrate(u+(load-previous)*lifting, fixed, load*lifting[fixed])
        solver += perf_counter()-start; start = perf_counter()
        if initial is None: initial = u.copy()
        if len(status['potential']) > 1:
            max_search_increase = max(max_search_increase, max(np.diff(status['potential'])))
        rows.append(dict(load=float(load), energy=state['energy'], elastic_energy=state['elastic_energy'],
            reaction=float(state['force'][fixed]@lifting[fixed]), iterations=status['iterations'],
            residual=status['relative_residual'], min_J=state['min_corner_J'],
            min_capacity=float(np.min(state['local']['heat_capacity'])),
            max_cauchy_asymmetry=float(np.max(abs(state['local']['sigma']-state['local']['sigma'].swapaxes(-1, -2))))))
        if step == increments: model.save(output/f'{label}-peak.npz', u, float(load), step)
        previous = load; diagnostics += perf_counter()-start
    start = perf_counter(); model.save(output/f'{label}-final.npz', u, float(load), step)
    reaction = np.array([row['reaction'] for row in rows]); energy = np.array([row['energy'] for row in rows])
    work = np.r_[0., np.cumsum((reaction[:-1]+reaction[1:])*np.diff(loads)/2)]
    increment = energy[increments]-energy[0]
    work_error = float(abs(work[increments]-increment)/abs(increment))
    return_error = float(np.max(abs(u-initial)))
    checks = dict(converged=max(r['residual'] for r in rows) <= 1e-9,
        admissible=min(r['min_J'] for r in rows) > 0,
        heat_capacity=min(r['min_capacity'] for r in rows) > 0,
        symmetric_cauchy=max(r['max_cauchy_asymmetry'] for r in rows) < 1e-12,
        energy_search=max_search_increase <= 1e-12, reversible_state=return_error < 1e-8,
        cycle_work=abs(work[-1])/abs(increment) < 1e-7)
    result = dict(n=n, increments=increments, variant=variant, parameters=asdict(p), rows=rows,
        work=work.tolist(), ramp_work_relative_error=work_error, return_max_displacement_error=return_error,
        initialization_seconds=setup, solver_seconds=solver,
        diagnostics_output_seconds=diagnostics+perf_counter()-start,
        persistent_model_bytes=storage_bytes(model), displacement_bytes=u.nbytes, unknowns=len(u), checks=checks)
    write_json(output/f'{label}.json', result)
    print(f'{label} reaction={rows[increments]["reaction"]:.8g} work_error={work_error:.4g} checks={all(checks.values())}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output; output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    manufactured_results = [manufactured_run(n, output) for n in (8, 16, 32, 64)]
    runs = [heterogeneous(n, 10, 'baseline', output) for n in (12, 24, 48)]
    runs += [heterogeneous(24, 20, 'baseline', output)]
    runs += [heterogeneous(24, 10, v, output) for v in ('no-growth', 'no-plastic-map')]
    coarse, fine = runs[1:3]
    reaction_difference = abs(coarse['rows'][10]['reaction']/fine['rows'][10]['reaction']-1)
    increment = lambda run: run['rows'][run['increments']]['elastic_energy']-run['rows'][0]['elastic_energy']
    energy_difference = abs(increment(coarse)/increment(fine)-1)
    gates = dict(finest_displacement=manufactured_results[-1]['errors']['displacement'] < .01,
        finest_gradient=manufactured_results[-1]['errors']['gradient'] < .02,
        decreasing_displacement=all(a['errors']['displacement'] > b['errors']['displacement'] for a,b in zip(manufactured_results, manufactured_results[1:])),
        decreasing_gradient=all(a['errors']['gradient'] > b['errors']['gradient'] for a,b in zip(manufactured_results, manufactured_results[1:])),
        mesh_reaction=reaction_difference < .02, mesh_energy=energy_difference < .02,
        half_step_work=runs[3]['ramp_work_relative_error'] < .005,
        decreasing_work_error=runs[3]['ramp_work_relative_error'] < runs[1]['ramp_work_relative_error'])
    result = dict(scope='Spatial finite elasticity with frozen prescribed internal fields; no thermal or plastic evolution, acceleration or novelty claim.',
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            platform=platform.platform(), machine=platform.machine(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
        manufactured=manufactured_results, loading=runs, mesh_reaction_relative_difference=reaction_difference,
        mesh_energy_relative_difference=energy_difference, gates=gates,
        all_numerical_checks=all(all(r['checks'].values()) for r in manufactured_results+runs),
        all_accuracy_gates=all(gates.values()),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024),
        study_seconds=perf_counter()-start)
    write_json(output/'report.json', result)
    print(json.dumps(dict(gates=gates, seconds=result['study_seconds'])), flush=True)


if __name__ == '__main__':
    main()
