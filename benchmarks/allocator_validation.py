"""Section 93: validation of the discrete/continuum allocator (gates V1-V6).

Each configuration runs as an independent job that saves its plastic strain (restricted to the 256^2 cells) and its
CPU time; `analyze` evaluates the declared gates from the saved runs.

    python benchmarks/allocator_validation.py run --method D --grid 1024 --tag ref-D1024
    python benchmarks/allocator_validation.py analyze
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from bmvr.representation_allocator import GlideEnsemble, Run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'allocator-validation-2026-09-30'
G0, TAU, N_SPARSE, STEPS = 256, .5, 200, 1000


def ensemble(seed=91, n_dense=500000):
    """Section 91 ensemble with a variable dense count; identical to Section 91 for n_dense = 500 000."""
    rng = np.random.default_rng(seed)
    xd = (.5 + .08*rng.standard_normal(n_dense)) % 1; yd = (.5 + .08*rng.standard_normal(n_dense)) % 1
    sd = np.where(rng.random(n_dense) < .6, 1, -1)
    ys = np.where(rng.random(N_SPARSE) < .5, rng.uniform(0, .2, N_SPARSE), rng.uniform(.8, 1., N_SPARSE))
    xs = rng.random(N_SPARSE); ss = np.where(rng.random(N_SPARSE) < .5, 1, -1)
    return np.r_[xd, xs], np.r_[yd, ys], np.r_[sd, ss], np.r_[-np.ones(n_dense, int), np.arange(N_SPARSE)]


def calibration(x, y, s):
    """Section 91 rule: b0 so that the initial internal stress is at most half the applied stress (256^2 probe)."""
    probe = GlideEnsemble(G0, b=1., tau=TAU)
    b0 = .5*TAU/float(np.abs(probe.stress(probe.deposit(x, y, s.astype(float)))).max())
    M = 1/(b0*TAU); dt = .25*(1/G0)/(M*b0*TAU)
    return b0, M, dt


def environment():
    env = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
               platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
               threads='OPENBLAS/OMP/MKL/VECLIB_NUM_THREADS=1', timer='time.process_time')
    try:
        import psutil
        env['cpu_percent_at_start'] = psutil.cpu_percent(interval=.5)
    except ImportError:
        env['cpu_percent_at_start'] = None
    return env


class Timers:
    """Wrap the ensemble's operators and the allocation step to split time by component (V4).

    Components use perf_counter (the Windows process_time tick is 15.6 ms); decomposition runs execute alone."""

    def __init__(self, e, run):
        self.t = dict(stress=0., particle_grid=0., transport=0., allocation=0.)
        for name, key in (('stress', 'stress'), ('deposit', 'particle_grid'), ('gather', 'particle_grid'), ('transport', 'transport')):
            setattr(e, name, self._wrap(getattr(e, name), key))
        run.allocate = self._wrap(run.allocate, 'allocation')

    def _wrap(self, f, key):
        def g(*a, **k):
            t0 = time.perf_counter(); r = f(*a, **k); self.t[key] += time.perf_counter() - t0; return r
        return g


def run_job(a):
    OUT.joinpath('runs').mkdir(parents=True, exist_ok=True)
    stem = OUT/'runs'/a.tag
    if stem.with_suffix('.json').exists() and not a.force:
        sys.exit(f'{stem}.json exists; pass --force to replace it')
    x, y, s, ids = ensemble(a.seed, a.ndense); b0, M, dt = calibration(x, y, s)
    f = a.grid//G0
    e = GlideEnsemble(a.grid, b=b0, M=M, tau=TAU, width=1.5*f, block=16*f, n_c=a.nc)
    env = environment(); t_init = time.process_time()
    r = Run(e, x, y, s, ids, a.method); t_init = time.process_time() - t_init
    timers = Timers(e, r) if a.decompose else None
    steps = STEPS*f*a.time_refine; step = dt/(f*a.time_refine)
    t0 = time.process_time(); w0 = time.perf_counter()
    for _ in range(steps):
        r.step(step)
    cpu = time.process_time() - t0; wall = time.perf_counter() - w0
    gamma = r.gamma.reshape(G0, f, G0, f).mean(axis=(1, 3)) if f > 1 else r.gamma
    np.savez_compressed(stem.with_suffix('.npz'), gamma=gamma)
    ledger = [l for l in r.ledger]
    rep = dict(tag=a.tag, method=a.method, grid=a.grid, seed=a.seed, n_dense=a.ndense, n_c=a.nc, steps=steps, dt=step,
               time_refine=a.time_refine, b0=b0, M=M, tau=TAU, cpu_seconds=cpu, wall_seconds=wall, init_cpu_seconds=t_init,
               cpu_per_base_step=cpu/STEPS, wall_per_base_step=wall/STEPS, final_particles=len(r.x),
               final_continuum_blocks=int(r.cont.sum()), conversion_events=len(ledger),
               max_conversion_error=max([max(l['count_error'], l['burgers_error']) for l in ledger], default=0.)/(a.ndense + N_SPARSE),
               components=({k: v/STEPS for k, v in timers.t.items()} if timers else None), environment=env)
    stem.with_suffix('.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: rep[k] for k in ('tag', 'cpu_per_base_step', 'final_particles')}), flush=True)


# ---------------------------------------------------------------- analysis
def load(tag):
    j = json.loads((OUT/'runs'/f'{tag}.json').read_text()); j['gamma'] = np.load(OUT/'runs'/f'{tag}.npz')['gamma']; return j


SPARSE = None


def errors(g, ref):
    global SPARSE
    if SPARSE is None:
        yc = (np.arange(G0) + .5)/G0; SPARSE = (yc < .2) | (yc > .8)
    B = 16; nb = G0//B; blk = lambda q: q.reshape(nb, B, nb, B).mean(axis=(1, 3))
    rel = lambda p, q: float(np.linalg.norm(p - q)/np.linalg.norm(q))
    return dict(all=rel(g, ref), block=rel(blk(g), blk(ref)), sparse=rel(g[:, SPARSE], ref[:, SPARSE]))


def analyze(_):
    rep = dict(environment=environment())
    # V1, V2
    ref = {t: load(t) for t in ('ref-D256', 'ref-D512', 'ref-D1024', 'ref-C256', 'ref-C512', 'ref-H64', 'ref-H256')}
    star = ref['ref-D1024']['gamma']
    v1 = errors(ref['ref-D512']['gamma'], star)
    v2 = {t[4:]: errors(ref[t]['gamma'], star) for t in ref if t != 'ref-D1024'}
    rep['V1'] = dict(D512_vs_Dstar=v1, gate=v1['all'] <= .01 and v1['sparse'] <= .02)
    rep['V2'] = dict(errors_vs_Dstar=v2,
                     gate_a=v2['H64']['sparse'] <= 1.25*v2['D256']['sparse'],
                     gate_b=v2['H64']['sparse'] < v2['C256']['sparse'],
                     H256_gate_a=v2['H256']['sparse'] <= 1.25*v2['D256']['sparse'])
    rep['V2']['gate'] = rep['V2']['gate_a'] and rep['V2']['gate_b']
    # V3
    ns = (5000, 20000, 50000, 100000, 200000, 500000, 1000000); rows = []
    for n in ns:
        d, c, h = (load(f'scale-{m}-{n}') for m in 'DCH')
        rows.append(dict(n_dense=n, cost_D=d['cpu_per_base_step'], cost_C=c['cpu_per_base_step'], cost_H=h['cpu_per_base_step'],
                         H_particles=h['final_particles'], H_blocks=h['final_continuum_blocks'],
                         H_err=errors(h['gamma'], d['gamma']), C_err=errors(c['gamma'], d['gamma'])))
    N = np.array(ns, float); cD = np.array([r['cost_D'] for r in rows]); cH = np.array([r['cost_H'] for r in rows])
    A = np.c_[np.ones_like(N), N]; coef, *_ = np.linalg.lstsq(A, cD, rcond=None)
    r2 = 1 - np.sum((A@coef - cD)**2)/np.sum((cD - cD.mean())**2)
    cheaper = [r['n_dense'] for r in rows if r['cost_H'] < r['cost_D']]
    rep['V3'] = dict(rows=rows, D_fit=dict(intercept=coef[0], per_dislocation=coef[1], r2=r2),
                     gate_a=r2 >= .98, gate_b=cH.max()/cH.min() <= 2, gate_c=bool(np.all(cH <= 1.1*cD)),
                     gate_d=all(r['cost_H'] <= r['cost_D']/3 for r in rows if r['n_dense'] >= 100000),
                     gate_e=all(r['H_err']['sparse'] <= .01 for r in rows),
                     first_N_H_cheaper=min(cheaper) if cheaper else None)
    rep['V3']['gate'] = all(rep['V3'][k] for k in ('gate_a', 'gate_b', 'gate_c', 'gate_d', 'gate_e'))
    # V4
    dD, dH = load('decomp-D'), load('decomp-H')
    def split(j):
        c = dict(j['components']); c['rest'] = j['wall_per_base_step'] - sum(c.values()); c['total'] = j['wall_per_base_step']; return c
    sD, sH = split(dD), split(dH)
    def proj(k):
        num = sD['stress'] + sD['rest'] + sD['transport'] + sD['allocation'] + sD['particle_grid']/k
        den = sH['stress'] + sH['rest'] + sH['transport'] + sH['allocation'] + sH['particle_grid']/k
        return num/den
    rep['V4'] = dict(D=sD, H=sH, measured_speedup=sD['total']/sH['total'],
                     projected_speedup={str(k): proj(k) for k in (1, 10, 100)},
                     note='Projection is a model: it divides only the particle-grid time by k and keeps the rest, including '
                          'the NumPy particle push in "rest", unchanged.')
    # V5
    seeds = (91, 92, 93, 94, 95); srows = []
    for sd in seeds:
        d, c, h = (load(f'seed-{m}-{sd}') for m in 'DCH')
        srows.append(dict(seed=sd, H=errors(h['gamma'], d['gamma']), C=errors(c['gamma'], d['gamma'])))
    Hs = np.array([r['H']['sparse'] for r in srows]); Cs = np.array([r['C']['sparse'] for r in srows])
    rep['V5'] = dict(rows=srows, H_sparse_mean=Hs.mean(), H_sparse_std=Hs.std(ddof=1), C_sparse_mean=Cs.mean(), C_sparse_std=Cs.std(ddof=1),
                     gate=bool(np.all(Hs < Cs)) and all(r['H']['block'] <= .005 for r in srows))
    # V6: seed 91 at N = 500 000 is the Section 91.2 configuration
    old = json.loads((ROOT/'results'/'representation-allocator-2026-09-28'/'post-hoc-1000'/'report.json').read_text())
    now = srows[0]; new = dict(H_sparse=now['H']['sparse'], C_sparse=now['C']['sparse'], H_block=now['H']['block'], C_block=now['C']['block'])
    saved = dict(H_sparse=old['sparse_cell_strain_error']['H'], C_sparse=old['sparse_cell_strain_error']['C'],
                 H_block=old['block_strain_error']['H'], C_block=old['block_strain_error']['C'])
    reld = {k: abs(new[k] - saved[k])/saved[k] for k in saved}
    rep['V6'] = dict(saved=saved, reproduced=new, relative_difference=reld, gate=all(v <= .02 for v in reld.values()))
    OUT.joinpath('report.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')
    print(json.dumps({k: rep[k].get('gate') for k in ('V1', 'V2', 'V3', 'V5', 'V6')}), flush=True)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--method', choices='DCH', required=True); r.add_argument('--grid', type=int, default=G0)
    r.add_argument('--time-refine', type=int, default=1, help='extra time refinement beyond the grid factor')
    r.add_argument('--seed', type=int, default=91); r.add_argument('--ndense', type=int, default=500000)
    r.add_argument('--nc', type=int, default=64); r.add_argument('--tag', required=True)
    r.add_argument('--decompose', action='store_true'); r.add_argument('--force', action='store_true')
    sub.add_parser('analyze')
    a = p.parse_args()
    run_job(a) if a.cmd == 'run' else analyze(a)


if __name__ == '__main__':
    main()
