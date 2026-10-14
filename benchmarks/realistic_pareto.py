"""Section 95: accuracy against cost on the copper cell structure of Section 94; gates P1-P3.

Particle dislocation dynamics with superdislocations (k dislocations of one sign lumped into one of Burgers vector kb and
mobility m_d/k), the continuum method, and the allocator at several thresholds, all on the Section 94 microstructure.

    python benchmarks/realistic_pareto.py     # requires results/realistic-copper-2026-09-30 (Section 94)
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import time

import numpy as np

import realistic_copper as rc
from bmvr.representation_allocator import GlideEnsemble, Run

OUT = rc.ROOT / 'results' / 'realistic-pareto-2026-09-30'
KS, NCS, SEED = (2, 4, 8, 16, 32, 64), (16, 64, 256), 95


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads((rc.OUT / 'report.json').read_text())
    ref = np.load(rc.OUT / 'fields.npz')
    gD, interior = ref['gamma_D'], ref['interior']
    x, y, s, ids, n_wall, n_int, wall_y = rc.microstructure(np.random.default_rng(rc.SEED))
    b_nd = rc.B_VEC/rc.L
    ens = GlideEnsemble(rc.G, b=b_nd, M=1., tau=rc.TAU/rc.MU, nu=rc.NU, block=rc.BLOCK, n_c=64)
    sig0 = ens.stress(b_nd*ens.deposit(x, y, s.astype(float)))
    dt = .25*ens.h/(b_nd*(np.abs(sig0).max() + rc.TAU/rc.MU))
    assert abs(dt*rc.T0 - base['numerics']['dt_s']) < 1e-20, 'time step differs from Section 94'
    nb = rc.G//rc.BLOCK
    blk = lambda g: g.reshape(nb, rc.BLOCK, nb, rc.BLOCK).mean(axis=(1, 3))
    rel = lambda a, b: float(np.linalg.norm(a - b)/np.linalg.norm(b))
    err = lambda g: dict(interior=rel(g[:, interior], gD[:, interior]), block=rel(blk(g), blk(gD)))
    rows = [dict(method='D', k=1, cost=base['cpu_per_step_s']['D'], particles=int(x.size), interior=0., block=0.)]
    rng = np.random.default_rng(SEED)
    for k in KS:
        keep = np.zeros(x.size, bool)
        for sign in (1, -1):
            idx = np.flatnonzero(s == sign); rng.shuffle(idx); keep[idx[::k]] = True
        e = GlideEnsemble(rc.G, b=k*b_nd, M=1./k, tau=rc.TAU/rc.MU, nu=rc.NU, block=rc.BLOCK, n_c=64)
        r = Run(e, x[keep], y[keep], s[keep], ids[keep], 'D')
        t0 = time.process_time()
        for _ in range(rc.STEPS):
            r.step(dt)
        c = (time.process_time() - t0)/rc.STEPS
        rows.append(dict(method='D', k=k, cost=c, particles=int(keep.sum()), **err(r.gamma)))
        print('D k', k, f'{c*1e3:.2f} ms', rows[-1]['interior'], flush=True)
    rows.append(dict(method='C', cost=base['cpu_per_step_s']['C'], particles=0, interior=base['errors_vs_D']['C']['interior'],
                     block=base['errors_vs_D']['C']['block']))
    for nc in NCS:
        if nc == 64:
            rows.append(dict(method='H', n_c=64, cost=base['cpu_per_step_s']['H'], particles=base['final']['H_particles'],
                             interior=base['errors_vs_D']['H']['interior'], block=base['errors_vs_D']['H']['block']))
            continue
        e = GlideEnsemble(rc.G, b=b_nd, M=1., tau=rc.TAU/rc.MU, nu=rc.NU, block=rc.BLOCK, n_c=nc)
        r = Run(e, x, y, s, ids, 'H')
        t0 = time.process_time()
        for _ in range(rc.STEPS):
            r.step(dt)
        c = (time.process_time() - t0)/rc.STEPS
        rows.append(dict(method='H', n_c=nc, cost=c, particles=len(r.x), **err(r.gamma)))
        print('H n_c', nc, f'{c*1e3:.2f} ms', rows[-1]['interior'], flush=True)
    H = next(r for r in rows if r['method'] == 'H' and r['n_c'] == 64)
    Ds = [r for r in rows if r['method'] == 'D' and r['k'] > 1]
    near = min(Ds, key=lambda r: abs(np.log(r['cost']/H['cost'])))
    reach = [r for r in rows if r['method'] == 'D' and r['interior'] <= H['interior']]
    cheapest = min(reach, key=lambda r: r['cost'])
    C = next(r for r in rows if r['method'] == 'C')
    gates = dict(P1_equal_cost_D=near['interior'] >= 5*H['interior'], P2_C=C['interior'] >= 5*H['interior'],
                 P3_D_to_match=cheapest['cost'] >= 10*H['cost'])
    rep = dict(section=95, seed=SEED, rows=rows, equal_cost_D=near, cheapest_D_matching_H=cheapest, gates=gates,
               environment=rc.environment())
    (OUT/'report.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')
    print(json.dumps(dict(gates=gates, near=near, cheapest=cheapest), indent=1, default=float), flush=True)


if __name__ == '__main__':
    main()
