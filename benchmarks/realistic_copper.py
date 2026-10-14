"""Section 94: a dislocation cell structure in deformed copper, in physical units; gates R1-R5.

Runs the discrete (D), continuum (C) and allocator (H) frameworks of Section 91 on the same microstructure and times a
direct-summation discrete dislocation dynamics (DDD) kernel as a further reference.

    python benchmarks/realistic_copper.py            # writes results/realistic-copper-2026-09-30/
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from bmvr.representation_allocator import GlideEnsemble, Run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'realistic-copper-2026-09-30'

# Copper (Section 69 calibration) and loading, SI units
MU, NU, B_VEC, DRAG, TAU = 48e9, 0.324, 0.2556e-9, 50e-6, 30e6
L = 50e-6                                  # periodic cell side (m)
WALLS, WALL_SPACING, WALL_THICK = 10, 5e-6, 0.5e-6
RHO_WALL, RHO_INT, WALL_POS_FRAC = 2e15, 1e12, 0.52
G, BLOCK, N_C, STEPS, SEED = 256, 4, 64, int(os.environ.get('STEPS', '1000')), 94
T0 = DRAG/MU                               # time unit (s): nondimensional mobility = 1


def microstructure(rng):
    """Edge dislocations (positions in units of L, signs) of the cell structure."""
    centres = (np.arange(WALLS) + .5)*WALL_SPACING
    n_wall_each = int(round(RHO_WALL*WALL_THICK*L))
    xw, yw = [], []
    for yc in centres:
        xw.append(rng.random(n_wall_each)*L)
        yw.append(yc + (rng.random(n_wall_each) - .5)*WALL_THICK)
    xw, yw = np.concatenate(xw), np.concatenate(yw)
    sw = np.where(rng.random(xw.size) < WALL_POS_FRAC, 1, -1)
    area_int = L*L - WALLS*WALL_THICK*L
    n_int = int(round(RHO_INT*area_int))
    xi, yi = [], []
    while sum(len(v) for v in xi) < n_int:            # rejection sampling outside the walls
        x, y = rng.random(4*n_int)*L, rng.random(4*n_int)*L
        dist = np.min(np.abs(y[:, None] - centres[None, :]), axis=1)
        keep = dist > WALL_THICK/2
        xi.append(x[keep]); yi.append(y[keep])
    xi, yi = np.concatenate(xi)[:n_int], np.concatenate(yi)[:n_int]
    si = np.where(rng.random(n_int) < .5, 1, -1)
    x = np.r_[xw, xi]/L; y = np.r_[yw, yi]/L; s = np.r_[sw, si]
    ids = np.r_[-np.ones(xw.size, int), np.arange(n_int)]
    return x % 1, y % 1, s, ids, xw.size, n_int, centres/L


def direct_sum_cost(rng, sizes=(1000, 2000, 4000), repeats=3):
    """CPU seconds per step of a vectorized O(N^2) edge-dislocation stress sum; fitted as a*N^2."""
    pref = MU*B_VEC/(2*np.pi*(1 - NU))
    out = {}
    for n in sizes:
        x, y, s = rng.random(n)*L, rng.random(n)*L, np.where(rng.random(n) < .5, 1., -1.)
        best = np.inf
        for _ in range(repeats):
            t0 = time.process_time()
            dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]
            r2 = dx*dx + dy*dy; np.fill_diagonal(r2, np.inf)
            sig = (pref*s[None, :]*dx*(dx*dx - dy*dy)/(r2*r2)).sum(axis=1)
            _ = B_VEC*(sig + TAU)*s/DRAG                   # velocities
            best = min(best, time.process_time() - t0)
        out[str(n)] = best
    n = np.array(sizes, float); c = np.array([out[str(k)] for k in sizes])
    a = float(np.sum(c*n**2)/np.sum(n**4))
    return out, a


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
                threads='one numerical thread', timer='time.process_time')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x, y, s, ids, n_wall, n_int, wall_y = microstructure(rng)
    N = x.size
    b_nd = B_VEC/L
    ens = GlideEnsemble(G, b=b_nd, M=1., tau=TAU/MU, nu=NU, block=BLOCK, n_c=N_C)
    sig0 = ens.stress(b_nd*ens.deposit(x, y, s.astype(float)))
    v_max = b_nd*(np.abs(sig0).max() + TAU/MU)            # nondimensional (units L/t0)
    dt = .25*ens.h/v_max
    runs = {}
    for method in ('D', 'C', 'H'):
        r = Run(ens, x, y, s, ids, method)
        t0 = time.process_time()
        for _ in range(STEPS):
            r.step(dt)
        cpu = time.process_time() - t0
        runs[method] = dict(run=r, cpu_per_step=cpu/STEPS)
        print(method, f'{cpu/STEPS*1e3:.2f} ms/step', 'particles', len(r.x), 'continuum blocks', int(r.cont.sum()), flush=True)
    # errors (plastic strain per cell), blocks and interior region
    yc = (np.arange(G) + .5)/G
    interior = np.min(np.abs(yc[:, None] - wall_y[None, :]), axis=1) > 1e-6/L    # > 1 um from a wall centre
    nb = G//BLOCK; blk = lambda g: g.reshape(nb, BLOCK, nb, BLOCK).mean(axis=(1, 3))
    rel = lambda a, b: float(np.linalg.norm(a - b)/np.linalg.norm(b))
    D = runs['D']['run'].gamma
    err = {m: dict(block=rel(blk(runs[m]['run'].gamma), blk(D)),
                   interior=rel(runs[m]['run'].gamma[:, interior], D[:, interior]),
                   mean=abs(runs[m]['run'].gamma.mean() - D.mean())/abs(D.mean())) for m in ('C', 'H')}
    H = runs['H']['run']
    ledger = H.ledger
    conv = max([max(l['count_error'], l['burgers_error']) for l in ledger], default=0.)/N
    t_phys = STEPS*dt*T0                                  # simulated physical time (s)
    cost = {m: runs[m]['cpu_per_step'] for m in runs}
    ds, a = direct_sum_cost(np.random.default_rng(SEED + 1))
    cost['DDD_direct_extrapolated'] = a*N**2
    # mean plastic shear strain: gamma field is (Burgers content x glide distance) per area, in units of b/L*L = b
    mean_strain = {m: float(runs[m]['run'].gamma.mean()) for m in runs}
    rep = dict(section=94, material=dict(mu_Pa=MU, nu=NU, b_m=B_VEC, drag_Pa_s=DRAG, tau_Pa=TAU),
               microstructure=dict(L_m=L, walls=WALLS, wall_spacing_m=WALL_SPACING, wall_thickness_m=WALL_THICK,
                                   rho_wall_m2=RHO_WALL, rho_interior_m2=RHO_INT, n_wall=int(n_wall), n_interior=int(n_int),
                                   n_total=int(N), wall_positive_fraction=WALL_POS_FRAC, seed=SEED),
               numerics=dict(grid=G, h_m=L/G, smoothing_m=1.5*L/G, block=BLOCK, n_c=N_C, steps=STEPS, dt_s=dt*T0,
                             t0_s=T0, simulated_time_s=t_phys, v_max_m_s=v_max*L/T0,
                             initial_max_internal_stress_Pa=float(np.abs(sig0).max()*MU)),
               cpu_per_step_s=cost, direct_sum_samples_s=ds, direct_sum_coefficient_s=a,
               simulated_time_per_cpu_hour_s={m: t_phys/(cost[m]*STEPS)*3600 for m in cost},
               errors_vs_D=err, mean_gamma=mean_strain, conversions=dict(events=len(ledger), max_relative_error=conv),
               final=dict(H_particles=len(H.x), H_continuum_blocks=int(H.cont.sum())), environment=environment())
    rep['gates'] = dict(R1_conservation=conv <= 1e-12,
                        R2_accuracy=err['H']['block'] <= .01 and err['H']['interior'] <= .02 and err['H']['mean'] <= .01,
                        R3_cost=cost['H'] <= cost['D']/5,
                        R4_H_better_than_C=err['H']['interior'] < err['C']['interior'])
    np.savez_compressed(OUT/'fields.npz', gamma_D=D, gamma_C=runs['C']['run'].gamma, gamma_H=H.gamma, cont=H.cont,
                        interior=interior)
    (OUT/'report.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')
    print(json.dumps(dict(gates=rep['gates'], errors=err, cost=cost, t_phys=t_phys), indent=1, default=float), flush=True)


if __name__ == '__main__':
    main()
