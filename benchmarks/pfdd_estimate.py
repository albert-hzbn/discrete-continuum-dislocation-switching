"""Section 96: cost of one phase-field dislocation dynamics (PFDD) step, extrapolated to core resolution for the copper cell.

    python benchmarks/pfdd_estimate.py     # writes results/pfdd-estimate-2026-09-30/report.json
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import time

import numpy as np

import realistic_copper as rc
from bmvr.extended_defects import wavevectors

OUT = rc.ROOT / 'results' / 'pfdd-estimate-2026-09-30'
SIZES = (512, 1024, 2048, 4096)


def pfdd_step_time(n, repeats=3):
    """Best CPU time of one explicit PFDD step on an n x n grid (unit cell, dimensionless)."""
    rng = np.random.default_rng(96)
    phi = rng.random((n, n))                      # slip in units of b
    KX, KY = wavevectors(n, 1.)
    k2 = KX**2 + KY**2; k4 = k2**2; k4[0, 0] = np.inf
    kernel = KX*KY*(2/(1 - rc.NU))*(-1j*KY)/k4    # Airy kernel of the Section 91 solver (stress per unit slip density)
    W, kappa, L, dt, tau = 1e-3, 1e-4, 1., 1e-2, 6e-4
    best = np.inf
    for _ in range(repeats):
        t0 = time.process_time()
        sig = np.real(np.fft.ifft2(np.fft.fft2(phi)*kernel))          # resolved shear stress
        lap = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0) + np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - 4*phi)*n*n
        dE = W*np.pi*np.sin(2*np.pi*phi) - kappa*lap - (sig + tau)  # crystalline + gradient - Peach-Koehler driving
        phi = phi - dt*L*dE
        best = min(best, time.process_time() - t0)
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = {n: pfdd_step_time(n) for n in SIZES}
    for n in SIZES:
        print(n, f'{t[n]:.4f} s', flush=True)
    x = np.array([n*n*np.log2(n*n) for n in SIZES]); y = np.array([t[n] for n in SIZES])
    a = float(np.sum(x*y)/np.sum(x*x))
    est = {}
    for label, h in (('h=b', rc.B_VEC), ('h=2b', 2*rc.B_VEC)):
        n = rc.L/h; cells = n*n
        est[label] = dict(grid_side=n, cells=cells, cpu_per_step_s=a*cells*np.log2(cells),
                          memory_bytes_5_fields=5*8*cells)
    rep = dict(section=96, sizes=list(SIZES), cpu_per_step_s={str(n): t[n] for n in SIZES}, fit_coefficient_s=a,
               estimate=est, environment=rc.environment())
    (OUT/'report.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')
    print(json.dumps(est, indent=1, default=float))


if __name__ == '__main__':
    main()
