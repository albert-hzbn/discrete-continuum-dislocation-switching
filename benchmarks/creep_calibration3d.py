"""Section 98 v2 calibration: single-cavity growth rate of the GB-plane closure against Raj and Ashby (1975).

The rate is measured over [t1, t1 + 0.1] t_u with t1 = 1 t_u, after the source transient (relaxation time l_s^2).
python benchmarks/creep_calibration3d.py
"""
from dataclasses import replace
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from bmvr.creep3d import Copper773, Creep3D, Params3D  # noqa: E402

OUT = Path(__file__).resolve().parents[1]/'results'/'creep-cavitation-3d-2026-09-30'


def ratio(l_s, R0, S, k_ex=10., t1=1.):
    P = replace(Params3D(n=96), count=1, r_range=(R0, R0), l_s=l_s, k_ex=k_ex)
    s = Creep3D(Copper773(), S, 'C', P)
    s.mechanics(); s.plane_steps(int(round(t1/P.dt)))
    V0, t0 = 4/3*np.pi*s.R[0]**3, s.t
    s.plane_steps(int(round(.1/P.dt)))
    rate = (4/3*np.pi*s.R[0]**3 - V0)/(s.t - t0)
    R = .5*(s.R[0] + (3*V0/(4*np.pi))**(1/3)); f = (2*R)**2
    q = -2*np.log(f) - (1 - f)*(3 - f)
    return float(rate/(2*np.pi*(s.S - 2*s.gamma/R)/q)), float(R)


if __name__ == '__main__':
    cases = [(0.03, 30.), (0.04, 30.), (0.08, 30.), (0.04, 60.), (0.04, 20.)]
    scan = {}
    for l_s in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        scan[l_s] = [ratio(l_s, R0, S) for R0, S in cases]
        print(l_s, [(round(r, 3), round(R, 4)) for r, R in scan[l_s]], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'calibration_scan.json').write_text(json.dumps(dict(cases=cases, scan={str(k): v for k, v in scan.items()}), indent=1))
