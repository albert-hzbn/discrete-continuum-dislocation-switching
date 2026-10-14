"""Independent spatial and temporal refinements at fixed physical interface width."""

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import json
from time import perf_counter
import numpy as np
from bmvr.grid import Grid, circle
from bmvr.energy import DoubleWell
from bmvr.solvers import ac_step, ch_step
from bmvr.adaptivity import transfer
from bmvr.diagnostics import radius


def solve(kind, n, dt, end=2.0):
    length = 64. if kind == "ac" else 32.
    grid = Grid.uniform(length, n, periodic=kind == "ch")
    if kind == "ac":
        v = circle(grid, 12.)
    else:
        v = .05 * (np.cos(4 * np.pi * grid.X / length) + np.cos(6 * np.pi * grid.Y / length)
                    + .5 * np.sin(2 * np.pi * (grid.X + grid.Y) / length))
    model = DoubleWell()
    symbol = grid.spectrum() if kind == "ch" else None
    steps = int(np.ceil(end / dt))
    dt = end / steps
    start = perf_counter()
    for _ in range(steps):
        v = ac_step(grid, v, model, dt) if kind == "ac" else ch_step(grid, v, dt, symbol=symbol)
    elapsed = perf_counter() - start
    return grid, v, {"n": n, "h": length / n, "dt": dt, "steps": steps, "end_time": end,
                     "solver_s": elapsed, "radius": radius(grid, v) if kind == "ac" else None}


def compare(solution, reference):
    g, v, row = solution
    rg, rv, _ = reference
    # Compare cell averages on the coarse grid; do not compare arbitrary samples.
    ref_on_grid = transfer(rg, g, rv)
    row["relative_l2"] = float(np.sqrt(g.integrate((v - ref_on_grid) ** 2) / g.integrate(ref_on_grid ** 2)))
    if row["radius"] is not None:
        row["radius_error"] = abs(row["radius"] - radius(rg, rv))
    return row


def main():
    output = ROOT / "results" / "convergence"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "fixed physical widths; restricted AC and CH models", "studies": {}}
    for kind in ("ac", "ch"):
        ref_n = 512 if kind == "ac" else 256
        ref_dt = .000625 if kind == "ac" else .005
        reference = solve(kind, ref_n, ref_dt)
        spatial = []
        ns = (64, 128, 256) if kind == "ac" else (32, 64, 128)
        for n in ns:
            h = (64 if kind == "ac" else 32) / n
            dt = .04 * h * h if kind == "ac" else .005
            spatial.append(compare(solve(kind, n, dt), reference))
        n_time = 128 if kind == "ac" else 64
        dt_values = (.04, .02, .01) if kind == "ac" else (.2, .1, .05, .025)
        time_reference = solve(kind, n_time, .001 if kind == "ac" else .0025)
        temporal = [compare(solve(kind, n_time, dt), time_reference) for dt in dt_values]
        space_orders = [float(np.log(spatial[i]["relative_l2"] / spatial[i+1]["relative_l2"]) / np.log(2)) for i in range(2)]
        time_orders = [float(np.log(temporal[i]["relative_l2"] / temporal[i+1]["relative_l2"]) / np.log(2)) for i in range(len(temporal)-1)]
        study = {"spatial": spatial, "temporal": temporal, "spatial_orders": space_orders, "temporal_orders": time_orders,
                 "spatial_reference": reference[2], "temporal_reference": time_reference[2],
                 "spatial_decreases": all(spatial[i+1]["relative_l2"] < spatial[i]["relative_l2"] for i in range(2)),
                 "temporal_decreases": all(temporal[i+1]["relative_l2"] < temporal[i]["relative_l2"] for i in range(len(temporal)-1))}
        summary["studies"][kind] = study
        print(kind, "spatial orders", space_orders, "temporal orders", time_orders, flush=True)
        np.savez_compressed(output / f"{kind}-reference.npz", xe=reference[0].xe, ye=reference[0].ye, field=reference[1])
    (output / "convergence.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    if not all(s["spatial_decreases"] and s["temporal_decreases"] for s in summary["studies"].values()):
        raise SystemExit("A declared convergence criterion failed; inspect convergence.json.")


if __name__ == "__main__":
    main()
