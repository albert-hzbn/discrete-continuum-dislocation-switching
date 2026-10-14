"""Matched joint-allocation ablations and synthetic orientation-resolved reactions."""

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import argparse
from dataclasses import asdict, replace
import json
import platform
from bmvr import _resource as resource
import subprocess
from time import perf_counter
import numpy as np
import scipy
from bmvr.grid import Grid, circle
from bmvr.defects import Screws, coupled_energy_forces
from bmvr.allocation import allocate, two_grid_indicator, audit_grid, TransferCache
from bmvr.diagnostics import field_errors, energy_increase
from bmvr.interface_reactions import InterfaceParameters, InterfaceReactionModel, NAMES, approach_exposure


def allocation_run(strategy):
    started = perf_counter()
    length, h, dt, final_time = 96., .5, .02, 4.
    grid = Grid.uniform(length, int(length / h))
    v = circle(grid, 5., center=(-14., -14.))
    c = .2 + .08 * np.exp(-((grid.X - 14) ** 2 + (grid.Y - 14) ** 2) / (2 * 1.5 ** 2))
    screws = Screws(np.array([[-3., 12.], [3., 12.]]), np.array([1., -1.]), core=1.)
    origins = screws.positions.copy()
    initial_inventory = grid.integrate(c)
    params = {"imposed": .08, "chemical": .3, "interface": .2}
    def evaluate(g, fields):
        r = coupled_energy_forces(g, fields[0], fields[1], screws, origins, **params)
        return {"energy": r["energy"], "rates": [-r["gv"], g.lap(r["mu"])],
                "line_velocity": screws.mobility * r["force"], "line_points": screws.positions,
                "line_core": screws.core, "line_mobility": screws.mobility}
    initialization = perf_counter() - started
    allocation_s, solver_s = 0., 0.
    audits = []
    if strategy != "uniform":
        started = perf_counter()
        grid, fields, event = allocate(grid, [v, c], evaluate, h, mode=strategy)
        v, c = fields
        event["time"] = 0.
        audits.append(event)
        allocation_s += perf_counter() - started
    times, energies, inventories, cells = [0.], [evaluate(grid, [v, c])["energy"]], [grid.integrate(c)], [grid.size]
    positions = [screws.positions.copy()]
    dt_used = []
    t, steps = 0., 0
    while t < final_time - 1e-12:
        started = perf_counter()
        current = evaluate(grid, [v, c])
        used = min(dt, final_time - t)
        for _ in range(24):
            nv, nc = v + used * current["rates"][0], c + used * current["rates"][1]
            ns = screws.moved(screws.positions + used * current["line_velocity"])
            trial = coupled_energy_forces(grid, nv, nc, ns, origins, **params)
            if nc.min() > 0 and trial["energy"] <= current["energy"] + 1e-10:
                break
            used *= .5
        else:
            raise RuntimeError("Coupled step failed its bounds/energy check.")
        v, c, screws = nv, nc, ns
        t += used
        dt_used.append(used)
        steps += 1
        solver_s += perf_counter() - started
        if strategy != "uniform" and steps % 25 == 0:
            started = perf_counter()
            if strategy == "geometry":
                grid, fields, event = allocate(grid, [v, c], evaluate, h, mode=strategy)
                v, c = fields
            else:
                indicator = two_grid_indicator(grid, [v, c], audit_grid(grid, h), evaluate)
                event = {k: value for k, value in indicator.items() if k != "ranking_map"}
                event.update({"reason": "current_grid_budgets_met", "accepted": True, "candidate_cells": grid.size})
                if not indicator["met"]:
                    grid, fields, event = allocate(grid, [v, c], evaluate, h, mode=strategy, refine_only=True)
                    v, c = fields
            event["time"] = t
            audits.append(event)
            allocation_s += perf_counter() - started
        if steps % 25 == 0 or t >= final_time - 1e-12:
            times.append(t)
            energies.append(evaluate(grid, [v, c])["energy"])
            inventories.append(grid.integrate(c))
            cells.append(grid.size)
            positions.append(screws.positions.copy())
    validation_start = perf_counter()
    final_indicator = two_grid_indicator(grid, [v, c], audit_grid(grid, h), evaluate)
    validation_s = perf_counter() - validation_start
    final_indicator = {k: value for k, value in final_indicator.items() if k != "ranking_map"}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {"strategy": strategy, "model": "isothermal coupled hybrid; separated structural and chemical features",
              "parameters": {"length": length, "h": h, "dt": dt, "final_time": final_time,
                             "profile_scale": float(np.sqrt(2)), "core_width": 1.,
                             "relative_indicator_tolerances": [.05, .05, .05], **params},
              "timings": {"initialization_s": initialization, "solver_s": solver_s,
                          "allocation_s": allocation_s, "final_validation_s": validation_s,
                          "evolution_and_allocation_s": solver_s + allocation_s},
              "steps": steps, "minimum_used_dt": min(dt_used), "final_dofs": 2 * grid.size + 2,
              "mean_field_cells": float(np.mean(cells)),
              "inventory_drift": float(np.max(np.abs(np.array(inventories) - initial_inventory))),
              "max_recorded_energy_increase": energy_increase(energies),
              "burgers_content": float(screws.burgers.sum()),
              "stored_field_bytes": int(v.nbytes + c.nbytes + grid.X.nbytes + grid.Y.nbytes + grid.vol.nbytes),
              "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
              "final_indicator": final_indicator, "audits": audits,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "platform": platform.platform(), "threads": 1}}
    report["checks"] = {"inventory": report["inventory_drift"] < 1e-8,
                         "burgers": report["burgers_content"] == 0,
                         "energy": report["max_recorded_energy_increase"] < 1e-7,
                         "positive_population": bool(c.min() > 0)}
    return report, {"xe": grid.xe, "ye": grid.ye, "phase": v, "population": c,
                    "time": np.array(times), "energy": np.array(energies), "inventory": np.array(inventories),
                    "cells": np.array(cells), "positions": np.array(positions)}


def reaction_run(output):
    times = np.linspace(0, 80, 401)
    distance, exposure = approach_exposure(times)
    scenarios = {
        "transmission_favored": InterfaceParameters(angle_degrees=10),
        "absorption_favored": InterfaceParameters(angle_degrees=65, drive_work=.4, binding_energy=.6, absorption_barrier=.2),
        "reflection_favored": InterfaceParameters(angle_degrees=65, drive_work=0., transmission_barrier=1.2,
                                                  absorption_barrier=1.2, reflection_barrier=.05),
    }
    results, arrays = [], {"time": times, "distance": distance, "exposure": exposure}
    for k, parameters in scenarios.items():
        model = InterfaceReactionModel(parameters)
        p = model.propagate(exposure)
        energy = model.ensemble_free_energy(p)
        exact = model.first_event_probabilities()
        waits, samples = model.sample_first_events(20000, seed=20260912)
        empirical = np.bincount(samples, minlength=4) / len(samples)
        errors = np.linalg.norm(model.mobile[samples] + model.residual[samples] - model.incident_burgers, axis=1)
        flux = model.stationary[:, None] * model.rates
        se = np.sqrt(exact * (1 - exact) / len(samples))
        row = {"name": k, "parameters": asdict(parameters), "state_energies": model.energies.tolist(),
               "rates": model.rates.tolist(), "stationary": model.stationary.tolist(),
               "first_event_probabilities": exact.tolist(), "sampled_probabilities": empirical.tolist(),
               "dominant_first_event": NAMES[int(np.argmax(exact))],
               "maximum_burgers_error": float(errors.max()), "minimum_probability": float(p.min()),
               "maximum_probability_sum_error": float(np.max(np.abs(p.sum(axis=1) - 1))),
               "maximum_ensemble_energy_increase": energy_increase(energy),
               "detailed_balance_error": float(np.max(np.abs(flux - flux.T))),
               "stationary_generator_residual": float(np.max(np.abs(model.generator @ model.stationary))),
               "events": [model.event(0, j) for j in (1, 2, 3)],
               "sampling_within_5se": bool(np.all(np.abs(empirical - exact) <= 5 * se + 1 / len(samples)))}
        row["checks"] = {"burgers": row["maximum_burgers_error"] < 1e-12,
                         "probability": row["minimum_probability"] >= -1e-12 and row["maximum_probability_sum_error"] < 1e-12,
                         "ensemble_dissipation": row["maximum_ensemble_energy_increase"] < 1e-10,
                         "detailed_balance": row["detailed_balance_error"] < 1e-12 and row["stationary_generator_residual"] < 1e-12,
                         "sampling": row["sampling_within_5se"]}
        results.append(row)
        arrays[k + "_probability"] = p
        arrays[k + "_ensemble_energy"] = energy
        arrays[k + "_sample_states"] = samples
        arrays[k + "_waiting_times"] = waits
    angles, drives = np.linspace(0, 80, 41), np.linspace(0, 1, 31)
    sweep = np.empty((len(drives), len(angles), 4))
    residuals = np.empty((len(angles), 3))
    for i, drive in enumerate(drives):
        for j, angle in enumerate(angles):
            model = InterfaceReactionModel(InterfaceParameters(angle_degrees=float(angle), drive_work=float(drive)))
            sweep[i, j] = model.first_event_probabilities()
            residuals[j] = model.residual[1]
    arrays.update({"angles": angles, "drives": drives, "first_event_sweep": sweep, "transmission_residuals": residuals})
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "reactions.npz", **arrays)
    report = {"scope": "synthetic orientation-resolved reaction closure; prescribed finite-width approach",
              "state_names": NAMES, "scenarios": results, "seed": 20260912,
              "limits": ["Not an atomistically calibrated prediction.", "No reactive 3D line-network or evolving-interface feedback.",
                         "Configuration energy may increase in a thermal event; ensemble free energy is the dissipative quantity."]}
    (output / "reactions.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for row in results:
        print(row["name"], row["dominant_first_event"], sum(row["checks"].values()), "/", len(row["checks"]), flush=True)


def collect(output):
    reports = []
    for path in sorted(output.glob("allocation-*/report.json")):
        r = json.loads(path.read_text())
        r["folder"] = path.parent.name
        reports.append(r)
    references = [r for r in reports if r["strategy"] == "uniform"]
    comparisons = []
    if references:
        reference = references[0]
        ref = np.load(output / reference["folder"] / "fields.npz")
        rg = Grid(ref["xe"], ref["ye"])
        runtime = lambda rs: float(np.median([r["timings"]["evolution_and_allocation_s"] for r in rs]))
        for strategy in ("geometry", "residual", "cost"):
            rows = [r for r in reports if r["strategy"] == strategy]
            if not rows:
                continue
            other = rows[0]
            data = np.load(output / other["folder"] / "fields.npz")
            grid = Grid(data["xe"], data["ye"])
            errors = {
                "phase": field_errors(rg, ref["phase"], grid, data["phase"]),
                "phase_fraction": field_errors(rg, (ref["phase"] + 1) / 2, grid, (data["phase"] + 1) / 2),
                "population": field_errors(rg, ref["population"], grid, data["population"]),
                "population_excess": field_errors(rg, ref["population"] - .2, grid, data["population"] - .2),
            }
            position_error = float(np.max(np.abs(data["positions"][-1] - ref["positions"][-1])))
            accurate = all(v["relative_l2"] <= .01 for v in errors.values()) and position_error <= .01
            comparisons.append({"strategy": strategy, "repetitions": min(len(rows), len(references)),
                                "reference_runtime_s": runtime(references), "runtime_s": runtime(rows),
                                "runtime_ratio": runtime(references) / runtime(rows),
                                "dof_ratio": reference["final_dofs"] / other["final_dofs"],
                                "errors": errors, "line_position_error": position_error, "accuracy_met": bool(accurate),
                                "final_indicator_budgets_met": other["final_indicator"]["met"],
                                "reference_peak_rss_bytes": float(np.median([r["peak_rss_bytes"] for r in references])),
                                "peak_rss_bytes": float(np.median([r["peak_rss_bytes"] for r in rows]))})
            data.close()
        ref.close()
    summary = {"reports": reports, "comparisons": comparisons, "date": "2026-09-12",
               "interpretation": "Two-grid operator discrepancy; no exact Galerkin optimality or novelty claim."}
    (output / "allocation-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    for r in comparisons:
        print(r["strategy"], "ratio", round(r["runtime_ratio"], 3), "DOF", round(r["dof_ratio"], 3),
              "accurate", r["accuracy_met"], "indicator", r["final_indicator_budgets_met"], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("all", "allocation", "reactions"), default="all")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, default=ROOT / "results/continuation-2026-09-12")
    p.add_argument("--worker", choices=("uniform", "geometry", "residual", "cost"))
    p.add_argument("--collect", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        report, arrays = allocation_run(args.worker)
        np.savez_compressed(args.output / "fields.npz", **arrays)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(args.worker, "dofs", report["final_dofs"], "allocation", round(report["timings"]["allocation_s"], 3),
              "checks", report["checks"], flush=True)
        return
    if args.repeats < 1:
        p.error("--repeats must be positive")
    if not args.collect:
        if args.part in ("all", "allocation"):
            for repeat in range(1, args.repeats + 1):
                # Rotate order between repetitions to reduce ordering bias.
                strategies = ("uniform", "geometry", "residual", "cost")
                shift = (repeat - 1) % 4
                for strategy in strategies[shift:] + strategies[:shift]:
                    folder = args.output / f"allocation-{strategy}-{repeat}"
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", strategy,
                                    "--output", str(folder)], cwd=ROOT, check=True)
        if args.part in ("all", "reactions"):
            reaction_run(args.output)
    collect(args.output)


if __name__ == "__main__":
    main()
