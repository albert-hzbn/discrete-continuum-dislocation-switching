"""Fresh-process timing of existing models with disjoint accounting and AC cost curves."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bmvr import _resource as resource
import numpy as np
import scipy
from bmvr.experiments import ac_run, case2, case9
from bmvr.grid import Grid
from bmvr.adaptivity import transfer
from bmvr.diagnostics import field_errors, radius
from continuation import allocation_run


def dump(path, value):
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported report value: {type(item).__name__}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=scalar) + "\n")


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def worker(config, output):
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    kind, variant = config["kind"], config["variant"]
    if kind == "ac":
        report, arrays = ac_run(n=config["n"], dt=config["dt"], adaptive=variant == "adaptive")
    elif kind == "allocation":
        report, arrays = allocation_run(variant)
    else:
        report, arrays = (case2 if kind == "ch" else case9)(variant)
    total = perf_counter() - start
    numerical_peak = rss()
    raw = report["timings"]
    categories = {
        "initialization_s": raw.get("initialization_s", 0.),
        "solver_s": raw.get("solver_s", 0.),
        "adaptation_s": raw.get("adaptation_s", raw.get("allocation_s", 0.)),
        "diagnostics_s": raw.get("diagnostics_s", 0.) + raw.get("final_validation_s", 0.),
        "compilation_s": 0.,
    }
    categories["other_experiment_s"] = total - sum(categories.values())
    if categories["other_experiment_s"] < -1e-9:
        raise RuntimeError("Overlapping experiment timing categories")
    start = perf_counter()
    np.savez_compressed(output / "fields.npz", **arrays)
    categories["array_output_s"] = perf_counter() - start
    start = perf_counter()
    dump(output / "report.json", report)
    categories["report_output_s"] = perf_counter() - start
    dump(output / "measurement.json", {
        "config": config, "categories": categories, "experiment_s": total,
        "numerical_peak_rss_bytes": numerical_peak, "post_output_peak_rss_bytes": rss(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform(),
                        "machine": platform.machine(), "processor": platform.processor(),
                        "logical_cpus": os.cpu_count(), "numerical_threads": 1},
    })


def configs():
    matrix = []
    for n in (128, 192, 256):
        for variant in ("reference", "adaptive"):
            matrix.append({"kind": "ac", "variant": variant, "n": n, "dt": .02 * (256/n)**2})
    for kind in ("ch", "hybrid"):
        matrix.extend({"kind": kind, "variant": v} for v in ("reference", "adaptive"))
    matrix.extend({"kind": "allocation", "variant": v} for v in ("uniform", "geometry", "residual", "cost"))
    return matrix


def name(config):
    return f"{config['kind']}-{config.get('n', 'default')}-{config['variant']}"


def load_run(folder):
    r = json.loads((folder / "report.json").read_text())
    m = json.loads((folder / "measurement.json").read_text())
    p = json.loads((folder / "process.json").read_text())
    categories = m["categories"]
    categories["process_measurement_overhead_s"] = p["launch_to_exit_s"] - sum(categories.values())
    if categories["process_measurement_overhead_s"] < 0:
        raise RuntimeError("Parent timing shorter than measured worker categories")
    return {"folder": str(folder.name), "report": r, **m, **p}


def read_fields(output, row):
    with np.load(output / row["folder"] / "fields.npz") as data:
        arrays = {key: data[key] for key in data.files}
    g = Grid(arrays["xe"], arrays["ye"], periodic=row["config"]["kind"] == "ch")
    return g, arrays


def collect(output):
    started = perf_counter()
    runs = [load_run(folder) for folder in sorted(output.glob("run-*"))]
    fine = load_run(output / "fine-reference")
    fg, fa = read_fields(output, fine)
    rows = []
    groups = {}
    for config in configs():
        group = [r for r in runs if r["config"] == config]
        if not group:
            continue
        groups[name(config)] = group
        median = lambda fn: float(np.median([fn(r) for r in group]))
        report = group[0]["report"]
        metrics = report.get("metrics", report)
        row = {"name": name(config), "config": config, "repetitions": len(group),
               "median_categories_s": {k: median(lambda r: r["categories"][k]) for k in group[0]["categories"]},
               "launch_to_exit_s": median(lambda r: r["launch_to_exit_s"]),
               "launch_to_exit_range_s": [min(r["launch_to_exit_s"] for r in group), max(r["launch_to_exit_s"] for r in group)],
               "evolution_s": median(lambda r: r["categories"]["solver_s"] + r["categories"]["adaptation_s"]),
               "post_output_peak_rss_bytes": median(lambda r: r["post_output_peak_rss_bytes"]),
               "final_dofs": metrics["final_dofs"],
               "stored_array_bytes": metrics.get("stored_array_bytes", metrics.get("stored_field_bytes")),
               "steps": metrics.get("steps"),
               "solver_iteration_count": metrics.get("solver_iterations"),
               "checks": report["checks"], "metrics": metrics,
               "all_repetition_checks_pass": all(all(r["report"]["checks"].values()) for r in group)}
        if config["kind"] == "ac":
            g, a = read_fields(output, group[0])
            ref = transfer(fg, g, (fa["field"] + 1)/2)
            error = field_errors(g, ref, g, (a["field"] + 1)/2)
            er = abs(float(a["energy"][-1] / fa["energy"][-1]) - 1)
            dr = abs(radius(g, a["field"]) - radius(fg, fa["field"]))
            row["fine_reference_comparison"] = {"phase_fraction": error, "radius_absolute": dr,
                "native_energy_relative": er, "accuracy_met": bool(error["relative_l2"] <= .01 and dr <= .05 and er <= .01)}
        rows.append(row)
    pairs = []
    for row in rows:
        config = row["config"]
        if config["variant"] in ("uniform", "reference"):
            continue
        rc = {**config, "variant": "uniform" if config["kind"] == "allocation" else "reference"}
        rr = next(r for r in rows if r["name"] == name(rc))
        rg, ra = read_fields(output, groups[rr["name"]][0])
        g, a = read_fields(output, groups[row["name"]][0])
        errors = {}
        for key in ("field", "concentration", "phase", "population"):
            if key in a:
                errors[key] = field_errors(rg, ra[key], g, a[key])
        if "phase" in a:
            errors["phase_fraction"] = field_errors(rg, (ra["phase"]+1)/2, g, (a["phase"]+1)/2)
            errors["population_excess"] = field_errors(rg, ra["population"]-.2, g, a["population"]-.2)
        energy_error = abs(float(a["energy"][-1]-ra["energy"][-1])) / max(abs(float(ra["energy"][-1])), 1e-30)
        position = float(np.max(np.abs(a["positions"][-1]-ra["positions"][-1]))) if "positions" in a else None
        pair = {"variant": row["name"], "reference": rr["name"], "evolution_ratio": rr["evolution_s"]/row["evolution_s"],
                "launch_to_exit_ratio": rr["launch_to_exit_s"]/row["launch_to_exit_s"],
                "dof_ratio": rr["final_dofs"]/row["final_dofs"], "field_errors": errors,
                "native_energy_relative": energy_error, "line_position_absolute": position,
                "fine_reference_accuracy_met": row.get("fine_reference_comparison", {}).get("accuracy_met")}
        if config["kind"] == "allocation":
            pair["declared_pair_accuracy_met"] = bool(all(e["relative_l2"] <= .01 for e in errors.values()) and position <= .01)
        pairs.append(pair)
    sources = list((ROOT / "src/bmvr").glob("*.py")) + [Path(__file__), ROOT/"benchmarks/continuation.py", ROOT/"docs/performance/PERFORMANCE_PROTOCOL.md"]
    summary = {"protocol": "docs/performance/PERFORMANCE_PROTOCOL.md", "runs": runs, "fine_reference": fine, "rows": rows, "pairs": pairs,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "collection_s_before_summary_output": perf_counter()-started,
        "notes": ["Component medians need not add to median total; each individual run closes exactly.",
                  "Fine-reference error and launch overhead are not eliminated by these comparisons.",
                  "No independent adaptive PF or parallel spatial solver scaling claim."]}
    dump(output / "summary.json", summary)
    for pair in pairs:
        print(pair["variant"], "evolution", round(pair["evolution_ratio"], 3), "launch-to-exit", round(pair["launch_to_exit_ratio"], 3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT/"results/performance-complete-2026-09-13")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--worker")
    p.add_argument("--collect", action="store_true")
    args = p.parse_args()
    if args.worker:
        worker(json.loads(args.worker), args.output)
        return
    if args.repeats < 1:
        p.error("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.collect:
        def launch(config, folder):
            if (folder/"report.json").exists():
                raise FileExistsError(f"Preserve existing results; choose a new output directory: {folder}")
            started = perf_counter()
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(config), "--output", str(folder)], check=True, cwd=ROOT)
            dump(folder/"process.json", {"launch_to_exit_s": perf_counter()-started})
            print(folder.name, "finished", flush=True)
        launch({"kind": "ac", "variant": "reference", "n": 512, "dt": .0025}, args.output/"fine-reference")
        matrix = configs()
        for repeat in range(args.repeats):
            shift = repeat*5 % len(matrix)
            for config in matrix[shift:]+matrix[:shift]:
                launch(config, args.output/f"run-{name(config)}-{repeat+1}")
    collect(args.output)


if __name__ == "__main__":
    main()
