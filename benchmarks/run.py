"""Run each benchmark in a fresh process with one numerical thread."""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import json
import platform
from bmvr import _resource as resource
import subprocess
from time import perf_counter
import numpy as np
import scipy
from bmvr.experiments import CASES
from bmvr.grid import Grid
from bmvr.diagnostics import field_errors


def worker(case, variant, output):
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    report, arrays = CASES[case](variant)
    report["timings"]["experiment_total_s"] = perf_counter() - start
    measured = sum(v for k, v in report["timings"].items() if k != "experiment_total_s")
    report["timings"]["unclassified_s"] = max(0., report["timings"]["experiment_total_s"] - measured)
    report["timing_notes"] = "Unclassified time includes uninstrumented setup/diagnostics. Peak RSS includes interpreter, imports, solver temporaries and diagnostic audit grids, measured before file I/O."
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["metrics"]["peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
    report["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "platform": platform.platform(),
                              "architecture": platform.machine(), "threads": 1, "seed_policy": "fixed per case"}
    start = perf_counter()
    np.savez_compressed(output / "fields.npz", **arrays)
    report["timings"]["array_io_s"] = perf_counter() - start
    report["data_file"] = "fields.npz"
    report["checks"] = {k: bool(v) for k, v in report["checks"].items()}
    report["all_implemented_checks_pass"] = all(report["checks"].values())
    with (output / "report.json").open("w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"case {case} {variant}: {sum(report['checks'].values())}/{len(report['checks'])} implemented checks; full coverage remains {report['status']}", flush=True)


def collect(output):
    reports = []
    for path in sorted(output.glob("case-*/report.json")):
        item = json.loads(path.read_text())
        item["folder"] = path.parent.name
        reports.append(item)
    comparisons = []
    for case in (1, 2, 9):
        ref = [r for r in reports if r["case"] == case and r["variant"] == "reference"]
        ada = [r for r in reports if r["case"] == case and r["variant"] == "adaptive"]
        if not ref or not ada:
            continue
        runtime = lambda rows: float(np.median([r["timings"]["solver_s"] + r["timings"]["adaptation_s"] for r in rows]))
        # Compare one deterministic field pair; runtime uses all repetitions.
        rr, ar = ref[0], ada[0]
        with np.load(output / rr["folder"] / "fields.npz") as ra, np.load(output / ar["folder"] / "fields.npz") as aa:
            rg = Grid(ra["xe"], ra["ye"], periodic=case == 2)
            ag = Grid(aa["xe"], aa["ye"], periodic=case == 2)
            errors = field_errors(rg, ra["field"], ag, aa["field"])
            concentration_errors = field_errors(rg, ra["concentration"], ag, aa["concentration"]) if case == 9 else None
            energy_error = abs(float(ra["energy"][-1] - aa["energy"][-1])) / max(1., abs(float(ra["energy"][-1])))
            position_error = float(np.max(np.abs(ra["positions"][-1] - aa["positions"][-1]))) if case == 9 else None
        peak_ref = float(np.median([r["metrics"]["peak_rss_bytes"] for r in ref]))
        peak_ada = float(np.median([r["metrics"]["peak_rss_bytes"] for r in ada]))
        comparison = {"case": case, "reference": "uniform PF" if case != 9 else "uniform-field hybrid",
                      "reference_runtime_s": runtime(ref), "adaptive_runtime_s": runtime(ada),
                      "speedup": runtime(ref) / runtime(ada), "repetitions": min(len(ref), len(ada)),
                      "reference_peak_rss_bytes": peak_ref, "adaptive_peak_rss_bytes": peak_ada,
                      "peak_rss_ratio": peak_ref / peak_ada,
                      "state_storage_ratio": rr["metrics"]["stored_array_bytes"] / ar["metrics"]["stored_array_bytes"],
                      "dof_ratio": rr["metrics"]["final_dofs"] / ar["metrics"]["final_dofs"],
                      "field_error": errors, "concentration_error": concentration_errors,
                      "free_energy_relative_error": energy_error, "line_position_absolute_error": position_error,
                      "field_tolerance_met": errors["relative_l2"] <= .01}
        if case == 1:
            comparison["radius_relative_error"] = abs(rr["metrics"]["final_radius"] - ar["metrics"]["final_radius"]) / rr["metrics"]["final_radius"]
            comparison["field_tolerance_met"] &= comparison["radius_relative_error"] <= .01
        if case == 9:
            comparison["field_tolerance_met"] &= concentration_errors["relative_l2"] <= .01 and position_error <= .01
        comparisons.append(comparison)
    summary = {"reports": reports, "comparisons": comparisons,
               "overall_status": "partial: reduced models only; original full acceptance not established"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", choices=["all"] + [str(i) for i in CASES])
    parser.add_argument("--variant", default="both", choices=["both", "reference", "adaptive"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "verification")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.worker:
        worker(int(args.case), args.variant, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.collect:
        cases = list(CASES) if args.case == "all" else [int(args.case)]
        for case in cases:
            variants = ("reference", "adaptive") if case in (1, 2, 9) else ("reference",)
            if args.variant != "both":
                if args.variant not in variants:
                    parser.error(f"Case {case} does not implement variant {args.variant}")
                variants = (args.variant,)
            repetitions = args.repeats if case in (1, 2, 9) else 1
            for variant in variants:
                for repeat in range(1, repetitions + 1):
                    folder = args.output / f"case-{case:02d}-{variant}-{repeat}"
                    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--case", str(case),
                               "--variant", variant, "--output", str(folder)]
                    subprocess.run(command, check=True, cwd=ROOT)
    summary = collect(args.output)
    for row in summary["comparisons"]:
        print(f"case {row['case']}: runtime ratio {row['speedup']:.3f}, DOF ratio {row['dof_ratio']:.3f}, relative L2 {row['field_error']['relative_l2']:.3g}", flush=True)


if __name__ == "__main__":
    main()
