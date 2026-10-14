"""Audit saved timing identities, deterministic repetitions and declared study gates."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT/"results/performance-complete-2026-09-13")
    args = p.parse_args()
    s = json.loads((args.input/"summary.json").read_text())
    runs = s["runs"]
    checks = {}
    checks["source_hashes_match"] = all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest() == digest for f,digest in s["source_sha256"].items())
    checks["matrix_42_runs"] = len(runs) == 42
    checks["one_thread"] = all(r["environment"]["numerical_threads"] == 1 for r in runs)
    checks["nonnegative_timing_categories"] = all(min(r["categories"].values()) >= 0 for r in runs)
    closure = max(abs(sum(r["categories"].values()) - r["launch_to_exit_s"]) for r in runs)
    checks["timing_accounting_closes"] = closure < 1e-12
    deterministic = True
    repetitions = True
    for row in s["rows"]:
        group = [r for r in runs if r["config"] == row["config"]]
        repetitions &= len(group) == 3
        with np.load(args.input/group[0]["folder"]/"fields.npz") as base:
            for run in group[1:]:
                with np.load(args.input/run["folder"]/"fields.npz") as other:
                    deterministic &= base.files == other.files and all(np.array_equal(base[k],other[k]) for k in base.files)
    checks["three_repetitions_per_configuration"] = bool(repetitions)
    checks["all_repetition_arrays_identical"] = bool(deterministic)
    matched = True
    for pair in s["pairs"]:
        other = next(r for r in s["rows"] if r["name"] == pair["variant"])
        ref = next(r for r in s["rows"] if r["name"] == pair["reference"])
        oc = {k:v for k,v in other["config"].items() if k != "variant"}
        rc = {k:v for k,v in ref["config"].items() if k != "variant"}
        matched &= oc == rc and other["steps"] == ref["steps"]
        og = next(r for r in runs if r["config"] == other["config"])
        rg = next(r for r in runs if r["config"] == ref["config"])
        with np.load(args.input/og["folder"]/"fields.npz") as a, np.load(args.input/rg["folder"]/"fields.npz") as b:
            matched &= np.array_equal(a["time"], b["time"])
    checks["paired_configuration_steps_and_times_match"] = bool(matched)
    checks["all_existing_numerical_checks_pass"] = all(all(r["report"]["checks"].values()) for r in runs+[s["fine_reference"]])
    result = {"scope":"Saved-run consistency checks; scientific acceptance gates reported separately.",
        "checks":checks,"maximum_timing_closure_error_s":closure,
        "numerical_checks_passed":sum(sum(r["report"]["checks"].values()) for r in runs+[s["fine_reference"]]),
        "numerical_checks_total":sum(len(r["report"]["checks"]) for r in runs+[s["fine_reference"]]),
        "ac_accuracy_gates":{r["name"]:r["fine_reference_comparison"]["accuracy_met"] for r in s["rows"] if "fine_reference_comparison" in r},
        "allocation_pair_accuracy_gates":{r["variant"]:r["declared_pair_accuracy_met"] for r in s["pairs"] if "declared_pair_accuracy_met" in r},
        "source_sha256":{str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "full_original_objective_complete":False}
    (args.input/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not all(checks.values()):
        raise SystemExit("Saved-run consistency check failed")


if __name__ == "__main__":
    main()
