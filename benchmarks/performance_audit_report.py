"""Tables and standalone figures from the complete timing measurements."""
import os
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path
import argparse
import csv
import json
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT/"results/performance-complete-2026-09-13")
    args = p.parse_args()
    started = perf_counter()
    summary = json.loads((args.input/"summary.json").read_text())
    rows, pairs = summary["rows"], summary["pairs"]
    figure_dir = args.input/"figures"
    figure_dir.mkdir(exist_ok=True)
    labels = [r["name"].replace("-default", "") for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for variant, color in (("reference", "#275d8c"), ("adaptive", "#b34e26")):
        ac = [r for r in rows if r["config"]["kind"] == "ac" and r["config"]["variant"] == variant]
        x = [r["launch_to_exit_s"] for r in ac]
        y = [100*r["fine_reference_comparison"]["phase_fraction"]["relative_l2"] for r in ac]
        ax[0].loglog(x, y, "o-", label=variant, color=color)
        for r, xx, yy in zip(ac, x, y):
            ax[0].annotate(str(r["config"]["n"]), (xx, yy), xytext=(5, 6 if variant == "reference" else -14), textcoords="offset points", fontsize=9)
    ax[0].axhline(1, ls=":", color="black", lw=1, label="1% field threshold")
    ax[0].margins(x=.12, y=.16)
    ax[0].set(xlabel="Launch-to-exit time (s)", ylabel="Phase-fraction relative L2 (%)", title="AC: fixed-width accuracy versus cost")
    ax[0].legend(fontsize=8)
    pos = np.arange(len(pairs))
    ax[1].barh(pos-.17, [r["evolution_ratio"] for r in pairs], .32, label="Solver + adaptation", color="#275d8c")
    ax[1].barh(pos+.17, [r["launch_to_exit_ratio"] for r in pairs], .32, label="Launch to exit", color="#b34e26")
    ax[1].axvline(1, color="black", ls=":", lw=1)
    ax[1].set(yticks=pos, yticklabels=[r["variant"].replace("-default", "") for r in pairs], xlabel="Uniform / variant elapsed time", title="Matched timing ratios")
    ax[1].invert_yaxis()
    ax[1].legend(fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(figure_dir/f"accuracy-cost.{ext}", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    # Stack one actual median-total run per configuration; component medians need not add.
    keys = ["initialization_s", "solver_s", "adaptation_s", "diagnostics_s", "other_experiment_s", "array_output_s", "report_output_s", "process_measurement_overhead_s"]
    names = ["Initialization", "Solver", "Adaptation", "Diagnostics", "Other numerical", "Arrays I/O", "Report I/O", "Process/measurement"]
    palette = ["#8ca3ae", "#275d8c", "#b34e26", "#619471", "#c6a44e", "#8d6b9c", "#bfb6a3", "#dedede"]
    selected = []
    for row in rows:
        group = sorted([r for r in summary["runs"] if r["config"] == row["config"]], key=lambda r:r["launch_to_exit_s"])
        selected.append(group[len(group)//2])
    left = np.zeros(len(rows))
    for key, label, color in zip(keys, names, palette):
        values = np.array([r["categories"][key] for r in selected])
        ax.barh(np.arange(len(rows)), values, left=left, label=label, color=color)
        left += values
    ax.set(yticks=np.arange(len(rows)), yticklabels=labels, xlabel="Elapsed time (s)", title="Complete timing: actual median-total run per configuration")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(figure_dir/f"timing-categories.{ext}", dpi=160)
    plt.close(fig)
    plot_time = perf_counter()-started
    lines = ["# Fresh performance measurements", "", "Generated from summary.json. Interpret with ../../docs/performance/PERFORMANCE_PROTOCOL.md; three repetitions per configuration and one finer AC reference. Component medians need not add to the median total.", "", "| Configuration | Evolution (s) | Launch–exit (s) | Range (s) | Final DOFs | Peak RSS (MiB) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lo, hi = r["launch_to_exit_range_s"]
        lines.append(f"| {r['name']} | {r['evolution_s']:.6f} | {r['launch_to_exit_s']:.6f} | {lo:.4f}–{hi:.4f} | {r['final_dofs']:,} | {r['post_output_peak_rss_bytes']/2**20:.2f} |")
    lines += ["", "| Variant | Evolution ratio | Launch–exit ratio | DOF ratio | Native energy discrepancy |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in pairs:
        lines.append(f"| {r['variant']} | {r['evolution_ratio']:.3f} | {r['launch_to_exit_ratio']:.3f} | {r['dof_ratio']:.3f} | {r['native_energy_relative']:.3e} |")
    lines += ["", "| AC configuration | Phase-fraction relative L2 | Radius error | Native energy error | All accuracy gates |", "| --- | ---: | ---: | ---: | --- |"]
    for r in rows:
        if "fine_reference_comparison" in r:
            a = r["fine_reference_comparison"]
            lines.append(f"| {r['name']} | {a['phase_fraction']['relative_l2']:.6g} | {a['radius_absolute']:.6g} | {a['native_energy_relative']:.6g} | {a['accuracy_met']} |")
    lines += ["", "Solver iteration counters are unavailable where the source reports only accepted steps. No iteration count of zero is inferred. The complete raw reports retain conservation/energy checks, transfer events and model parameters.", "", "![Accuracy and ratios](figures/accuracy-cost.png)", "", "![Timing categories](figures/timing-categories.png)", ""]
    (args.input/"REPORT.md").write_text("\n".join(lines))
    with (args.input/"performance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "evolution_s", "launch_to_exit_s", "final_dofs", "post_output_peak_rss_bytes"])
        writer.writeheader()
        writer.writerows({key:r[key] for key in writer.fieldnames} for r in rows)
    (figure_dir/"plotting.json").write_text(json.dumps({"plotting_and_figure_io_s":plot_time,"total_report_generation_s_before_metadata":perf_counter()-started},indent=2)+"\n")
    print("Generated two PNG/SVG figure pairs and tables.")


if __name__ == "__main__":
    main()
