"""Generate saved figures, performance tables, and a coverage-aware results report."""

import os
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path
import json
import csv
from time import perf_counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docpaths import write_doc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "verification"


def main():
    summary = json.loads((DATA / "summary.json").read_text())
    convergence = json.loads((ROOT / "results/convergence/convergence.json").read_text())
    reports = summary["reports"]
    selected = {}
    for case in range(1, 10):
        variant = "adaptive" if case in (1, 9) else "reference"
        selected[case] = next(r for r in reports if r["case"] == case and r["variant"] == variant)
    arrays = {k: dict(np.load(DATA / r["folder"] / "fields.npz")) for k, r in selected.items()}
    figures = ROOT / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), layout="constrained")
    def field(ax, case, key="field", cmap="coolwarm"):
        a = arrays[case]
        artist = ax.pcolormesh(a["xe"], a["ye"], a[key], shading="flat", cmap=cmap, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return artist
    im = field(axes[0, 0], 1)
    axes[0, 0].set_title("1  Inclusion shrinkage\nAdaptive field grid")
    fig.colorbar(im, ax=axes[0, 0], label="structural field")
    im = field(axes[0, 1], 2)
    axes[0, 1].set_title("2  Spinodal decomposition\nUniform fallback")
    fig.colorbar(im, ax=axes[0, 1], label="composition order parameter")
    field(axes[0, 2], 3, cmap="tab20")
    axes[0, 2].set_title("3  Grain growth · phase labels")
    im = field(axes[1, 0], 4, "equilibrium_sxx")
    axes[1, 0].set_title("4  Coherent precipitate\nStationary normal stress")
    a = arrays[4]
    axes[1, 0].contour((a["xe"][:-1] + a["xe"][1:]) / 2,
                       (a["ye"][:-1] + a["ye"][1:]) / 2,
                       a["equilibrium_field"], levels=[0], colors="white", linewidths=1)
    fig.colorbar(im, ax=axes[1, 0], label="sigma_xx")
    a = arrays[5]
    axes[1, 1].plot(a["x"], a["exact"], color="#172e55", label="analytical")
    axes[1, 1].plot(a["x"][::10], a["field"][::10], "o", ms=3, color="#009e88", label="finite volume")
    axes[1, 1].set(xlabel="distance from sink", ylabel="defect fraction", title="5  Absorbing-sink diffusion")
    axes[1, 1].legend(frameon=False)
    a = arrays[6]
    axes[1, 2].plot(a["time"], a["same_separation"], label="same sign", color="#b94947")
    axes[1, 2].plot(a["time"], a["opposite_separation"], label="opposite sign", color="#32669a")
    axes[1, 2].set(xlabel="time", ylabel="line separation", title="6  Screw interaction / annihilation")
    axes[1, 2].legend(frameon=False)
    a = arrays[7]
    axes[2, 0].plot(a["time"], a["weak_drive_x"], label="weak drive")
    axes[2, 0].plot(a["time"], a["strong_drive_x"], label="stronger drive")
    axes[2, 0].axhline(0, color="#777777", lw=.7, ls="--")
    axes[2, 0].set(xlabel="time", ylabel="line x position", title="7  Diffuse-interface binding only")
    axes[2, 0].legend(frameon=False)
    a = arrays[8]
    axes[2, 1].plot(a["low_time"], a["low_area"], label="low vacancy inventory")
    axes[2, 1].plot(a["high_time"], a["high_area"], label="high vacancy inventory")
    axes[2, 1].set(xlabel="time", ylabel="pore area", title="8  Vacancy-driven pore evolution")
    axes[2, 1].legend(frameon=False)
    im = field(axes[2, 2], 9, "concentration", "viridis")
    a = arrays[9]
    x = (a["xe"][:-1] + a["xe"][1:]) / 2
    y = (a["ye"][:-1] + a["ye"][1:]) / 2
    axes[2, 2].contour(x, y, a["field"], levels=[0], colors="white", linewidths=1)
    pos = a["positions"][-1]
    axes[2, 2].scatter(pos[:, 0], pos[:, 1], s=25, marker="x", color="#e86756")
    axes[2, 2].set_xlim(-16, 16)
    axes[2, 2].set_ylim(-16, 16)
    axes[2, 2].set_title("9  Coupled populations and lines\nFinite-width interface")
    fig.colorbar(im, ax=axes[2, 2], label="defect population")
    fig.suptitle("Restricted variational verification suite", fontsize=18)
    fig.supxlabel("Dimensionless models · physical profile widths fixed · broader material physics remains unverified", fontsize=10)
    fig.savefig(figures / "verification-overview.png", dpi=180)
    fig.savefig(figures / "verification-overview.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    rows = summary["comparisons"]
    labels = ["AC inclusion", "CH spinodal", "Coupled hybrid"]
    colors = ["#267582", "#a0a8ae", "#172e55"]
    for ax, metric, title in zip(axes[:2], ["speedup", "dof_ratio"], ["Runtime ratio (reference / adaptive)", "Unknown-count ratio (reference / adaptive)"]):
        vals = [r[metric] for r in rows]
        ax.bar(labels, vals, color=colors, width=.65)
        ax.axhline(1, ls="--", color="#777777", lw=.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.22)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * .025, f"{v:.2f}×", ha="center")
        ax.tick_params(axis="x", labelsize=8)
    for name, style in (("ac", "o-"), ("ch", "s-")):
        study = convergence["studies"][name]
        axes[2].loglog([r["solver_s"] for r in study["spatial"]], [r["relative_l2"] for r in study["spatial"]], style, label=name.upper())
    axes[2].set(xlabel="solver seconds", ylabel="relative L2 error", title="Spatial accuracy versus cost")
    axes[2].legend(frameon=False)
    fig.suptitle("Measured prototype cost and accuracy", fontsize=15)
    fig.supxlabel("Runtime includes adaptation; three fresh-process repetitions; one thread; no AMR-PF superiority claim", fontsize=9)
    fig.savefig(figures / "performance.png", dpi=180)
    fig.savefig(figures / "performance.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for name, style in (("ac", "o-"), ("ch", "s-")):
        s = convergence["studies"][name]
        axes[0].loglog([r["h"] for r in s["spatial"]], [r["relative_l2"] for r in s["spatial"]], style, label=name.upper())
        axes[1].loglog([r["dt"] for r in s["temporal"]], [r["relative_l2"] for r in s["temporal"]], style, label=name.upper())
    axes[0].set(xlabel="cell width", ylabel="relative L2 error", title="Fixed-width spatial refinement")
    axes[1].set(xlabel="time step", ylabel="relative L2 error", title="Time refinement on a fixed grid")
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=.2, which="both")
    fig.savefig(figures / "convergence.png", dpi=180)
    fig.savefig(figures / "convergence.svg")
    plt.close(fig)
    plotting_s = perf_counter() - start

    with (ROOT / "results/performance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "reference", "repetitions", "reference_runtime_s", "adaptive_runtime_s", "speedup",
                                              "dof_ratio", "state_storage_ratio", "peak_rss_ratio", "relative_l2", "free_energy_relative_error"])
        writer.writeheader()
        for r in rows:
            record = {k: r[k] for k in writer.fieldnames if k in r}
            record["relative_l2"] = r["field_error"]["relative_l2"]
            writer.writerow(record)
    unique_passed = sum(sum(r["checks"].values()) for r in selected.values())
    unique_total = sum(len(r["checks"]) for r in selected.values())
    lines = [
        "# Results of the first restricted verification implementation", "",
        "Generated from saved numerical data on 2026-09-07. This is a reproducible research prototype, not a completed novel universal materials method.",
        "", f"All nine benchmark cases execute. The selected runs pass {unique_passed}/{unique_total} implemented numerical checks. Passing these checks does not imply that every requested physical capability is implemented.",
        "", "## Matched performance comparisons", "",
        "Each timing is the median of three fresh-process runs on the same machine with one numerical thread. Solver and adaptation time are included; initialization, diagnostics, I/O, and plotting are separate in the JSON reports. Physical interface width, domain, coefficients, final time, and base fine spacing are identical within each pair.",
        "", "| Case | Reference time (s) | Adaptive time (s) | Runtime ratio | DOF ratio | Relative field L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['case']}: {r['reference']} | {r['reference_runtime_s']:.4f} | {r['adaptive_runtime_s']:.4f} | {r['speedup']:.2f} | {r['dof_ratio']:.2f} | {r['field_error']['relative_l2']:.3g} |")
    lines += ["", "A ratio above one favors the adaptive variant. The Cahn-Hilliard case uses a full-resolution fallback because the domain is active. Case 9 compares sparse dislocations with uniform versus adaptive fields; it is not an all-field dislocation-PF comparison.",
              "", "Peak RSS includes the interpreter, imports, solver temporaries, and the occasional common-grid transfer audit. Persistent numerical-array storage is reported separately; it must not be presented as total peak memory.",
              "", "| Case | Reference peak RSS (MiB) | Adaptive peak RSS (MiB) | Persistent array-storage ratio |",
              "| --- | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['reference_peak_rss_bytes']/2**20:.2f} | {r['adaptive_peak_rss_bytes']/2**20:.2f} | {r['state_storage_ratio']:.2f} |")
    lines += ["", "## Recovery, conservation, and convergence", "",
              f"- AC radius-squared slope: {selected[1]['metrics']['radius_squared_slope']:.6f}, versus the finite-width curvature approximation -2.",
              f"- CH maximum recorded mass drift: {selected[2]['metrics']['mass_drift']:.3g}; the finite-time coarsening exponent is {selected[2]['metrics']['finite_time_exponent']:.3f}, not an established asymptotic exponent.",
              f"- Grain mean junction-sector error: {selected[3]['metrics']['junction_mean_abs_error_degrees']:.2f} degrees; {selected[3]['metrics']['grain_eliminations']} grains fall below the declared area threshold. These are transient/threshold diagnostics.",
              f"- Coherent elastic-equilibrium residual: {selected[4]['metrics']['elastic_equilibrium_residual']:.3g}; separately optimized structural-force RMS: {selected[4]['metrics']['equilibrium_force_rms']:.3g}.",
              f"- Absorbing-sink profile relative L2: {selected[5]['metrics']['final_relative_l2']:.3g}; combined domain-plus-sink balance error: {selected[5]['metrics']['mass_plus_sink_balance_error']:.3g}.",
              f"- Screw far-field relative error for r >= 10a: {selected[6]['metrics']['far_field_relative_error']:.3g}; equal-sign repulsion and charge-balanced annihilation are observed.",
              "- Pore growth/shrinkage follows high/low vacancy inventory, with conserved combined inventory and decreasing recorded energy.",
              f"- Coupled run mass drift: {selected[9]['metrics']['inventory_drift']:.3g}; final plastic shear: {selected[9]['metrics']['final_plastic_shear']:.6g}; accepted grid transfers: {selected[9]['metrics']['accepted_transfers']}.",
              "", "| Study | Observed spatial orders | Observed time orders |", "| --- | --- | --- |"]
    for name, s in convergence["studies"].items():
        lines.append(f"| {name.upper()} | {', '.join(f'{v:.2f}' for v in s['spatial_orders'])} | {', '.join(f'{v:.2f}' for v in s['temporal_orders'])} |")
    lines += ["", "The spatial comparison uses reference cell averages on each coarse grid. AC spatial runs also reduce dt proportional to h squared, so those rates combine spatial error with a smaller first-order time error. Reference-solution error can inflate observed orders. No universal order claim follows from these few grids.",
              "", "## Coverage and unresolved requirements", "",
              "| Case | Implemented evidence | Remaining requirement |", "| --- | --- | --- |"]
    coverage = {
        1: ("Circular motion, profile/area diagnostics, fixed-width uniform comparison and refinement", "General anisotropic motion and full BMVR allocation ablation"),
        2: ("Spinodal separation, mass/energy, structure factors, finite-time length evolution and refinement", "Asymptotic coarsening exponent and late-stage adaptive savings"),
        3: ("Multiphase migration, simplex, areas, junction-sector estimates and small-grain loss", "Converged triple angles, statistical kinetics and adaptive comparison"),
        4: ("Plane-strain fields, energy derivatives, anisotropy, two-precipitate interaction and stationary solve", "Shape convergence, heterogeneous elasticity and adaptive comparison"),
        5: ("Analytical planar sink and inventory balance with spatial refinement", "Dislocation/pore-specific sink strength and material calibration"),
        6: ("Straight screws, forces, repulsion, annihilation, linear drag and fixed-core quadrature", "Curved lines, multiplication, general crystallography and material core calibration"),
        7: ("Reciprocal binding to a fixed diffuse interface under applied drive", "**Absorption, crystallographic transmission/reflection and residual Burgers content remain disabled**"),
        8: ("Closed vacancy/pore inventory, growth/shrinkage, surface energy and dissipation", "Quantitative capillary/growth-law convergence, gas/elastic effects and adaptive comparison"),
        9: ("Three defect-dimensionality representations, reciprocal binding, diffusion, elastic pair interaction, mean plastic relaxation and coarse bulk grids", "Full local crystal plasticity, curved-line topology, 3D validation and the joint residual-per-cost allocation policy")}
    for k, (done, missing) in coverage.items():
        lines.append(f"| {k} | {done} | {missing} |")
    lines += ["", "The ten original acceptance criteria are supported only for these restricted models. The full requested acceptance is **not established**. In particular, novelty, general topology, finite-strain thermomechanics, fracture propagation, material-specific calibration, and superiority to an independent adaptive-PF baseline remain open.",
              "", "No implemented check was relaxed to force a positive performance result. The CH time step was reduced after the temporal refinement study; both reference and fallback use the same smaller step. The grain potential was corrected to a multi-obstacle formulation after spurious absent-phase fractions were observed; the theory review and regression test record that correction. The earlier first-pass results are retained separately.",
              "", "## Artifacts and reproduction", "",
              "- [Theory and derivations](unified_multiscale_defect_framework.md)",
              "- [Internal theory review](THEORY_REVIEW.md)",
              "- [All nine cases](benchmarks/run.py)",
              "- [Convergence studies](benchmarks/convergence.py)",
              "- [Numerical reports and comparisons](results/verification/summary.json)",
              "- [Convergence data](results/convergence/convergence.json)",
              "- [Performance table](results/performance.csv)",
              "- [Verification overview](results/figures/verification-overview.png)",
              "- [Performance figure](results/figures/performance.png)",
              "- [Convergence figure](results/figures/convergence.png)",
              "", "See README.md for installation and commands. Each case folder includes report.json and fields.npz. Every plotted value is generated from these saved results.",
              "", f"Figure generation took {plotting_s:.3f} s and is excluded from solver timings.", ""]
    write_doc("RESULTS.md","\n".join(lines))
    print(f"Saved three figure sets, performance.csv, and RESULTS.md; {unique_passed}/{unique_total} implemented checks.", flush=True)


if __name__ == "__main__":
    main()
