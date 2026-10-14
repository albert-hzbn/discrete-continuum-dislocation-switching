"""Render continuation figures from saved data without rerunning simulations."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#555b66", "#267d65", "#b27830", "#4265aa"]


def render(folder):
    started = perf_counter()
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    summary = json.loads((folder / "allocation-summary.json").read_text())
    reactions = json.loads((folder / "reactions.json").read_text())
    strategies = ("uniform", "geometry", "residual", "cost")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(figures / f"{name}.{suffix}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for i, strategy in enumerate(strategies):
        rows = [r for r in summary["reports"] if r["strategy"] == strategy]
        # Select the median-total run so stacked components sum to its measured total.
        row = sorted(rows, key=lambda r: r["timings"]["evolution_and_allocation_s"])[len(rows) // 2]
        solver, allocation = (row["timings"][k] for k in ("solver_s", "allocation_s"))
        axes[0].bar(i, solver, color=COLORS[i], label="Solver" if i == 0 else None)
        axes[0].bar(i, allocation, bottom=solver, color=COLORS[i], alpha=.35,
                    hatch="///", label="Allocation and audits" if i == 0 else None)
        totals = [r["timings"]["evolution_and_allocation_s"] for r in rows]
        axes[0].scatter(np.full(len(totals), i), totals, color="black", s=12, zorder=4)
        with np.load(folder / row["folder"] / "fields.npz") as data:
            axes[1].step(data["time"], 2 * data["cells"] + 2, where="post",
                         label=strategy, color=COLORS[i], linestyle=":" if strategy == "residual" else "-")
    axes[0].set(xticks=range(4), xticklabels=strategies, ylabel="Seconds", title="Median run components; dots show all 3 totals")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Time", ylabel="Active scalar unknowns", title="Cost and residual meshes coincide")
    axes[1].legend(fontsize=8)
    fig.suptitle("Matched coupled benchmark: allocation overhead is included")
    save(fig, "allocation-cost")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, strategy in zip(axes, ("uniform", "geometry", "cost")):
        row = next(r for r in summary["reports"] if r["strategy"] == strategy)
        with np.load(folder / row["folder"] / "fields.npz") as data:
            mesh = ax.pcolormesh(data["xe"], data["ye"], data["population"] - .2,
                                 cmap="magma", vmin=0, vmax=.018, shading="flat", rasterized=True)
            ax.vlines(data["xe"], 4, 28, color="white", alpha=.25, lw=.35)
            ax.hlines(data["ye"], 4, 28, color="white", alpha=.25, lw=.35)
        ax.set(xlim=(4, 28), ylim=(4, 28), xlabel="x", ylabel="y", title=strategy, aspect="equal")
    fig.colorbar(mesh, ax=axes, label="Population excess, c − 0.2", shrink=.8)
    fig.suptitle("Chemical feature away from the structural interface, t = 4")
    save(fig, "chemical-mesh")

    with np.load(folder / "reactions.npz") as data:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        for i, row in enumerate(reactions["scenarios"]):
            label = row["name"].replace("_favored", " favored")
            axes[0, 0].bar(np.arange(3) + (i - 1) * .25,
                           row["first_event_probabilities"][1:], width=.25, label=label, color=COLORS[i + 1])
            axes[0, 0].scatter(np.arange(3) + (i - 1) * .25, row["sampled_probabilities"][1:],
                               color="black", s=12, zorder=4)
            axes[1, 1].plot(data["time"], data[row["name"] + "_ensemble_energy"], label=label, color=COLORS[i + 1])
        axes[0, 0].set(xticks=range(3), xticklabels=reactions["state_names"][1:], ylabel="Probability",
                       title="First event: exact bars, sampled dots")
        axes[0, 0].legend(fontsize=8)
        mesh = axes[0, 1].pcolormesh(data["angles"], data["drives"], data["first_event_sweep"][:, :, 1],
                                    cmap="viridis", vmin=0, vmax=1, shading="auto", rasterized=True)
        axes[0, 1].set(xlabel="Misorientation (degrees)", ylabel="Drive work / energy unit", title="Default-barrier transmission probability")
        fig.colorbar(mesh, ax=axes[0, 1], label="Probability")
        axes[1, 0].plot(data["angles"], np.linalg.norm(data["transmission_residuals"], axis=1), label="Vector ledger")
        axes[1, 0].plot(data["angles"], 2 * np.sin(np.deg2rad(data["angles"]) / 2), "--", label="2 sin(θ / 2)")
        axes[1, 0].set(xlabel="Misorientation (degrees)", ylabel="Residual |b| / incident |b|", title="Conservation at transmission")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].set(xlabel="Time / model time unit", ylabel="Ensemble free energy / energy unit", title="Prescribed finite-width encounter")
        axes[1, 1].legend(fontsize=8)
        fig.suptitle("Synthetic interface reaction closure; no material calibration")
        save(fig, "interface-reactions")
    elapsed = perf_counter() - started
    (figures / "plotting.json").write_text(json.dumps({"plotting_and_figure_io_s": elapsed,
                                                     "source": str(folder.relative_to(ROOT))}, indent=2))
    print(f"Saved three PNG/SVG figure pairs in {figures}; {elapsed:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results/continuation-2026-09-12")
    render(parser.parse_args().input.resolve())
