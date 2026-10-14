"""Section 93 validation figures, generated from results/allocator-validation-2026-09-30 only.

Run from the repository root after `python benchmarks/allocator_validation.py analyze`:
    python manuscript/figures/make_validation_figs.py
Outputs fig_allocfields.* (plastic-strain fields against the converged reference) and fig_validation.*.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures import AQUA, BLUE, GRID, INK, MUTED, ORANGE, label, save  # shared style

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "results" / "allocator-validation-2026-09-30"


def gamma(tag):
    return np.load(V / "runs" / f"{tag}.npz")["gamma"]


def fields():
    """D*, H, C plastic strain and the sparse-region error maps of H and C against D*."""
    ref, h, c = gamma("ref-D1024"), gamma("ref-H64"), gamma("ref-C256")
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.85), layout="constrained")
    vmax = float(np.abs(ref).max())
    titles = [r"$D^*$ (1024$^2$)", r"H (256$^2$)", r"C (256$^2$)", r"$|$H$-D^*|$", r"$|$C$-D^*|$"]
    emax = float(max(np.abs(h - ref).max(), np.abs(c - ref).max()))
    for k, (ax, g) in enumerate(zip(axes, [ref, h, c, np.abs(h - ref), np.abs(c - ref)])):
        if k < 3:
            im = ax.imshow(g.T, origin="lower", cmap="Blues", vmin=0, vmax=vmax, extent=(0, 1, 0, 1))
        else:
            im2 = ax.imshow(g.T, origin="lower", cmap="Oranges", vmin=0, vmax=emax, extent=(0, 1, 0, 1))
        for yb in (.2, .8):
            ax.axhline(yb, color=MUTED, lw=.5, ls="--")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titles[k], fontsize=8.5)
    fig.colorbar(im, ax=axes[:3], shrink=.8, label="plastic strain", pad=.01)
    fig.colorbar(im2, ax=axes[3:], shrink=.8, label="abs. difference", pad=.01)
    save(fig, "fig_allocfields")


def validation():
    rep = json.loads((V / "report.json").read_text())
    rows = rep["V3"]["rows"]; N = np.array([r["n_dense"] for r in rows], float)
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.8), layout="constrained")

    a = ax[0, 0]
    for key, col, mk, name in (("cost_D", ORANGE, "s", "discrete D"), ("cost_C", AQUA, "^", "continuum C"), ("cost_H", BLUE, "o", "allocator H")):
        y = 1e3*np.array([r[key] for r in rows]); a.plot(N, y, mk + "-", color=col, label=name)
    a.legend(loc="upper left", fontsize=8.5)
    a.set(xscale="log", yscale="log", xlabel="Dense dislocations $N$", ylabel="CPU time per step (ms)", xlim=(3e3, 2e6))
    label(a, "(a)")

    b = ax[0, 1]
    b.plot(N, [100*r["H_err"]["sparse"] for r in rows], "o-", color=BLUE)
    b.plot(N, [100*r["C_err"]["sparse"] for r in rows], "^-", color=AQUA)
    b.text(N[-1]*1.4, 100*rows[-1]["H_err"]["sparse"], "H", fontsize=8.5, va="center")
    b.text(N[-1]*1.4, 100*rows[-1]["C_err"]["sparse"], "C", fontsize=8.5, va="center")
    b.axhline(1.0, color=MUTED, lw=.8, ls=":"); b.text(N[0], 1.1, "declared 1% gate", color=MUTED, fontsize=8.5, va="bottom")
    b.set(xscale="log", xlabel="Dense dislocations $N$", ylabel="Sparse-region error vs D (%)", xlim=(3e3, 3e6))
    label(b, "(b)")

    c = ax[1, 0]
    seeds = [r["seed"] for r in rep["V5"]["rows"]]; x = np.arange(len(seeds)); w = .38
    c.bar(x - w/2, [100*r["H"]["sparse"] for r in rep["V5"]["rows"]], w, color=BLUE, label="allocator H")
    c.bar(x + w/2, [100*r["C"]["sparse"] for r in rep["V5"]["rows"]], w, color=AQUA, label="continuum C")
    c.set_xticks(x, [str(s) for s in seeds]); c.set(xlabel="Ensemble seed", ylabel="Sparse-region error vs D (%)")
    c.set_ylim(0, 3.0); c.legend(loc="upper center", ncol=2, fontsize=8.5); c.grid(axis="x", visible=False)
    label(c, "(c)")

    d = ax[1, 1]
    parts = ("stress", "particle_grid", "transport", "allocation", "rest")
    names = ("stress solve", "particle–grid", "transport", "allocation", "other")
    cols = (MUTED, ORANGE, AQUA, BLUE, GRID)
    for i, key in enumerate(("D", "H")):
        base, total = 0., rep["V4"][key]["total"]
        for p, n, col in zip(parts, names, cols):
            v = 100*max(rep["V4"][key][p], 0.)/total
            d.barh(i, v, left=base, height=.55, color=col, edgecolor="white", linewidth=.8, label=n if i == 0 else None)
            base += v
        d.text(103, i, f"{1e3*total:.1f} ms", fontsize=8.5, va="center")
    d.set_yticks([0, 1], ["discrete D", "allocator H"]); d.set(xlabel="Share of time per step (%)", xlim=(0, 118), ylim=(-.6, 2.1))
    d.legend(fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(.45, 1.02), columnspacing=0.8, handlelength=1.2); d.grid(axis="y", visible=False)
    label(d, "(d)")
    save(fig, "fig_validation")


if __name__ == "__main__":
    fields()
    validation()
    print("wrote fig_allocfields, fig_validation")
