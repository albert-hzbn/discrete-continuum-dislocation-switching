"""Section 95 figure: accuracy against cost on the copper cell structure, from results/realistic-pareto-2026-09-30.

Run from the repository root:
    python manuscript/figures/make_pareto_fig.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import AQUA, BLUE, INK, MUTED, ORANGE, save

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "results" / "realistic-pareto-2026-09-30"


def main():
    rows = json.loads((V / "report.json").read_text())["rows"]
    D = sorted([r for r in rows if r["method"] == "D" and r["k"] > 1], key=lambda r: r["k"])
    D1 = next(r for r in rows if r["method"] == "D" and r["k"] == 1)
    C = next(r for r in rows if r["method"] == "C")
    H = sorted([r for r in rows if r["method"] == "H"], key=lambda r: r["n_c"])
    ms = 1e3
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 6.0), layout="constrained")
    for ax, key, title, letter in ((axes[0], "interior", "Cell interiors (sparse)", "(a)"),
                                   (axes[1], "block", "Cell walls (dense)", "(b)")):
        ax.plot([ms*r["cost"] for r in D], [100*r[key] for r in D], "s-", color=ORANGE,
                label=r"particle DD with superdislocations ($k$ lines lumped)")
        for r in D:
            if r["k"] in (2, 8, 32):
                ax.annotate(rf"$k={r['k']}$", (ms*r["cost"], 100*r[key]), textcoords="offset points", xytext=(5, 7) if r["k"] != 2 else (-4, 9), ha="left" if r["k"] != 2 else "right",
                            fontsize=8.5, color=INK, va="bottom")
        ax.plot([ms*r["cost"] for r in H], [100*r[key] for r in H], "o", color=BLUE, markersize=10,
                markeredgecolor="white", markeredgewidth=1.2, label=r"allocator ($n_c=16,64,256$)", zorder=3)
        ax.plot([ms*C["cost"]], [100*C[key]], "^", color=AQUA, markersize=10, markeredgecolor="white",
                markeredgewidth=1.2, label="continuum DD", zorder=4)
        done = set()
        for r in H:                                   # thresholds with (nearly) identical results share one label
            near = [q for q in H if abs(q["cost"]/r["cost"] - 1) < .15 and abs(q[key]/r[key] - 1) < .15]
            if id(near[0]) in done:
                continue
            done.update(id(q) for q in near)
            ax.annotate(r"$n_c=" + ", ".join(str(q["n_c"]) for q in near) + "$", (ms*r["cost"], 100*r[key]),
                        textcoords="offset points", xytext=(10, 0) if key == "block" else ((0, 11) if len(done) <= 2 else (8, -9)), ha="left" if (key == "block" or len(done) > 2) else "center", va="center" if key == "block" else ("bottom" if len(done) <= 2 else "top"), fontsize=8, color=BLUE)
        ax.axvline(ms*D1["cost"], color=MUTED, lw=1.2, ls=":")
        ax.text(ms*D1["cost"]*0.9, 0.3, "particle DD\n(reference)", color=MUTED, fontsize=8.5, ha="right",
                va="center", transform=ax.get_xaxis_transform())
        ax.set(xscale="log", yscale="log", xlabel="CPU time per step (ms)", ylabel="Error relative to particle DD (%)")
        lo, hi = ax.get_ylim(); ax.set_ylim(lo/2.2, hi*1.8); ax.set_xlim(2.4, ax.get_xlim()[1])
        ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=1, fontsize=8, frameon=False)
    save(fig, "fig_pareto")
    print("wrote fig_pareto")


if __name__ == "__main__":
    main()
