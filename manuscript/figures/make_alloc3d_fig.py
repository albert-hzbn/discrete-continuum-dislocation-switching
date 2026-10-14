"""Section 99 figure: 3D allocator test with ExaDiS (copper, loop-rich wall and channel), from results/allocator-3d-2026-09-30.

Run from the repository root:
    python manuscript/figures/make_alloc3d_fig.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from make_figures import AQUA, BLUE, INK, MUTED, ORANGE, label, save

ROOT = Path(__file__).resolve().parents[2]
RA = ROOT / "results" / "allocator-3d-2026-09-30"


def load(tag):
    p = RA / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    alpha = json.loads((RA / "calibration.json").read_text())["alpha"]
    seeds = [s for s in (1, 2, 3) if load(f"D_s{s}") is not None]
    fig = plt.figure(figsize=(7.2, 5.4), layout="constrained")
    g = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ax_img = fig.add_subplot(g[0, :])
    ax_img.imshow(Image.open(Path(__file__).parent / "fig_alloc3d_render.png"))
    ax_img.set_axis_off()
    label(ax_img, "(a)")
    a, b, c = (fig.add_subplot(g[1, k]) for k in range(3))
    for s, ls in zip(seeds, ("-", "--", ":")):
        for tag, col, lab in ((f"D_s{s}", INK, "full DDD"), (f"H_s{s}_a{alpha:g}", BLUE, "allocator"), (f"0_s{s}", ORANGE, "no wall")):
            r = load(tag)
            if r is None:
                continue
            log = np.array(r["log"], float)
            a.plot(1e9*log[:, 0], 1e5*log[:, 2], ls=ls, color=col, lw=1.6, label=lab if s == seeds[0] else None)
            b.plot(1e9*log[:, 0], 1e5*log[:, 3], ls=ls, color=col, lw=1.6)
    a.set(xlabel="time (ns)", ylabel=r"channel $\varepsilon^p_{zz}$ ($10^{-5}$)"); a.legend(fontsize=7.5)
    b.set(xlabel="time (ns)", ylabel=r"wall-region $\varepsilon^p_{zz}$ ($10^{-5}$)")
    label(a, "(b)"); label(b, "(c)")
    cost = {k: [load(f"{t}_s{s}" if k != "H" else f"H_s{s}_a{alpha:g}")["wall_s"] for s in seeds]
            for k, t in (("D", "D"), ("H", "H"), ("0", "0"))}
    names, cols = ["full DDD", "allocator", "no wall"], [INK, BLUE, ORANGE]
    vals = [np.mean(cost[k]) for k in ("D", "H", "0")]
    c.bar(range(3), vals, color=cols, width=.6)
    for i, v in enumerate(vals):
        c.text(i, v*1.35, f"{v:.0f} s" if v >= 10 else f"{v:.2g} s", ha="center", va="bottom", fontsize=7.5)
    c.set(yscale="log", xticks=range(3), ylabel="wall time per ns (s, 16 threads)", ylim=(0.5, max(vals)*30))
    c.set_xticklabels(names, fontsize=7.5); c.grid(axis="x", visible=False); c.minorticks_off()
    label(c, "(d)")
    save(fig, "fig_alloc3d")
    print("wrote fig_alloc3d", seeds)


if __name__ == "__main__":
    main()
