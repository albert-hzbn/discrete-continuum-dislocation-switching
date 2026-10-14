"""Sections 97-98 figures: creep cavitation coupled to fracture in 2D (SI) and in 3D copper (main text).

Run from the repository root:
    python manuscript/figures/make_creep_fig.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures import AQUA, BLUE, INK, MUTED, ORANGE, label, save

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / "results" / "creep-cavitation-2026-09-30"
R3 = ROOT / "results" / "creep-cavitation-3d-2026-09-30"
MU_773 = 4.21e10*(1 - 0.54*(773 - 300)/1356)          # copper shear modulus at 773 K (Frost and Ashby 1982)


def damage_map(ax, d, extent, title):
    im = ax.imshow(d.T, origin="lower", extent=extent, cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=9)
    ax.grid(False)
    return im


def fig2d():
    rep = json.loads((R2 / "report.json").read_text())
    S = np.array(rep["stresses"])
    fA = np.load(R2 / "A_0.016_64.npz"); fC = np.load(R2 / "C_0.016_64.npz")
    iA = len(fA["t"]) - 2                                   # last frame before rupture
    iC = int(np.argmin(np.abs(fC["t"] - fA["t"][iA])))
    t = fA["t"][iA]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), layout="constrained")
    ext = (0, 1, 0, 1)
    d = 1 - (1 - fA["q"][iA])*(1 - fA["p"][iA])
    ax = axes[0, 0]
    im = damage_map(ax, d, ext, rf"coupled (A), $t={t:.2f}$")
    cb = fig.colorbar(im, ax=ax, shrink=.9); cb.set_label(r"$1-(1-q)(1-p)$")
    label(ax, "(a)")
    ax = axes[0, 1]
    sig = fA["sigma"][iA]/.016
    im = ax.imshow(sig.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-16, vmax=16, interpolation="nearest")
    ax.set_title(rf"coupled (A), $\sigma_{{yy}}/S$ at $t={t:.2f}$", loc="left", fontsize=9); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=.9); cb.set_label(r"$\sigma_{yy}/S$")
    label(ax, "(b)")
    ax = axes[1, 0]
    im = damage_map(ax, fC["p"][iC], ext, rf"fixed stress (C), $t={fC['t'][iC]:.2f}$")
    cb = fig.colorbar(im, ax=ax, shrink=.9); cb.set_label(r"$p$")
    label(ax, "(c)")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_ylim(.35, .65); ax.set(xlabel=r"$x/L$", ylabel=r"$y/L$")
    ax = axes[1, 1]
    ax.plot(S, rep["t_f_A"], "o-", color=BLUE, label="coupled (A): crack")
    ax.plot(S, rep["t_c_C"], "s--", color=ORANGE, label="fixed stress (C): coalescence")
    ax.plot([.016], [rep["t_f_A_n96"]], "D", color=AQUA, markersize=7, label=r"A on $96^2$ cells")
    ax.set(xscale="log", yscale="log", xlabel=r"applied stress $S/\mu$", ylabel=r"failure time $t_f$")
    ax.set_xticks(S); ax.set_xticklabels([f"{s:g}" for s in S], fontsize=7.5)
    ax.set_ylim(1.8, 30); ax.set_yticks([2, 5, 10, 20]); ax.set_yticklabels(["2", "5", "10", "20"]); ax.minorticks_off()
    ax.legend(fontsize=8, loc="lower left")
    label(ax, "(d)")
    save(fig, "fig_creep2d")
    print("wrote fig_creep2d")


def fig3d(S=30, n=48):
    from PIL import Image
    tu = 5.6455e4/3600*1.0 if False else None
    rows = {}
    for m in ("A", "A0", "C"):
        for st in (20, 30, 45, 60):
            f = R3/f"{m}_{st}_{n}.json"
            if f.exists():
                rows[(m, st)] = json.loads(f.read_text())
    fig = plt.figure(figsize=(7.2, 4.9), layout="constrained")
    g = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    im = Image.open(Path(__file__).parent/"fig_creep3d_render.png").convert("RGB")
    from PIL import ImageChops
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox(); im = im.crop(bbox)
    ax_r = fig.add_subplot(g[:, 0]); ax_r.imshow(im); ax_r.set_axis_off()
    label(ax_r, "(a)")
    b = fig.add_subplot(g[0, 1])
    for m, col, mk, lab in (("C", ORANGE, "s", "fixed stress: coalescence"), ("A0", AQUA, "^", "no damage: coalescence"),
                            ("A", BLUE, "o", "coupled: rupture")):
        st = [k[1] for k in rows if k[0] == m and rows[k]["t_f_hours"]]
        if st:
            b.plot(st, [rows[(m, x)]["t_f_hours"] for x in st], mk + "-", color=col, label=lab)
        cens = [k[1] for k in rows if k[0] == m and not rows[k]["t_f_hours"]]
        for x in cens:                                          # no rupture within t_max: lower bound
            tmax = rows[(m, x)]["history"][-1]["t_hours"]
            b.plot([x], [tmax], mk, color=col, mfc="white"); b.annotate("", (x, tmax*1.35), (x, tmax*1.02),
                                                                          arrowprops=dict(arrowstyle="-|>", color=col, lw=1))
    b.set(xscale="log", yscale="log", xlabel=r"applied stress $S$ (MPa)", ylabel="time to failure (h)", ylim=(90, 700))
    b.set_xticks([20, 30, 45, 60]); b.set_xticklabels(["20", "30", "45", "60"]); b.minorticks_off()
    b.set_yticks([100, 200, 400]); b.set_yticklabels(["100", "200", "400"]); b.minorticks_off()
    b.legend(fontsize=7, loc="lower left")
    label(b, "(b)")
    c = fig.add_subplot(g[1, 1])
    for m, col, ls, lab in (("A", BLUE, "-", "coupled"), ("C", ORANGE, "--", "fixed stress")):
        if (m, S) in rows:
            h = rows[(m, S)]["history"]
            c.plot([x["t_hours"] for x in h], [x["area_fraction"] for x in h], ls, color=col, label=lab + ", cavitated area")
            if m == "A":
                c.plot([x["t_hours"] for x in h], [x.get("max_q", 0) for x in h], ":", color=INK, lw=1.4, label="coupled, max damage $q$")
    c.axhline(np.pi/4, color=MUTED, lw=1, ls="-.")
    c.set(xlabel="time (h)", ylabel="fraction", ylim=(0, 1.45))
    c.legend(fontsize=6.8, loc="upper left")
    label(c, "(c)")
    save(fig, "fig_creep3d")
    print("wrote fig_creep3d", sorted(rows))


if __name__ == "__main__":
    import sys
    if "3d" in sys.argv:
        fig3d()
    else:
        fig2d()
