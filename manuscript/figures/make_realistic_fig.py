"""Section 94 figure: dislocation cell structure in copper, from results/realistic-copper-2026-09-30
and the PFDD estimate of results/pfdd-estimate-2026-09-30.

Run from the repository root:
    python manuscript/figures/make_realistic_fig.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures import AQUA, BLUE, INK, MUTED, ORANGE, save

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "results" / "realistic-copper-2026-09-30"


def title(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", fontsize=9.5)


def main():
    rep = json.loads((V / "report.json").read_text())
    pfdd = json.loads((ROOT / "results" / "pfdd-estimate-2026-09-30" / "report.json").read_text())
    f = np.load(V / "fields.npz")
    L = rep["microstructure"]["L_m"]; L_um = L*1e6; b = rep["material"]["b_m"]
    G = f["gamma_D"].shape[0]
    gD, gC, gH = ((b/L)*f[k] for k in ("gamma_D", "gamma_C", "gamma_H"))   # plastic shear strain

    fig = plt.figure(figsize=(7.0, 6.2), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 0.95])
    ax_a, ax_b, ax_c = top.subplots(1, 3, gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
    W = 15.0                                   # zoom window (um): three walls and two cells
    im = ax_a.imshow(1e6*gD.T, origin="lower", extent=(0, L_um, 0, L_um), cmap="Blues", vmin=0,
                     vmax=np.percentile(1e6*gD, 99.5), interpolation="nearest")
    cont = f["cont"]; nb = cont.shape[0]; bs = L_um/nb
    rows = np.where(cont.any(axis=0))[0]
    for j in rows:
        if j*bs < W:
            ax_a.add_patch(plt.Rectangle((0, j*bs), W, bs, fill=False, ec=ORANGE, lw=1.0, ls="--"))
    ax_a.set(xlim=(0, W), ylim=(0, W), xlabel=r"$x$ ($\mu$m)", ylabel=r"$y$ ($\mu$m)")
    ax_a.grid(False)
    cb = top.colorbar(im, ax=ax_a, shrink=.9, pad=.02)
    cb.set_label(r"$\gamma_{\mathrm{D}}\ (10^{-6})$")
    title(ax_a, "(a) Strain, D")

    y = (np.arange(G) + .5)/G*L_um
    sel = y < W
    ax_b.plot(y[sel], gD.mean(axis=0)[sel], color=INK, lw=2.4, label="D")
    ax_b.plot(y[sel], gC.mean(axis=0)[sel], color=AQUA, lw=1.8, ls="--", label="C")
    ax_b.plot(y[sel], gH.mean(axis=0)[sel], color=BLUE, lw=1.6, ls=":", label="H")
    ax_b.set(yscale="log", xlabel=r"$y$ ($\mu$m)", ylabel=r"$\langle\gamma\rangle_x$", xlim=(0, W), ylim=(8e-10, 2e-4))
    ax_b.legend(loc="upper center", ncol=3, fontsize=8.5, handlelength=1.8, columnspacing=1.0)
    title(ax_b, "(b) Strain across two cells")

    for g, col, ls, name in ((gC, AQUA, "-", "C"), (gH, BLUE, "-", "H")):
        ax_c.plot(y[sel], np.abs(g - gD).mean(axis=0)[sel], color=col, lw=1.8, ls=ls, label=name)
    ax_c.set(yscale="log", xlabel=r"$y$ ($\mu$m)", ylabel=r"$\langle|\gamma-\gamma_{\mathrm{D}}|\rangle_x$", xlim=(0, W), ylim=(5e-12, 3e-7))
    ax_c.legend(loc="upper center", ncol=2, fontsize=8.5)
    title(ax_c, "(c) Deviation from D")

    ax_d, ax_e = bottom.subplots(1, 2, gridspec_kw=dict(width_ratios=[2.3, 1]))
    cost = dict(rep["cpu_per_step_s"])
    cost["PFDD"] = pfdd["estimate"]["h=2b"]["cpu_per_step_s"]
    names = ["D", "C", "H", "DDD_direct_extrapolated", "PFDD"]
    shown = ["particle DD (D)", "continuum DD (C)", "allocator (H)", "direct-sum DDD*", r"PFDD, $h=2b$*"]
    faces = [ORANGE, AQUA, BLUE, "white", "white"]
    edges = [ORANGE, AQUA, BLUE, MUTED, MUTED]
    vals = [1e3*cost[n] for n in names]
    bars = ax_d.barh(range(5), vals, color=faces, edgecolor=edges, height=.62)
    for bar in bars[3:]:
        bar.set_hatch("///")
    for i, v in enumerate(vals):
        ax_d.text(v*1.3, i, f"{v:.3g} ms" if v < 1e4 else f"{v/1e3:.0f} s", va="center", fontsize=8.5, color=INK)
    ax_d.set_xscale("log"); ax_d.set_yticks(range(5), shown); ax_d.invert_yaxis()
    ax_d.set(xlabel="CPU time per step (ms)", xlim=(1, 3e8)); ax_d.grid(axis="y", visible=False)
    ax_d.text(1.0, 1.02, "* estimated from measured kernels", transform=ax_d.transAxes, ha="right", va="bottom",
              fontsize=8, color=MUTED)
    title(ax_d, "(d) Cost per step")

    err = rep["errors_vs_D"]
    x = np.arange(2); w = .38
    hC = [100*err["C"]["block"], 100*err["C"]["interior"]]
    hH = [100*err["H"]["block"], 100*err["H"]["interior"]]
    ax_e.bar(x - w/2, hC, w, color=AQUA, label="C")
    ax_e.bar(x + w/2, hH, w, color=BLUE, label="H")
    ax_e.text(1 - w/2, hC[1] + .5, f"{hC[1]:.3g}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax_e.text(1 + w/2, hH[1] + .5, f"{hH[1]:.2g}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax_e.set_xticks(x, ["walls", "interiors"])
    ax_e.set(ylabel="Error relative to D (%)", ylim=(0, 16.5))
    ax_e.legend(fontsize=8.5, loc="upper left"); ax_e.grid(axis="x", visible=False)
    title(ax_e, "(e) Accuracy")
    save(fig, "fig_realistic")
    print("wrote fig_realistic")


if __name__ == "__main__":
    main()
