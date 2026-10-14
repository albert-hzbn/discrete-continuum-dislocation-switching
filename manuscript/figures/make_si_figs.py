"""Supporting-information figures, replotted from the saved study results in the manuscript style.

Run from the repository root:
    python manuscript/figures/make_si_figs.py [name ...]

Every quantitative panel is drawn from the study's report JSON. Field maps are drawn from saved fields where they
exist (results/, or results/_regen/ for fields regenerated deterministically for this purpose, checked against the
original reports). Where a study's field file no longer exists, the original rendering of that panel is embedded
unchanged (function orig) and is marked as such in this file.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from make_figures import AQUA, BLUE, GRID, INK, MUTED, ORANGE, save

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
PURPLE, GOLD, PINK, TEAL = "#8a5cc2", "#c9a227", "#d6457a", "#4f9fb8"
CYCLE = [BLUE, ORANGE, AQUA, PURPLE, GOLD, PINK, TEAL]


def J(path):
    return json.loads((R / path).read_text(encoding="utf-8"))


def tag(ax, text, x=-0.2):
    ax.text(x, 1.04, text, transform=ax.transAxes, fontweight="bold", fontsize=10.5, va="bottom")


def tags(fig, axes, labels=None, dx=None):
    """Bold panel letters as left-aligned axes titles, so the layout engine reserves space for them."""
    labels = labels or [f"({c})" for c in "abcdefghijklmnop"]
    for a, lab in zip(axes, labels):
        a.set_title(lab, loc="left", fontweight="bold", fontsize=10.5, pad=5)


def slope_label(ax, x, y, order, dx=0.62, color=MUTED, text=None):
    """Reference triangle-free slope line of the given order through (x[-1], y[-1]) on log axes."""
    x0, y0 = x[-1], y[-1]
    xs = np.array([x0, x0/dx**-1 if dx < 1 else x0*dx])
    xs = np.array([x0, x0*2.2])
    ax.plot(xs, y0*(xs/x0)**order*0.7, ls=":", color=color, lw=1.2)
    ax.text(xs[1], y0*(xs[1]/x0)**order*0.7, text or rf"$\propto h^{{{order}}}$", color=color, fontsize=8, va="top")


def logticks(ax, axis, ticks, fmt="{:g}"):
    """Plain-number major ticks at the given values on a log axis, without minor labels."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
    a.set_minor_formatter(NullFormatter()); a.set_minor_locator(NullLocator())


def orig(ax, src, grid, index, crop=None):
    """Embed one panel of an original report figure whose field data no longer exist (unchanged pixels)."""
    from make_composites import split_panels
    im = Image.open(R / src).convert("RGB")
    panel = split_panels(im, grid)[index] if grid else im
    if crop:
        w, h = panel.size
        panel = panel.crop((int(crop[0]*w), int(crop[1]*h), int(crop[2]*w), int(crop[3]*h)))
    ax.imshow(panel, interpolation="lanczos")
    ax.set_axis_off()


# ----------------------------------------------------------------------------------------------------------------
def pf_convergence():
    c = J("convergence/convergence.json")["studies"]
    s = J("microstructure-2026-09-13/summary.json")["final_junctions"]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.25), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, .95, 1]))
    for name, col, mk in (("ac", BLUE, "o"), ("ch", ORANGE, "s")):
        sp, tp = c[name]["spatial"], c[name]["temporal"]
        ax[0].loglog([r["h"] for r in sp], [r["relative_l2"] for r in sp], mk + "-", color=col, label=name.upper())
        ax[1].loglog([r["dt"] for r in tp], [r["relative_l2"] for r in tp], mk + "-", color=col, label=name.upper())
    h = np.array([.25, 1.0]); ax[0].loglog(h, 3e-4*(h/.25)**2, ":", color=MUTED); ax[0].text(.5, 5e-4, "slope 2", color=MUTED, fontsize=8)
    ax[0].set(xlabel=r"cell width $h$", ylabel=r"relative $L_2$ error")
    logticks(ax[0], "x", [.25, .5, 1])
    ax[1].set(xlabel=r"time step $\Delta t$", ylabel=r"relative $L_2$ error")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8, loc="center right")
    ph = np.load(R / "_regen/micro/junction-128.npz")["phases"]
    pal = np.array([[.16, .47, .84], [.92, .41, .20], [.11, .69, .48]])
    rgb = np.clip(np.einsum("ayx,ac->yxc", ph, pal) + (1 - ph.sum(0))[..., None], 0, 1)
    ax[2].imshow(rgb, origin="lower", extent=(-16, 16, -16, 16))
    t = np.linspace(0, 2*np.pi, 200); ax[2].plot(6*np.cos(t), 6*np.sin(t), "w--", lw=1)
    ax[2].set(xlim=(-8, 8), ylim=(-8, 8), xlabel=r"$x$", ylabel=r"$y$"); ax[2].grid(False)
    hh = [r["h"] for r in s]; e = [r["rings"][-1]["mean_abs_error"] for r in s]
    ax[3].loglog(hh, e, "o-", color=BLUE)
    for r in s:
        ax[3].annotate(rf"$n={r['n']}$", (r["h"], r["rings"][-1]["mean_abs_error"]), textcoords="offset points",
                       xytext=(10, 0) if r["n"] == 128 else (-8, 6), ha="left" if r["n"] == 128 else "right", va="center" if r["n"] == 128 else "bottom", fontsize=8)
    ax[3].set(xlabel=r"grid spacing $h$", ylabel=r"mean $|\theta-120^\circ|$ (deg)")
    logticks(ax[3], "x", [.25, 1/3, .5], "{:.2g}"); logticks(ax[3], "y", [.2, .4, .8])
    tags(fig, ax)
    save(fig, "fig_pf_convergence")


def pf_kinetics():
    r = J("kinetic-laws-2026-09-26/report.json"); c, a, s = r["curvature"], r["anisotropic"], r["stress"]
    p = J("precipitate-interactions-2026-09-13/report.json")
    f = np.load(R / "precipitate-interactions-2026-09-13/fields.npz")
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 5.4), layout="constrained")
    ax = ax.ravel()
    for (name, run), col in zip(c["runs"].items(), CYCLE):
        k = np.array(run["kappa"]); ax[0].plot(k, -np.array(run["V"])/k, ".", ms=2.5, color=col, label=rf"$R_0={name}$")
    k = np.array(c["fine"]["kappa"]); ax[0].plot(k, -np.array(c["fine"]["V"])/k, ".", ms=2.5, color=INK, label=r"$R_0=12$, $h/2$")
    ax[0].axhline(1, color=MUTED, lw=1, ls="--")
    ax[0].set(xlabel=r"curvature $\kappa$", ylabel=r"$-V/(L\kappa)$", ylim=(.985, 1.03))
    ax[0].legend(fontsize=7, markerscale=3, ncol=2, loc="upper left", handletextpad=.2, columnspacing=.6); ax[0].set_ylim(.985, 1.06)
    th = np.linspace(0, 360, 360, endpoint=False)
    for (key, run), col in zip(a["runs"].items(), (BLUE, MUTED, ORANGE)):
        ax[1].plot(th, np.array(run["radii"])/run["a0"] - 1, color=col, lw=1.6, label=f"{float(key):g}")
        if key != "0":
            ax[1].plot(th, -float(key)/14*np.cos(4*np.radians(th)) + a["control_lattice_amplitude"]*np.cos(4*np.radians(th)),
                       ":", color=INK, lw=1.1)
    ax[1].set(xlabel=r"polar angle (deg)", ylabel=r"$r/\bar r-1$", xticks=[0, 180, 360], ylim=(-.028, .045), yticks=[-.02, -.01, 0, .01, .02, .03])
    ax[1].legend(fontsize=7.5, loc="upper center", ncol=3, columnspacing=.8, handlelength=1.2, title=r"$\delta$", title_fontsize=7.5); ax[1].set_ylim(-.028, .055)

    taus = np.array(s["taus"])
    for (N, row), col, mk in zip(s["rows"].items(), CYCLE, "os^"):
        ax[2].plot(taus, row["velocities"], mk, color=col, ms=5, label=rf"$N={N}$")
    ax[2].plot(taus, -.2*taus + s["rows"]["48"]["intercept"], "-", color=INK, lw=1, label=r"$-Mb\tau+v_0$")
    ax[2].set(xlabel=r"applied shear $\tau$", ylabel=r"mean radial velocity $v_r$")
    ax[2].legend(fontsize=7.5, loc="lower left")
    for d, col in zip((9., 12., 15.), CYCLE):
        rows = [q for q in p["scan"] if q["distance"] == d and q["second_tensor_orientation"] == 0.]
        ax[3].plot([np.degrees(q["direction"]) for q in rows], [q["interaction_energy"] for q in rows], "o-", color=col,
                   ms=4.5, label=rf"$d={d:g}$")
    ax[3].axhline(0, color=INK, lw=.8)
    ax[3].set(xlabel="separation direction (deg)", ylabel=r"interaction energy $E_{\mathrm{int}}$")
    ax[3].legend(fontsize=7.5, loc="upper right")
    x, y = f["x"], f["y"]; sxx = f["stress"][0, 0]; v = np.percentile(np.abs(sxx), 99)   # arrays are [y, x]
    im = ax[4].imshow(sxx, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap="RdBu_r", vmin=-v, vmax=v)
    for ph in ("phi1", "phi2"):
        ax[4].contour(x, y, f[ph], levels=[.5], colors=INK, linewidths=1)
    ax[4].set(xlim=(-12, 12), ylim=(-8, 8), xlabel=r"$x$", ylabel=r"$y$"); ax[4].grid(False)
    fig.colorbar(im, ax=ax[4], shrink=.45, pad=.02, aspect=12).set_label(r"$\sigma_{xx}$")
    for orient, col, lab in ((0., BLUE, r"$0^\circ$"), (np.pi/2, ORANGE, r"$90^\circ$")):
        rows = [q for q in p["scan"] if q["direction"] == 0. and q["second_tensor_orientation"] == orient]
        ax[5].plot([q["distance"] for q in rows], [1e3*q["force2"][0] for q in rows], "o-", color=col, label="rotated " + lab)
    ax[5].axhline(0, color=INK, lw=.8)
    ax[5].set(xlabel=r"separation $d$", ylabel=r"force $F_x$ ($10^{-3}$)")
    ax[5].text(.03, .5, "repulsion", transform=ax[5].get_yaxis_transform(), fontsize=8, color=MUTED, va="bottom", ha="left")
    ax[5].text(.03, -.5, "attraction", transform=ax[5].get_yaxis_transform(), fontsize=8, color=MUTED, va="top", ha="left")
    ax[5].legend(fontsize=7.5, loc="upper right", title="eigenstrain 2", title_fontsize=7.5); ax[5].set_ylim(top=24)
    tags(fig, ax)
    save(fig, "fig_pf_kinetics")


def gridcost():
    s = J("performance-complete-2026-09-13/summary.json")["rows"]
    g = J("graded-cost-2026-09-26/report.json")["candidates"]
    a = J("amr-message-passing-2026-09-26/amr/report.json"); m = J("amr-message-passing-2026-09-26/message-passing/report.json")
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.9), layout="constrained")
    ax = ax.ravel()
    for variant, col, mk, lab in (("reference", BLUE, "o", "uniform"), ("adaptive", ORANGE, "s", "adaptive")):
        rows = [r for r in s if r["config"]["kind"] == "ac" and r["config"]["variant"] == variant]
        x = [r["launch_to_exit_s"] for r in rows]; y = [100*r["fine_reference_comparison"]["phase_fraction"]["relative_l2"] for r in rows]
        ax[0].loglog(x, y, mk + "-", color=col, label=lab)
        for r, xx, yy in [q for q in zip(rows, x, y)][-1:]:
            ax[0].annotate(str(r["config"]["n"]), (xx, yy), textcoords="offset points", xytext=(6, 4), ha="left", fontsize=7.5)
    ax[0].axhline(1, ls=":", color=INK, lw=1)
    ax[0].set(xlabel="time to solution (s)", ylabel="phase-fraction error (%)"); ax[0].margins(y=.12)
    ax[0].legend(fontsize=7.5, loc="upper right")
    logticks(ax[0], "x", [.4, .5, .7]); logticks(ax[0], "y", [.3, .5, 1, 1.5])
    fields = ["c", "eta", "T", "slip", "damage"]
    for fam, col, mk in (("uniform", BLUE, "o"), ("graded", ORANGE, "s")):
        keys = [k for k in g if g[k]["family"] == fam]
        w = [max(g[k]["errors"]["t0.2"][f] for f in fields)*100 for k in keys]
        ax[1].loglog([g[k]["wall_seconds"] for k in keys], w, mk + "-", color=col, label=fam)
        for k, e in [q for q in zip(keys, w)][-1:]:
            ax[1].annotate(str(g[k]["n"]), (g[k]["wall_seconds"], e), textcoords="offset points", xytext=(6, 4), ha="left", fontsize=7.5)
    ax[1].set(xlabel="time to solution (s)", ylabel="worst field error (%)"); ax[1].margins(y=.12)
    ax[1].legend(fontsize=7.5, loc="upper right")
    logticks(ax[1], "x", [300, 1000]); logticks(ax[1], "y", [2, 4, 6, 10])
    names, labels = ["uniform", "focused", "amr"], ["uniform", "focused", "AMR"]
    c = a["cost"]
    ax[2].bar(range(3), [c[k]["solver_plus_adaptation_s"] for k in names], color=[BLUE, AQUA, ORANGE], width=.6)
    for i, k in enumerate(names):
        ax[2].text(i, c[k]["solver_plus_adaptation_s"]*1.04 + .008, f"{c[k]['mean_cells']/1e3:.0f}k\ncells", ha="center", va="bottom", fontsize=7)
    ax[2].set(xticks=range(3), ylabel="solver + adaptation time (s)", ylim=(0, max(c[k]["solver_plus_adaptation_s"] for k in names)*1.35))
    ax[2].set_xticklabels(labels, fontsize=8); ax[2].grid(axis="x", visible=False)
    P = [r["parts"] for r in m["strong"]["uniform"]]
    ax[3].plot(P, P, "--", color=MUTED, lw=1, label="ideal")
    for k, col, mk in (("uniform", BLUE, "o"), ("adaptive", ORANGE, "s")):
        ax[3].plot(P, [r["speedup"] for r in m["strong"][k]], mk + "-", color=col, label=k)
    ax[3].set(xlabel="MPI processes", ylabel="strong-scaling speedup", xticks=P)
    ax[3].legend(fontsize=7.5)
    tags(fig, ax, dx=.05)
    save(fig, "fig_gridcost")


def costs2():
    lp = J("local-plasticity-2026-09-26/report.json")["runs"]["uniform"]["rows"]
    tc = J("thermochemical-cost-2026-09-22/report.json")
    rc = J("response-cache-optimized-2026-09-13/summary.json")["comparisons"]
    groups = {key: [x for x in tc["runs"] if x["result"]["degree"] == d] for key, d in (("q1", 1), ("q2", 2))}
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.7), layout="constrained")
    ax = ax.ravel()
    t = [x["time"] for x in lp]
    ax[0].plot(t, [x["mean_stress"] for x in lp], color=INK, label=r"mean stress $\bar\sigma$")
    ax[0].set(xlabel=r"time $t$", ylabel=r"mean shear stress $\bar\sigma$")
    ax[1].plot(t, [x["mean_gamma"] for x in lp], color=AQUA, label=r"continuum slip $\langle\gamma\rangle$")
    ax[1].plot(t, [100*x["plastic_screws"] for x in lp], color=BLUE, ls="--", label=r"$100\times$ screw glide")
    ax[1].set(xlabel=r"time $t$", ylabel="plastic shear strain"); ax[1].legend(fontsize=7.5)
    lab = [r"$Q_1$, $n=48$", r"$Q_2$, $n=24$"]
    parts = ["initialization_seconds", "solver_seconds", "diagnostics_output_seconds", "other"]
    names = ["initialization", "solver", "diagnostics, I/O", "start-up, exit"]
    cols = ["#b6c5d2", BLUE, GOLD, MUTED]
    bottom = np.zeros(2)
    for f, n_, col in zip(parts, names, cols):
        vals = []
        for key in ("q1", "q2"):
            run = sorted(groups[key], key=lambda x: x["launch_to_exit_seconds"])[1]
            vals.append(run["result"][f] if f != "other" else run["launch_to_exit_seconds"] - sum(run["result"][k] for k in parts[:-1]))
        ax[2].bar([0, 1], vals, bottom=bottom, color=col, label=n_, width=.55); bottom += vals
    ax[2].set(xticks=[0, 1], ylabel="time to solution (s)"); ax[2].set_xticklabels(lab); ax[2].grid(axis="x", visible=False)
    ax[2].legend(fontsize=7, loc="upper center", ncol=2, columnspacing=.8); ax[2].set_ylim(0, 76)
    fn = ["c", "eta", "T", "slip", "energy"]; fl = [r"$c$", r"$\eta$", r"$T$", r"$\gamma$", r"$E$"]
    for i, (key, col) in enumerate((("q1", ORANGE), ("q2", BLUE))):
        e = [max(x["accuracy_errors"][n_] for x in groups[key])/(.01 if n_ == "energy" else .05) for n_ in fn]
        ax[3].bar(np.arange(5) + (i - .5)*.36, e, width=.36, color=col, label=lab[i])
    ax[3].axhline(1, color=INK, ls="--", lw=1)
    ax[3].set(xticks=range(5), ylabel="error / allowed error", ylim=(0, 1.35)); ax[3].set_xticklabels(fl)
    ax[3].legend(fontsize=7.5, loc="upper center", ncol=2, columnspacing=.8); ax[3].grid(axis="x", visible=False)
    x = np.arange(len(rc))
    ax[4].bar(x - .18, [r["online_ratio"] for r in rc], .34, color=BLUE, label="evolution + audits")
    ax[4].bar(x + .18, [r["launch_ratio"] for r in rc], .34, color=ORANGE, label="launch to exit")
    ax[4].axhline(1, color=INK, ls=":", lw=1)
    ax[4].set(xticks=x, xlabel=r"grid $n$ per axis", ylabel="full / cached time", ylim=(0, 2.3)); ax[4].set_xticklabels([str(r["n"]) for r in rc])
    ax[4].legend(fontsize=7, loc="upper left", ncol=1); ax[4].grid(axis="x", visible=False)
    ax[5].plot([r["affine_online_s"] for r in rc], [100*r["errors"]["alpha"]["relative_l2"] for r in rc], "o-", color=AQUA)
    for r in rc:
        ax[5].annotate(rf"$n={r['n']}$", (r["affine_online_s"], 100*r["errors"]["alpha"]["relative_l2"]), textcoords="offset points", xytext=(0, 11), ha="center", va="bottom", fontsize=7.5)
    ax[5].axhline(1, color=INK, ls=":", lw=1)
    ax[5].set(xlabel="cached run time (s)", ylabel=r"Nye-tensor error (%)"); ax[5].margins(x=.25, y=.2)
    tags(fig, ax, dx=.06)
    save(fig, "fig_costs2")


def loops():
    cv = J("curved-loops-fcc-2026-09-12/convergence/report.json")
    fr = J("frank-read-elastic-2026-09-28/report.json"); dc = J("disclination-dynamics-2026-09-28/report.json")
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6), layout="constrained")
    ax = ax.ravel()
    rows = cv["field_grid"][:-1]
    ax[0].semilogy([r["n"] for r in rows], [r["relative_energy_error"] for r in rows], "o-", color=BLUE, label="energy")
    ax[0].semilogy([r["n"] for r in rows], [r["relative_force_error"] for r in rows], "s--", color=ORANGE, label="nodal force")
    ax[0].set(xlabel="grid cells per axis", ylabel="relative error"); ax[0].legend(fontsize=7.5)
    rows = cv["polygon"][:-1]
    ax[1].loglog([r["nodes"] for r in rows], [r["relative_energy_error"] for r in rows], "o-", color=BLUE, label="energy")
    ax[1].loglog([r["nodes"] for r in rows], [r["relative_area_error"] for r in rows], "s--", color=ORANGE, label="area")
    logticks(ax[1], "x", [8, 16, 32, 64])
    ax[1].set(xlabel="polygon nodes", ylabel="relative error"); ax[1].legend(fontsize=7.5)
    rows = cv["quadrature"][:-1]
    ax[2].semilogy([r["order"] for r in rows], [r["relative_force_error"] for r in rows], "o-", color=BLUE)
    ax[2].set(xlabel="Gauss points per segment", ylabel="relative force error")
    c, fit, nu = fr["critical"], fr["fit"], .324
    L = np.array([x["L"] for x in c]); y = np.array([x["scaled"] for x in c])
    xx = np.linspace(np.log(40), np.log(500))
    ax[3].plot(np.log(L), y, "o", color=BLUE, label=r"computed $\tau_c$")
    ax[3].plot(xx, fit["slope"]*xx + fit["intercept"], "-", color=BLUE, lw=1.2, label=f"fit, slope {fit['slope']:.3f}")
    for k, lab, col in ((1/(2*np.pi), r"screw $1/2\pi$", ORANGE), (1/(2*np.pi*(1 - nu)), r"edge $1/2\pi(1-\nu)$", AQUA)):
        ax[3].plot(xx, k*(xx - np.log(L[0])) + y[0], ":", color=col, lw=1.4, label=lab)
    ax[3].set(xlabel=r"$\ln(L/a)$", ylabel=r"$\tau_cL/(\mu b)$"); ax[3].legend(fontsize=7)
    ax[4].semilogx(L, [x["ratio"] for x in c], "o-", color=BLUE)
    ax[4].axhline(1, color=MUTED, ls="--", lw=1)
    ax[4].set(xlabel=r"source length $L/a$", ylabel=r"$\tau_c/\tau_{\mathrm{line\ tension}}$")
    logticks(ax[4], "x", [50, 100, 200, 400])
    W = np.array(dc["Q3"]["W_samples"]); stride = max(dc["Q3"]["steps"]//300, 1)
    ax[5].plot(np.arange(len(W))*stride, W, color=PURPLE)
    ax[5].set(xlabel="time step", ylabel=r"pair energy $W$")
    tags(fig, ax, dx=.06)
    save(fig, "fig_loops")


def copper():
    cal = J("calibration-copper-2026-09-27/report.json")["validation"]["rows"]
    ee = J("embryo-atomistics-2026-09-27/report.json")["E_f_eV"]; em = J("embryo-atomistics-mishin-2026-09-28/report.json")["E_f_eV"]
    vl = J("vacancy-loop-2026-09-28/rows.json")
    zj = J("screw-core-2026-09-28/report-zjw.json"); mi = J("screw-core-2026-09-28/report-mishin.json")
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.4), layout="constrained")
    inv = np.array([1e3/r["T"] for r in cal]); band = np.array([r["band"] for r in cal])
    ax[0].fill_between(inv, band[:, 0], band[:, 2], color=BLUE, alpha=.22, lw=0, label="95% band")
    ax[0].plot(inv, band[:, 1], color=BLUE, lw=1.4, label="median")
    ax[0].plot(inv, [r["measured"] for r in cal], "--", color=INK, lw=1.4, label="measured")
    ax[0].set(yscale="log", xlabel=r"$1000/T$ (K$^{-1}$)", ylabel=r"$D_{\mathrm{self}}$ (m$^2$ s$^{-1}$)")
    ax[0].legend(fontsize=7, loc="lower left")
    for E, lab, col, mk in ((ee, "EMT", ORANGE, "s"), (em, "Mishin", BLUE, "o")):
        n = np.array(sorted(int(k) for k in E)); ax[1].plot(n, [E[str(k)]/k**(2/3) for k in n], mk + "-", ms=3, lw=1.2, color=col, label=lab)
    ax[1].set(xscale="log", xlabel=r"vacancies $n$", ylabel=r"$E_f(n)/n^{2/3}$ (eV)"); ax[1].legend(fontsize=7)
    n = [r["n"] for r in vl]
    ax[2].plot(n, [r["E_void"] for r in vl], "o-", color=BLUE, ms=4, label="compact void")
    ax[2].plot(n, [r["E_loop"] for r in vl], "s-", color=ORANGE, ms=4, label="Frank loop")
    ax[2].set(xlabel=r"vacancies $n$", ylabel=r"formation energy (eV)"); ax[2].legend(fontsize=7)
    R = ["60", "120"]
    sep = [mi["core"][r]["separation_A"] for r in R]
    ax[3].bar(range(2), sep, color=BLUE, width=.55, label="atomistic (Mishin)")
    ax[3].axhline(mi["anisotropic_prediction_A"], color=ORANGE, ls="--", lw=1.3, label="anisotropic elasticity")
    ax[3].axhline(mi["isotropic_prediction_A"], color=MUTED, ls=":", lw=1.3, label="isotropic elasticity")
    ax[3].set(xticks=range(2), xlabel=r"cylinder radius $R$ (Å)", ylabel="partial separation (Å)", ylim=(0, 22))
    ax[3].set_xticklabels(R); ax[3].legend(fontsize=6.5, loc="upper left"); ax[3].grid(axis="x", visible=False)
    tags(fig, ax, dx=.055)
    save(fig, "fig_copper")


def fracture2():
    s = J("spatial-fracture-complete-2026-09-22/summary.json")["reports"]
    v = J("vacancy-fracture-2026-09-26/report.json")["homogeneous"]["sink_relaxation"]
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0), layout="constrained")
    ax = ax.ravel()
    curves = [("mesh-24", r"$n=24$"), ("mesh-32", r"$n=32$"), ("mesh-48", r"$n=48$")] + ([("mesh-64", r"$n=64$")] if "mesh-64" in s else [])
    curves += [("load-halfstep", r"$n=32$, half step"), ("heterogeneous", r"$n=32$, tough layer")]
    for (name, lab), col, ls in zip(curves, CYCLE, ["-", "-", "-", "-", "--", ":"]):
        r = s[name]; h = r["history"][:r["ramp_steps"] + 1]
        ax[0].plot([q["load"] for q in h], [q["reaction"] for q in h], color=col, ls=ls, lw=1.5, label=lab)
    ax[0].set(xlabel=r"grip displacement $U$", ylabel=r"reaction force $F$ per thickness"); ax[0].legend(fontsize=7.5, loc="upper right"); ax[0].set_ylim(top=0.075)
    h = s["mesh-32"]["history"]; i = np.arange(len(h)); e0 = s["mesh-32"]["initial_energy"]
    for key, lab, col in (("elastic_energy", "elastic", BLUE), ("fracture_energy", "fracture", ORANGE), ("energy", "total", INK)):
        ax[1].plot(i, [1e3*r[key] for r in h], color=col, lw=1.6, label=lab)
    ax[1].plot(i, [1e3*(r["work_trapezoid"] + e0) for r in h], ":", color=AQUA, lw=1.8, label="external work")
    ax[1].axvline(20, color=MUTED, ls="--", lw=1); ax[1].text(20.6, .97, "unloading", transform=ax[1].get_xaxis_transform(), fontsize=7.5, color=MUTED, ha="left", va="top")
    ax[1].set(xlabel="load step", ylabel=r"energy per thickness ($10^{-3}$)", ylim=(-.05, 2.1)); ax[1].legend(fontsize=6.5, loc="upper left", ncol=2, columnspacing=.6)
    regen = R / "_regen/vacfrac/vacancy-notch-64-0.00025-final.npz"   # regenerated deterministically (same command)
    if regen.exists():
        sys.path.insert(0, str(ROOT / "src"))
        from bmvr.vacancy_fracture import NodalVacancyFracture2D
        eq = J("vacancy-fracture-2026-09-26/report.json")["notch"]["parameters"]["equilibrium"]
        mdl, st, _ = NodalVacancyFracture2D.load(regen)
        n = mdl.mechanics.mesh.n; xe, ye = mdl.mechanics.mesh.edges
        im = ax[2].pcolormesh(xe, ye, (st.vac/eq - 1).reshape(n, n).T, cmap="viridis", rasterized=True)
        ax[2].contour(np.linspace(xe[0], xe[-1], n + 1), np.linspace(ye[0], ye[-1], n + 1), st.damage.reshape(n + 1, n + 1).T,
                      levels=[.5], colors="white", linewidths=1)
        ax[2].set(xlabel=r"$x$", ylabel=r"$y$", aspect="equal", xlim=(0, .7), ylim=(.15, .85)); ax[2].grid(False)
        fig.colorbar(im, ax=ax[2], shrink=.8, pad=.02).set_label(r"$x/x_{\mathrm{eq}}-1$")
    else:
        orig(ax[2], "vacancy-fracture-2026-09-26/figures/vacancy-fracture.png", (1, 3), 0)     # field data no longer saved
    dts = [float(k) for k in v["runs"]]; errs = [v["runs"][k]["error"] for k in v["runs"]]
    ax[3].loglog(dts, errs, "o-", color=BLUE, label="sink relaxation")
    ax[3].loglog(dts, errs[0]*(np.array(dts)/dts[0])**2, "--", color=MUTED, lw=1, label="slope 2")
    ax[3].set(xlabel=r"time step $\Delta t$", ylabel=r"error $|x(T)-x_{\mathrm{exact}}|$"); ax[3].legend(fontsize=7, loc="lower right")
    logticks(ax[3], "x", dts, "{:g}"); logticks(ax[3], "y", [3e-9, 1e-8, 3e-8], "{:.0e}")
    tags(fig, ax, dx=.05)
    save(fig, "fig_fracture2")


def fracture():
    tn = J("transition-elastic-notch-2026-09-27/report.json"); th = J("thermofracture-2026-09-22/report.json")
    fig = plt.figure(figsize=(7.2, 6.4), layout="constrained")
    ax = fig.subplot_mosaic("aabbcc;dddddd;eeefff", height_ratios=[1, .72, .95])
    for (key, v), col in zip(tn["pairs"].items(), CYCLE):
        ts = [(.4 if k == "final" else float(k[1:])) for k in v]
        lab = key.replace("_", "/")
        ax["a"].plot(ts, [100*v[k]["slip"] for k in v], "o-", color=col, ms=4, label=lab + " slip")
        ax["a"].plot(ts, [100*v[k]["damage"] for k in v], "s--", color=col, ms=4, mfc="white", label=lab + " damage")
    ax["a"].axhline(5, color=INK, ls="--", lw=1)
    ax["a"].set(xlabel=r"time $t$", ylabel="successive-mesh discrepancy (%)"); ax["a"].legend(fontsize=6, ncol=1)
    for (n, xr), col in zip(tn["runs"].items(), CYCLE):
        ev = xr["events"]
        ax["b"].step([e["time"] for e in ev], np.cumsum([e["count"] for e in ev]), where="post", color=col, label=rf"$n={n}$ released")
        er = xr.get("erosions", [])
        if er:
            ax["b"].plot([e["time"] for e in er], np.cumsum([len(e["cells"]) for e in er]), "x", color=col, ms=6, label=rf"$n={n}$ eroded")
    ax["b"].set(xlabel=r"time $t$", ylabel="cumulative count"); ax["b"].legend(fontsize=6.5)
    runs = th["runs"]; spatial = [x for x in runs if x["variant"] == "coupled" and x["dt"] == .00025]
    coupled = next(x for x in spatial if x["n"] == 16)
    fields, flab = ["c", "eta", "T", "slip", "damage"], [r"$c$", r"$\eta$", r"$T$", r"$\gamma$", r"$q$"]
    x = np.arange(5)
    ax["c"].bar(x - .18, [100*th["spatial"][k] for k in fields], .36, color=BLUE, label=r"$n=16$ vs $24$")
    ax["c"].bar(x + .18, [100*th["time_fine"][k] for k in fields], .36, color=ORANGE, label=r"$\Delta t$ vs $\Delta t/2$")
    ax["c"].axhline(5, color=INK, ls="--", lw=1); ax["c"].axhline(1, color=MUTED, ls=":", lw=1)
    ax["c"].set(xticks=x, ylabel="relative discrepancy (%)"); ax["c"].set_xticklabels(flab, fontsize=7.5)
    ax["c"].legend(fontsize=6.5); ax["c"].grid(axis="x", visible=False)
    orig(ax["d"], "dynamic-branching-2026-09-27/figures/dynamic-branching-refinement.png", None, 0)   # field data no longer saved
    for run, col, lab, ls, lw in zip([coupled] + [x for x in runs if x["variant"] in ("frozen-damage", "no-phase-resistance")], CYCLE,
                                     ["coupled", "frozen damage", "no phase resistance"], ["-", "-", "--"], [3.2, 1.8, 1.6]):
        ax["e"].plot([r["time"] for r in run["rows"]], [r["damage_integral"] for r in run["rows"]], color=col, ls=ls, lw=lw, label=lab)
    ax["e"].set(xlabel=r"time $t$", ylabel=r"$\int q\,\mathrm{d}V$"); ax["e"].legend(fontsize=7)
    for run, col in zip(spatial, CYCLE):
        E = np.array([r["total_energy"] for r in run["rows"]])
        ax["f"].plot([r["time"] for r in run["rows"]], 1e9*(E - E[0])/run["energy_scale"], color=col, label=rf"$n={run['n']}$")
    ax["f"].set(xlabel=r"time $t$", ylabel=r"$\Delta E/E_s$ ($10^{-9}$)"); ax["f"].legend(fontsize=7)
    tags(fig, [ax[k] for k in "abcdef"], dx=.05)
    save(fig, "fig_fracture")


def finitestrain():
    fe = J("finite-elasticity-complete-2026-09-22/report.json"); tp = J("thermoplastic-2026-09-22/report.json")
    qm = J("quadratic-mechanics-2026-09-22/report.json"); tb = J("thermochemical-2026-09-22/report.json")
    tf = J("thermochemical-refinement-2026-09-22/report.json")
    fig, ax = plt.subplots(3, 3, figsize=(7.2, 6.9), layout="constrained")
    ax = ax.ravel()
    man = fe["manufactured"]; n = np.array([r["n"] for r in man])
    for (key, lab), col, mk in zip((("displacement", r"displacement"), ("gradient", r"$F-I$"), ("piola", r"Piola stress $P$")), CYCLE, "os^"):
        ax[0].loglog(n, [r["errors"][key] for r in man], mk + "-", color=col, label=lab)
    ax[0].set(xlabel="elements per side", ylabel=r"relative $L_2$ error"); ax[0].legend(fontsize=6.8)
    logticks(ax[0], "x", list(n))
    runs = fe["loading"]
    for run, col in zip(runs[:3], CYCLE):
        ax[1].plot([r["load"] for r in run["rows"]], [r["reaction"] for r in run["rows"]], ".-", color=col, ms=4, label=rf"$n={run['n']}$")
    ax[1].set(xlabel="shear amplitude", ylabel="grip reaction"); ax[1].legend(fontsize=7)
    for run, col in zip((runs[1], runs[3]), (BLUE, ORANGE)):
        st = run["increments"] + 1; E = np.array([r["energy"] for r in run["rows"]]); W = np.array(run["work"])
        ax[2].plot([r["load"] for r in run["rows"][:st]], 1e3*(W - E + E[0])[:st], ".-", color=col, ms=4, label=f"{st - 1} increments")
    ax[2].set(xlabel="shear amplitude", ylabel=r"work $-\ \Delta$free energy ($10^{-3}$)"); ax[2].legend(fontsize=7)
    heat = [r for r in tp["runs"] if r["variant"] == "heat"]
    ax[3].loglog([r["n"] for r in heat], [r["continuum_mode_error"] for r in heat], "o-", color=BLUE, label="continuum mode")
    ax[3].loglog([r["n"] for r in heat], [r["discrete_time_error"] for r in heat], "s--", color=ORANGE, label="discrete mode")
    ax[3].set(xlabel="cells per side", ylabel="temperature error"); ax[3].legend(fontsize=7)
    logticks(ax[3], "x", sorted({r["n"] for r in heat}))
    for run, col in zip([r for r in tp["runs"] if r["variant"] == "coupled" and r["n"] == 16], CYCLE):
        t = np.array([r["time"] for r in run["rows"]]); E = np.array([r["total_energy"] for r in run["rows"]])
        ax[4].plot(t, 1e9*(E - E[0])/run["energy_drift_scale"], color=col, label=rf"$\Delta t={run['dt']:g}$")
    ax[4].set(xlabel=r"time $t$", ylabel=r"energy drift ($10^{-9}$, scaled)"); ax[4].legend(fontsize=7)
    for run in tp["runs"]:
        if run["n"] == 16 and run["dt"] == .001 and run["variant"] in ("coupled", "no-slip", "no-conduction"):
            s = np.array([r["entropy"] for r in run["rows"]])
            col = {"coupled": INK, "no-slip": BLUE, "no-conduction": ORANGE}[run["variant"]]
            ax[5].plot([r["time"] for r in run["rows"]], s - s[0], color=col, label=run["variant"].replace("-", " "))
    ax[5].set(xlabel=r"time $t$", ylabel=r"entropy production $\Delta S$"); ax[5].legend(fontsize=7)
    for data, lab, col in ((fe["manufactured"], r"$Q_1$", ORANGE), (qm["manufactured"], r"$Q_2$", BLUE)):
        for key, ls, mk in (("displacement", "-", "o"), ("gradient", "--", "s")):
            ax[6].loglog([x["unknowns"] for x in data], [x["errors"][key] for x in data], mk + ls, color=col, ms=4,
                         label=f"{lab} " + ("$u$" if key == "displacement" else r"$\nabla u$"))
    ax[6].set(xlabel="displacement unknowns", ylabel="relative error"); ax[6].legend(fontsize=6.5, ncol=2, columnspacing=.5)
    labs = [r"$Q_1$ 16/24", r"$Q_1$ 32/48", r"$Q_2$ 16/24"]
    errs = [tb["mesh_comparison"]["slip"], tf["comparison_32_48"]["slip"], qm["mesh_comparison"]["slip"]]
    ax[7].bar(range(3), 100*np.array(errs), color=[ORANGE, ORANGE, BLUE], width=.6)
    ax[7].axhline(5, color=INK, ls="--", lw=1)
    ax[7].set(xticks=range(3), ylabel="slip discrepancy (%)"); ax[7].set_xticklabels(labs, fontsize=7); ax[7].grid(axis="x", visible=False)
    for run, col in zip(qm["runs"], CYCLE):
        E = np.array([x["total_energy"] for x in run["rows"]])
        ax[8].plot([x["time"] for x in run["rows"]], 1e11*(E - E[0])/run["energy_scale"], color=col, label=rf"$Q_2$, $n={run['n']}$")
    ax[8].set(xlabel=r"time $t$", ylabel=r"energy drift ($10^{-11}$, scaled)"); ax[8].legend(fontsize=7)
    tags(fig, ax, dx=.055)
    save(fig, "fig_finitestrain")


def interfaces3d():
    fig = plt.figure(figsize=(7.2, 4.6), layout="constrained")
    top = [fig.add_subplot(2, 4, k + 1, projection="3d") for k in range(4)]
    for ax, branch in zip(top, ("incident", "transmitted", "absorbed", "reflected")):
        g = J(f"interface-graphs-2026-09-12/{branch}/report.json")["final_graph"]
        nodes, edges = np.array(g["nodes"]), np.array(g["edges"])
        yy, zz = np.meshgrid([-1.3, 1.3], [-.55, .55])
        ax.plot_surface(np.zeros_like(yy), yy, zz, color="#9ec3bb", alpha=.25, shade=False)
        for (i, j), kind, b in zip(edges, g["kinds"], g["burgers"]):
            col = PINK if kind else (GOLD if abs(b[2]) > 1e-10 else BLUE)
            ax.plot(nodes[[i, j], 0], nodes[[i, j], 1], nodes[[i, j], 2], color=col, lw=2.6 if kind else 1.8)
        ax.scatter(*nodes.T, color=INK, s=6)
        ax.set(xlim=(-3.2, .65), ylim=(-1.3, 1.3), zlim=(-.55, .55), xticks=[-3, -1.5, 0], yticks=[-1, 0, 1], zticks=[])
        ax.tick_params(labelsize=6, pad=-3)
        ax.view_init(elev=24, azim=-60); ax.set_box_aspect((3.8, 2.6, 1.1))
        ax.set_title(branch, fontsize=8.5, pad=-2)
    conv = J("three-dimensional-2026-09-12/convergence.json")
    bottom = [fig.add_subplot(2, 3, 4 + k) for k in range(3)]
    for key, col, mk in zip(("phase", "population", "slip", "stress"), CYCLE, "os^D"):
        for ax, group, xf in ((bottom[0], "space", "h"), (bottom[1], "time", "dt"), (bottom[2], "space", "solver_s")):
            rows = conv[group]
            ax.loglog([r[xf] for r in rows], [r["errors"][key]["relative_l2"] for r in rows], mk + "-", color=col, ms=4, label=key)
    bottom[0].set(xlabel=r"cell spacing $h$", ylabel=r"relative $L_2$ error"); logticks(bottom[0], "x", [.48, .64, .94])
    bottom[1].set(xlabel=r"time step $\Delta t$", ylabel=r"relative $L_2$ error"); logticks(bottom[1], "x", [.005, .01, .02, .04])
    bottom[2].set(xlabel="solver time (s)", ylabel=r"relative $L_2$ error"); logticks(bottom[2], "x", [.3, .6, 1.4])
    bottom[0].legend(fontsize=6.5)
    fig.text(.01, .985, "(a)", fontweight="bold", fontsize=10.5, va="top")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Line2D([], [], color=BLUE, lw=2), Line2D([], [], color=GOLD, lw=2), Line2D([], [], color=PINK, lw=2.6),
                Patch(color="#9ec3bb", alpha=.5)], ["incoming", "outgoing", "residual", r"plane $x=0$"],
               loc="outside lower center", ncol=4, fontsize=7.5, frameon=False)
    tags(fig, bottom, labels=["(b)", "(c)", "(d)"], dx=.05)
    save(fig, "fig_interfaces3d")


def transfer():
    s = J("representation-transfer-2026-09-13/summary.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35), layout="constrained", gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    orig(ax[0], "representation-transfer-2026-09-13/figures/transfer-convergence.png", (1, 3), 0)   # field data no longer saved
    for a, label in zip(ax[1:], ("space", "time")):
        rows = s[label]
        x = [s["reports"][r["run"]]["grid"]["h"] if label == "space" else s["reports"][r["run"]]["dt"] for r in rows]
        for key, col, mk, lab in (("alpha", BLUE, "o", r"Nye tensor $\alpha$"), ("stress", ORANGE, "s", r"stress $\sigma$")):
            a.loglog(x, [r["errors"][key]["relative_l2"] for r in rows], mk + "-", color=col, label=lab)
        a.set(xlabel=r"cell spacing $h$" if label == "space" else r"time step $\Delta t$", ylabel=r"relative $L_2$ error")
        logticks(a, "x", sorted(x), "{:.3g}"); a.legend(fontsize=7)
    tags(fig, ax, dx=.05)
    save(fig, "fig_transfer")


def coarsening():
    r = J("coarsening-conditioning-2026-09-26/report.json")
    ss = J("spherical-sinks-2026-09-13/report.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), layout="constrained")
    wins, lab = ["512", "1024", "2048"], ["512–1024", "1024–2048", "2048–4096"]
    for k, (d, col, mk) in enumerate((("A", BLUE, "o"), ("B", ORANGE, "s"), ("C", AQUA, "^"))):
        m = [r["exponents"][d][w]["energy"][0] for w in wins]; e = [2*r["exponents"][d][w]["energy"][1] for w in wins]
        hd = r["discretizations"][d]
        ax[0].errorbar(np.arange(3) + (k - 1)*.13, m, yerr=e, fmt=mk, color=col, ms=5, capsize=3,
                       label=rf"$h={hd['h']:g}$, $\Delta t={hd['dt']:g}$")
    ax[0].axhline(1/3, color=INK, ls="--", lw=1); ax[0].text(2.3, 1/3 + .005, "1/3", fontsize=8)
    ax[0].set(xticks=range(3), ylabel=r"coarsening exponent ($\pm2$ SE)", xlabel="time window"); ax[0].set_xticklabels(lab, fontsize=7)
    ax[0].legend(fontsize=6.3, loc="upper left"); ax[0].grid(axis="x", visible=False); ax[0].set_ylim(.27, .44)
    z = np.load(R / "grain-mesh-2026-09-13/comparison.npz")
    for seed, col in zip((11, 29, 47), CYCLE):
        ax[1].plot(z[f"time_{seed}"], z[f"mean_area_96_{seed}"], "-", color=col, lw=1.6, label=f"seed {seed}")
        ax[1].plot(z[f"time_{seed}"], z[f"mean_area_128_{seed}"], "--", color=col, lw=1.3)
    ax[1].plot([], [], "k-", lw=1.4, label=r"$96^2$"); ax[1].plot([], [], "k--", lw=1.2, label=r"$128^2$")
    ax[1].set(xlabel=r"time $t$", ylabel=r"mean grain area $\langle A\rangle$"); ax[1].legend(fontsize=6.3, ncol=2, columnspacing=.6)
    tr = ss["time_refinement"]
    ax[2].loglog([t["dt"] for t in tr], [t["combined_relative_l2"] for t in tr], "o-", color=BLUE, label="bulk + stored")
    ax[2].loglog([t["dt"] for t in tr], [t["surface_absolute_error"] for t in tr], "s--", color=ORANGE, label="stored population")
    d = np.array([t["dt"] for t in tr]); ax[2].loglog(d, tr[0]["combined_relative_l2"]*(d/d[0]), ":", color=MUTED, lw=1, label="slope 1")
    ax[2].set(xlabel=r"time step $\Delta t$", ylabel="error"); ax[2].legend(fontsize=7, loc="lower right")
    logticks(ax[2], "x", list(d), "{:g}")
    tags(fig, ax, dx=.05)
    save(fig, "fig_coarsening")


def sinks():
    ss = J("spherical-sinks-2026-09-13/report.json"); pc = J("microstructure-2026-09-13/pore-capillary/report.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), layout="constrained")
    for k, col, mk in ((0.1, BLUE, "o"), (1.0, ORANGE, "s"), (10.0, AQUA, "^")):
        rows = [q for q in ss["steady"] if abs(q["reactivity"] - k) < 1e-12]
        if rows:
            ax[0].loglog([q["h"] for q in rows], [q["profile_relative_l2"] for q in rows], mk + "-", color=col, label=rf"$k_s={k:g}$")
            ax[1].loglog([q["h"] for q in rows], [q["capture_relative_error"] for q in rows], mk + "-", color=col, label=rf"$k_s={k:g}$")
    hs = sorted({q["h"] for q in ss["steady"]})
    for a, yl in ((ax[0], r"profile error (rel. $L_2$)"), (ax[1], "capture-coefficient error")):
        a.set(xlabel=r"grid spacing $h$", ylabel=yl); a.legend(fontsize=7); logticks(a, "x", hs[1:], "{:g}")
    rows = sorted(pc["rows"], key=lambda q: q["radius"])
    fine = [q for q in rows if q["n"] == max(x["n"] for x in rows)]
    R = np.array([q["radius"] for q in fine])
    ax[2].plot(1/R, [q["capillary_potential"] for q in fine], "o", color=BLUE, label=r"equilibrium, computed")
    ax[2].plot(1/R, [q["capillary_expected"] for q in fine], "-", color=BLUE, lw=1.2, label=r"$\gamma/R$")
    ax[2].plot(1/R, [q["initial_neutral_potential"] for q in fine], "s", color=ORANGE, label="zero growth, computed")
    ax[2].plot(1/R, [q["neutral_expected"] for q in fine], "--", color=ORANGE, lw=1.2, label=r"$6\gamma/(5R)$")
    ax[2].set(xlabel=r"$1/R$", ylabel=r"chemical potential $\mu$"); ax[2].legend(fontsize=6.5)
    tags(fig, ax, dx=.05)
    save(fig, "fig_sinks_pores")


def seggrowth():
    s = J("segregation-2026-09-26/report.json"); pg = J("pore-growth-2026-09-26/report.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), layout="constrained")
    rows = s["rows"]
    xb = np.array([q["bulk_solute"] for q in rows]); G = np.array([q["adsorption"] for q in rows])
    ax[0].semilogx(xb, G, "o", color=BLUE, label="computed")
    ax[0].set(xlabel=r"bulk solute fraction $x_b$", ylabel=r"adsorption $\Gamma$")
    mu = np.array([q["mu"] for q in rows]); g = np.array([q["gamma"] for q in rows])
    dg = -np.gradient(g, mu)
    ax[1].plot(dg, G, "o", color=BLUE, label="computed")
    lim = [0, max(G.max(), dg.max())*1.05]; ax[1].plot(lim, lim, "--", color=INK, lw=1, label=r"$\Gamma=-\mathrm{d}\gamma/\mathrm{d}\mu$")
    ax[1].set(xlabel=r"$-\mathrm{d}\gamma/\mathrm{d}\mu$", ylabel=r"adsorption $\Gamma$"); ax[1].legend(fontsize=7)
    for (mob, h), col, mk in zip((("single", .5), ("single", .25)), (BLUE, ORANGE), "os"):
        q = [x for x in pg["rows"] if x["mobility"] == mob and x["h"] == h]
        ax[2].plot([x["drive"] for x in q], [x["velocity"] for x in q], mk, color=col, ms=4.5, label=rf"$h={h:g}$")
    dd = np.linspace(min(x["drive"] for x in pg["rows"]), max(x["drive"] for x in pg["rows"]), 10)
    ax[2].plot(dd, pg["post_hoc_regression"]["fits"]["single-0.25"]["slope"]*dd, "-", color=INK, lw=1, label=r"$V=M_I(\mu-\gamma/R)$")
    ax[2].axhline(0, color=MUTED, lw=.6); ax[2].axvline(0, color=MUTED, lw=.6)
    ax[2].set(xlabel=r"driving force $\mu-\gamma/R$", ylabel=r"interface velocity $V$"); ax[2].legend(fontsize=7)
    tags(fig, ax, dx=.05)
    save(fig, "fig_segregation_growth")


def clusters():
    c = J("critical-nucleus-2026-09-28/report.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), layout="constrained")
    sft = [J(f"sft-2026-09-28/N{k}.json") for k in (6, 9, 13)]
    n = [q["n"] for q in sft]
    for key, lab, col, mk in (("E_void", "compact void", BLUE, "o"), ("E_frank", "Frank loop", ORANGE, "s"),
                              ("E_annealed", "annealed (SFT)", AQUA, "^")):
        ax[0].plot(n, [q[key] for q in sft], mk + "-", color=col, ms=5, label=lab)
    ax[0].set(xlabel=r"vacancies $n$", ylabel="formation energy (eV)"); ax[0].legend(fontsize=7)
    rows = c["rows"]; dg = np.array([q["dg"] for q in rows])
    ax[1].plot(1/dg, [q["barrier"] for q in rows], "o", color=BLUE, label="finite-width saddle")
    ax[1].plot(1/dg, [q["barrier_cnt"] for q in rows], "--", color=INK, lw=1.2, label="sharp-interface theory")
    ax[1].set(xlabel=r"$1/\Delta g$", ylabel=r"nucleation barrier $\Delta G^*$"); ax[1].legend(fontsize=7)
    for key, col, lab in (("plus", ORANGE, "perturbed up: grows"), ("minus", BLUE, "perturbed down: dissolves")):
        e = np.array(c["flows"][key]["energy_samples"])
        ax[2].plot(np.linspace(0, c["flows"][key]["time"], len(e)), e - e[0], color=col, label=lab)
    ax[2].set(xlabel=r"time $t$", ylabel=r"energy change $\Delta E$"); ax[2].legend(fontsize=7)
    tags(fig, ax, dx=.05)
    save(fig, "fig_clusters")


def linefeedback():
    k = J("interface-graphs-2026-09-12/kinetics/report.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), layout="constrained", gridspec_kw=dict(width_ratios=[1.15, .9, 1]))
    base = "curved-loops-fcc-2026-09-12/"
    z = np.load(R / (base + "coupled-moving/checkpoint.npz"), allow_pickle=True)
    rep = J(base + "coupled-moving/report.json"); mov, pin = rep["metrics"], J(base + "coupled-pinned/report.json")["metrics"]
    L = rep["grid"]["length"]; n = z["phase"].shape[0]; mid = n//2
    slip = np.sqrt((z["slip"]**2).sum(axis=0))[:, :, mid]
    im = ax[0].imshow(slip.T, origin="lower", extent=(0, L, 0, L), cmap="viridis")
    ax[0].contour(np.linspace(0, L, n), np.linspace(0, L, n), z["phase"][:, :, mid].T, levels=[.5], colors="white", linewidths=1)
    ax[0].set(xlabel=r"$x$", ylabel=r"$y$"); ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], shrink=.85, pad=.02).set_label(r"slip magnitude $|\gamma|$")
    vals = [mov["initial_energy"] - mov["final_energy"], pin["initial_energy"] - pin["final_energy"]]
    ax[1].bar([0, 1], vals, color=[BLUE, MUTED], width=.55)
    ax[1].set(xticks=[0, 1], ylabel=r"energy released $-\Delta F$", ylim=(66.3, 66.55))
    ax[1].set_xticklabels(["moving\nloop", "pinned\nloop"], fontsize=7.5); ax[1].grid(axis="x", visible=False)
    ax[1].text(.5, .92, rf"difference ${(vals[0] - vals[1])*1e3:.2f}\times10^{{-3}}$", transform=ax[1].transAxes, ha="center", fontsize=7)
    names = k["state_names"][1:]; x = np.arange(len(names))
    ax[2].bar(x - .18, k["first_event_probabilities"][1:], .36, color=BLUE, label="master equation")
    ax[2].bar(x + .18, k["sampled_probabilities"][1:], .36, color=ORANGE, label="sampled")
    ax[2].set(xticks=x, ylabel="first-event probability", ylim=(0, .55)); ax[2].set_xticklabels(["trans-\nmitted", "ab-\nsorbed", "re-\nflected"], fontsize=7.5)
    ax[2].legend(fontsize=7, loc="upper right"); ax[2].grid(axis="x", visible=False)
    tags(fig, ax, dx=.04)
    save(fig, "fig_linefeedback")


FIGS = {"fig_linefeedback": linefeedback, "fig_coarsening": coarsening, "fig_sinks_pores": sinks, "fig_segregation_growth": seggrowth, "fig_clusters": clusters,
        "fig_interfaces3d": interfaces3d, "fig_transfer": transfer, "fig_finitestrain": finitestrain, "fig_fracture": fracture, "fig_fracture2": fracture2, "fig_copper": copper, "fig_loops": loops, "fig_pf_convergence": pf_convergence, "fig_pf_kinetics": pf_kinetics, "fig_gridcost": gridcost, "fig_costs2": costs2}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
        print("wrote", name)
