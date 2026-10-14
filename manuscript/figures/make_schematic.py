"""Schematic of the framework and the allocator (main-text Figure 1).

Run from the repository root:
    python manuscript/figures/make_schematic.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from make_figures import AQUA, BLUE, INK, MUTED, ORANGE, save

LIGHT, PALE_O, GREY = "#eef3fb", "#fbe3d6", "#f4f3ef"
rng = np.random.default_rng(3)


def box(ax, x, y, w, h, text, fc="white", ec=MUTED, fs=8, lw=1.1, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.018", fc=fc, ec=ec, lw=lw))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight=weight, color=color)


def arrow(ax, p, q, color=MUTED, lw=1.3, style="-|>", rad=0.0, ms=9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def edge(ax, x, y, s, size=0.016, color=INK, lw=1.3):
    """Edge dislocation symbol: perpendicular sign, s = +1 (upright) or -1 (inverted)."""
    ax.plot([x - size, x + size], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=5)
    ax.plot([x, x], [y, y + s*1.3*size], color=color, lw=lw, solid_capstyle="butt", zorder=5)


# ---------------------------------------------------------------------------------------------------------------
def framework(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.set_axis_off()
    x0, y0, w, h = .04, .2, .92, .5
    ax.add_patch(Rectangle((x0, y0), w, h, fc=GREY, ec=INK, lw=1.1, zorder=0))
    xx, yy = np.meshgrid(np.linspace(x0, x0 + w, 240), np.linspace(y0, y0 + h, 140))
    blob = 1/(1 + np.exp((np.hypot(xx - .2, yy - .5) - .085)/.01))                  # phase field eta
    ax.imshow(np.ma.masked_less(blob, .03), extent=(x0, x0 + w, y0, y0 + h), origin="lower", cmap="Blues", vmin=-.15,
              vmax=1.25, zorder=1, aspect="auto")
    band = np.exp(-((xx - .47)/.05)**2)                                                # density rho
    ax.imshow(np.ma.masked_less(band, .04), extent=(x0, x0 + w, y0, y0 + h), origin="lower", cmap="Oranges", vmin=0,
              vmax=1.5, zorder=1, alpha=.95, aspect="auto")
    for x, y, s in ((.65, .58, 1), (.76, .6, -1), (.71, .47, -1), (.64, .30, 1), (.74, .27, 1)):   # lines X_l
        edge(ax, x, y, s)
    t = np.linspace(0, 1, 60)                                                           # crack q
    cx, cy = .83 + .1*t, .28 + .03*np.sin(7*t)
    ax.plot(cx, cy, color="#8a1c1c", lw=5, alpha=.25, solid_capstyle="round", zorder=2)
    ax.plot(cx, cy, color=INK, lw=2, solid_capstyle="round", zorder=3)
    for x, y in rng.uniform([.09, .27], [.33, .37], (16, 2)):                           # vacancies c_v
        ax.plot(x, y, "o", ms=2, color=AQUA, zorder=3)
    lab = dict(fontsize=7.5, ha="center", va="center", color=INK, zorder=6)
    ax.text(.16, .655, r"field $\eta,c$", **lab)
    ax.text(.47, .655, r"$\rho_\pm$", **lab)
    ax.text(.70, .655, r"lines $X_l$", **lab)
    ax.text(.86, .40, "damage\n" + r"$q$", **{**lab, "linespacing": 1.1})   # two lines keep it inside the frame
    ax.text(.21, .232, r"vacancies $c_v$", **lab)
    # couplings between representations
    arrow(ax, (.555, .40), (.625, .40), BLUE, style="<|-|>", ms=7)
    ax.text(.59, .425, "switch", fontsize=7.5, ha="center", va="bottom", color=BLUE)
    box(ax, .04, .83, .92, .14, r"one free energy  $\mathcal{F}[c,\ \eta,\ \rho_\pm,\ \{X_l\},\ q,\ u]$", fc=LIGHT, ec=BLUE, fs=8)
    arrow(ax, (.5, .83), (.5, .71), BLUE)
    box(ax, .04, .02, .44, .12, "one dissipation\nprinciple",
        fc=LIGHT, ec=BLUE, fs=7.5)
    box(ax, .52, .02, .44, .12, "exact\nledgers", fc=LIGHT, ec=BLUE, fs=7.5)
    arrow(ax, (.26, .14), (.26, .195), BLUE); arrow(ax, (.74, .14), (.74, .195), BLUE)


def allocation(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.set_axis_off()
    nb, x0, y0, s, sy = 6, .06, .42, .88/6, .09
    band = {2, 3}
    counts = np.where(np.isin(np.arange(nb), list(band))[None, :], rng.integers(380, 520, (nb, nb)),
                      rng.integers(0, 3, (nb, nb)))
    for i in range(nb):
        for j in range(nb):
            cont = j in band
            ax.add_patch(Rectangle((x0 + i*s, y0 + j*sy), s, sy, fc=PALE_O if cont else "white", ec=MUTED, lw=.7,
                                   hatch="////" if cont else None, zorder=1))
            if not cont:
                slots = [(.3, .3), (.7, .62)]
                for k in range(counts[i, j]):
                    fx, fy = slots[k]
                    edge(ax, x0 + (i + fx + rng.uniform(-.08, .08))*s, y0 + (j + fy)*sy, rng.choice([-1, 1]),
                         size=.011, lw=1.1)
    ax.text(x0 + 3*s, y0 + 2.5*sy, r"$n_\beta$=437", fontsize=7.5, ha="center", va="center", color=INK, zorder=4,
            bbox=dict(fc="white", ec=MUTED, lw=.6, boxstyle="round,pad=0.25"))
    ax.text(x0 + 1.5*s, y0 + 4.5*sy, r"$n_\beta$=2", fontsize=7.5, ha="center", va="center", color=INK, zorder=4)
    ax.text(x0 + 3*s, y0 - .05, "hatched: continuum    open: discrete", fontsize=7.5, ha="center", va="top", color=INK)
    # hysteresis loop: state against block count
    hx0, hy0, hw, hh = .14, .06, .76, .24
    ax.plot([hx0, hx0 + hw], [hy0, hy0], color=INK, lw=.9); ax.plot([hx0, hx0], [hy0, hy0 + hh], color=INK, lw=.9)
    xc, xh = hx0 + .62*hw, hx0 + .34*hw
    lo, hi = hy0 + .12*hh, hy0 + .82*hh
    ax.plot([hx0, xc, xc, hx0 + hw], [lo, lo, hi, hi], color=ORANGE, lw=1.6)
    ax.plot([hx0 + hw, xh, xh, hx0], [hi + .03, hi + .03, lo + .03, lo + .03], color=BLUE, lw=1.6, ls="--")
    arrow(ax, (xc, lo + .03), (xc, hi - .03), ORANGE, ms=8); arrow(ax, (xh, hi), (xh, lo + .06), BLUE, ms=8)
    ax.text(xc, hy0 - .045, r"$n_c$", ha="center", fontsize=7.5); ax.text(xh, hy0 - .045, r"$n_c/2$", ha="center", fontsize=7.5)
    ax.text(hx0 + hw + .02, hy0, r"$n_\beta$", fontsize=8, va="center")
    ax.text(hx0 - .025, lo + .015, "discrete", fontsize=7.5, ha="right", va="center")
    ax.text(hx0 - .025, hi + .015, "continuum", fontsize=7.5, ha="right", va="center")


def timestep(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.set_axis_off()
    x0, w = .03, .74
    realloc = [(r"count $n_\beta$ per block", "white", MUTED),
               ("switch blocks\n(absorb / emit)", PALE_O, ORANGE),
               (r"check ledger: $N$, $\Sigma_i\, s_ib_0$", LIGHT, BLUE)]
    every = [("one stress solve (FFT)\nfrom particles + density", "white", MUTED),
             ("glide particles,\n" + r"transport $\rho_\pm$", "white", MUTED)]
    ys = [.87, .69, .51, .3, .12]
    for (t, fc, ec), y in zip(realloc + every, ys):
        box(ax, x0, y - .065, w, .13, t, fc=fc, ec=ec, fs=7.5)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, (x0 + w/2, y1 - .067), (x0 + w/2, y2 + .067), MUTED, ms=8)
    # brackets: reallocation block (every N_r steps) and time-step loop (every step)
    xr = x0 + w + .04
    ax.plot([xr, xr + .03, xr + .03, xr], [ys[0] + .065, ys[0] + .065, ys[2] - .065, ys[2] - .065], color=ORANGE, lw=1)
    ax.text(xr + .05, (ys[0] + ys[2])/2, r"every $N_r$ steps", rotation=270, fontsize=7.5, va="center", ha="left", color=ORANGE)
    arrow(ax, (x0 + w + .005, ys[4]), (x0 + w + .005, ys[3]), MUTED, rad=.6, ms=8)
    ax.text(xr + .1, (ys[3] + ys[4])/2, "every step", rotation=270, fontsize=7.5, va="center", ha="left", color=MUTED)


def main():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.05), layout="constrained", gridspec_kw=dict(width_ratios=[1.2, 1, .82]))
    framework(ax[0]); allocation(ax[1]); timestep(ax[2])
    for a, t, l in zip(ax, ("Framework", "Allocation", "One time step"), "abc"):
        a.set_title(f"({l}) {t}", loc="left", fontsize=9, fontweight="bold")
    save(fig, "fig_schematic")
    print("wrote fig_schematic")


if __name__ == "__main__":
    main()
