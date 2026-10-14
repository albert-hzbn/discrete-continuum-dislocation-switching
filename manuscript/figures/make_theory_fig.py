"""Illustrations of the theoretical framework (main text) and of the numerics (SI), computed from simple 1D solutions.

Run from the repository root:
    python manuscript/figures/make_theory_fig.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from make_figures import AQUA, BLUE, INK, MUTED, ORANGE, save

PURPLE = "#8a5cc2"


def tag(fig, axes):
    fig.canvas.draw()
    for a, l in zip(axes, "abcdefgh"):
        b = a.get_position()
        fig.text(b.x0 - .075, b.y1 + .022, f"({l})", fontweight="bold", fontsize=10.5, va="bottom")


def theory():
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.9), layout="constrained")
    ax = ax.ravel()
    # (a) double well + gradient -> diffuse interface of width w
    e = np.linspace(-1.35, 1.35, 300); W, kap = 1., 1.
    ax[0].plot(e, W*(1 - e**2)**2/4, color=BLUE, lw=2)
    ax[0].set(xlabel=r"phase field $\eta$", ylabel=r"$f_{\mathrm{chem}}=Wf_0(\eta)$", ylim=(-.02, .5), xticks=[-1, 0, 1])
    ax[0].annotate("", (0, 0), (0, W/4), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    ax[0].text(.05, .015, r"barrier $W/4$", fontsize=8.0, color=MUTED, va="bottom")
    ax[0].set_xticks([-1, 0, 1], ["$-1$
phase A", "0", "1
phase B"])
    ins = ax[0].inset_axes([.36, .62, .3, .28])
    xx = np.linspace(-6, 6, 200); ell = np.sqrt(2*kap/W)
    ins.plot(xx, np.tanh(xx/ell), color=ORANGE, lw=1.6)
    ins.axvspan(-ell, ell, color=ORANGE, alpha=.15, lw=0)
    ins.set(xticks=[], yticks=[]); ins.tick_params(labelsize=8.0); ins.set_xlabel(r"interface profile, width $\sim\sqrt{\kappa/W}$", fontsize=8.0, labelpad=2)
    # (b) one-variable Rayleighian: downhill at speed m F'(x)
    x = np.linspace(-1.6, 1.8, 300); F = .25*x**4 - .5*x**2 + .15*x
    ax[1].plot(x, F, color=INK, lw=1.8)
    for m, col, lab in ((1., BLUE, r"mobility $m$"), (.25, ORANGE, r"$m/4$")):
        xs, t, pts = [1.72], 0., []
        for k in range(480):
            xs.append(xs[-1] - .002*m*(xs[-1]**3 - xs[-1] + .15))
        xs = np.array(xs)[::60]
        ax[1].plot(xs, .25*xs**4 - .5*xs**2 + .15*xs, "o", color=col, ms=7 if m == 1. else 4.5, label=lab, zorder=3 if m != 1. else 2)
    ax[1].set(xlabel=r"state $x$", ylabel=r"free energy $\mathcal{F}(x)$", yticks=[])
    ax[1].legend(fontsize=8.0, title=r"$\dot x=-m\,\mathcal{F}'(x)$, equal times", title_fontsize=8.0, loc="upper left")
    # (c) conserved vs non-conserved relaxation of the same bump (1D periodic, spectral)
    n, L = 256, 40.; xg = np.linspace(0, L, n, endpoint=False); k = 2*np.pi*np.fft.fftfreq(n, L/n)
    c0 = .5 + .45*np.exp(-((xg - L/2)/2.5)**2)
    def relax(conserved, t_end=6., dt=.01):
        c = c0.copy()
        for _ in range(int(t_end/dt)):
            if conserved:
                c = np.real(np.fft.ifft(np.fft.fft(c)/(1 + dt*k**2)))          # dc/dt = lap(c - 0.5)
            else:
                c = c - dt*.4*(c - .5)                                            # dc/dt = -L (c - 0.5)
        return c
    ax[2].plot(xg, c0, color=MUTED, lw=1.2, label="initial")
    ax[2].plot(xg, relax(False), color=ORANGE, lw=1.8, label="non-conserved: decays in place")
    ax[2].plot(xg, relax(True), color=BLUE, lw=1.8, ls="--", label="conserved: spreads, area fixed")
    ax[2].set(xlabel=r"position $x$", ylabel="field value", ylim=(.45, 1.2), xticks=[])
    ax[2].legend(fontsize=8.0, loc="upper right")
    # (d) a line defect spread into a field (Nye tensor) with kernel width l
    X, Y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    th = np.linspace(0, 2*np.pi, 80, endpoint=False); R0, ell = 1.6, .25
    a = sum(np.exp(-((X - R0*np.cos(t))**2 + (Y - R0*np.sin(t))**2)/(2*ell**2)) for t in th)
    ax[3].imshow(a, extent=(-3, 3, -3, 3), origin="lower", cmap="Blues")
    ax[3].plot(R0*np.cos(th), R0*np.sin(th), "--", color=ORANGE, lw=1)
    ax[3].annotate("", (R0 + ell*2, 0), (R0 - ell*2, 0), arrowprops=dict(arrowstyle="<->", color=INK, lw=.9))
    ax[3].text(R0 + .25, .35, r"$\sim\ell$", fontsize=8)
    ax[3].set(xticks=[], yticks=[], xlabel=r"loop line (dashed) and its density $\alpha$"); ax[3].grid(False)
    tag(fig, ax)
    save(fig, "fig_theory")


def numerics():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), layout="constrained")
    # (a) five-point stencil and a conservative face flux
    a = ax[0]; a.set(xlim=(-.3, 4.3), ylim=(-.3, 4.3), aspect="equal"); a.set_axis_off()
    for i in range(4):
        for j in range(4):
            a.add_patch(Rectangle((i, j), 1, 1, fc="white", ec=MUTED, lw=.8))
    for (i, j), col in (((1.5, 1.5), ORANGE), ((2.5, 1.5), BLUE), ((.5, 1.5), BLUE), ((1.5, 2.5), BLUE), ((1.5, .5), BLUE)):
        a.plot(i, j, "o", color=col, ms=8)
    a.text(1.5, 1.2, r"$i,j$", ha="center", fontsize=8.0)
    a.add_patch(FancyArrowPatch((2.0, 1.72), (2.0, 1.72), arrowstyle="-", lw=0))
    a.annotate("", (2.35, 1.95), (1.65, 1.95), arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.6))
    a.text(2.0, 2.18, r"$J_{ij}$", ha="center", fontsize=8, color=AQUA)
    a.text(2, -.25, "same flux leaves one cell\nand enters its neighbour", ha="center", va="top", fontsize=8.0)
    # (b) cloud-in-cell weights
    a = ax[1]; a.set(xlim=(-.1, 2.1), ylim=(-.1, 2.1), aspect="equal"); a.set_axis_off()
    for i in range(2):
        for j in range(2):
            a.add_patch(Rectangle((i, j), 1, 1, fc="white", ec=MUTED, lw=.8)); a.plot(i + .5, j + .5, "s", color=MUTED, ms=5)
    px, py = .82, 1.18
    a.add_patch(Rectangle((px - .5, py - .5), 1, 1, fc=BLUE, alpha=.15, ec=BLUE, lw=1, ls="--"))
    a.plot(px, py, "o", color=ORANGE, ms=9)
    tx, ty = px - .5, py - .5
    for (cx, cy), w in (((.5, .5), (1 - tx)*(1 - ty)), ((1.5, .5), tx*(1 - ty)), ((.5, 1.5), (1 - tx)*ty), ((1.5, 1.5), tx*ty)):
        a.text(cx, cy - .22, f"{w:.2f}", ha="center", fontsize=8.0, color=BLUE)
    a.text(1, -.05, "weights sum to 1: one dislocation\ndeposited = one on the grid", ha="center", va="top", fontsize=8.0)
    # (c) MUSCL-minmod upwind transport of a sharp wall vs first-order upwind
    n, L = 200, 1.; xg = (np.arange(n) + .5)/n; h = 1/n
    r0 = np.where(abs(xg - .25) < .06, 1., 0.)
    def advect(r, order2, steps=300, v=1., dt=.5*h):
        r = r.copy()
        for _ in range(steps):
            d = np.roll(r, -1) - r
            s = np.where(d*np.roll(d, 1) > 0, np.sign(d)*np.minimum(abs(d), abs(np.roll(d, 1))), 0.) if order2 else 0*r
            F = v*(r + .5*s)
            r = r - dt/h*(F - np.roll(F, 1))
        return r
    ax[2].plot(xg, np.roll(r0, 150), color=MUTED, lw=1.2, label="exact")
    ax[2].plot(xg, advect(r0, False), color=ORANGE, lw=1.6, label="first-order upwind")
    ax[2].plot(xg, advect(r0, True), color=BLUE, lw=1.8, label="MUSCL--minmod")
    ax[2].set(xlabel=r"position $x$", ylabel=r"density $\rho$", ylim=(-.05, 1.3))
    ax[2].legend(fontsize=8.0, loc="upper right")
    for a, t in zip(ax, ("finite-volume fluxes", "particle-to-grid transfer", "density transport")):
        a.set_title(t, fontsize=8.5)
    tag(fig, ax)
    save(fig, "fig_numerics")


if __name__ == "__main__":
    theory(); numerics(); print("wrote fig_theory, fig_numerics")
