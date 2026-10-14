"""Manuscript figures, generated only from saved benchmark reports.

Run from the repository root:
    python manuscript/figures/make_figures.py

Every plotted number is read from results/*/report.json; nothing is retyped.
Output: manuscript/figures/fig_crossover.pdf, fig_allocator.pdf (and .png previews).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RES = ROOT / "results"

# Validated categorical order (blue, orange, aqua); text and axes stay neutral ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#dddcd6"

plt.rcParams.update({
    "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 9.5, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.9, "xtick.major.width": 0.9,
    "ytick.major.width": 0.9, "xtick.major.size": 4, "ytick.major.size": 4, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 2.0, "lines.markersize": 6.5, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(path):
    return json.loads((RES / path).read_text(encoding="utf-8"))


def label(ax, text):
    """Bold panel letter as a left-aligned axes title (the layout engine reserves its space)."""
    old = ax.get_title(loc="left")
    ax.set_title(f"{text}  {old}" if old else text, loc="left", fontweight="bold", fontsize=10.5, pad=5)


def text_clashes(fig, pad_pt=1.5):
    """Pairs of visible text elements whose boxes, grown by pad_pt points, overlap (tick labels of one axis excepted)."""
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); pad = pad_pt*fig.dpi/72
    hidden = set()
    for ax in fig.axes:                      # ticks outside the view limits are not drawn
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - 1e-12*abs(hi - lo) <= tk.get_loc() <= hi + 1e-12*abs(hi - lo)) or not tk.label1.get_visible():
                    hidden.add(id(tk.label1)); hidden.add(id(tk.label2))
                if not tk.label2.get_visible():
                    hidden.add(id(tk.label2))
    for ax in fig.axes:                      # tick labels of axes that are switched off are not drawn
        if not ax.axison:
            hidden.update(id(t) for t in ax.get_xticklabels() + ax.get_yticklabels())
            hidden.update(id(t) for t in (ax.xaxis.label, ax.yaxis.label))
    items = []
    for t in fig.findobj(Text):
        if id(t) in hidden or not t.get_visible() or not t.get_text().strip() or t.get_alpha() == 0:
            continue
        try:
            bb = Text.get_window_extent(t, r)
        except Exception:
            continue
        if bb.width < 1 or bb.height < 1:
            continue
        axis = getattr(t, "_clash_group", None)
        items.append((t, bb.expanded(1, 1).padded(pad)))
    from matplotlib.legend import Legend
    owner = {}
    for lg in fig.findobj(Legend):                   # rows of one legend are spaced by matplotlib itself
        for t in lg.get_texts():
            owner[id(t)] = id(lg)
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (a, ba), (b, bb) = items[i], items[j]
            if owner.get(id(a)) is not None and owner.get(id(a)) == owner.get(id(b)):
                continue
            if ba.overlaps(bb):
                out.append((a.get_text()[:30], b.get_text()[:30]))
    return out


def text_feature_clashes(fig, pad_pt=2.0):
    """Text inside a plot (annotations, labels, legends) closer than pad_pt to a data line, marker or bar."""
    import numpy as np
    from matplotlib.patches import Rectangle as _Rect
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); pad = pad_pt*fig.dpi/72
    out = []
    for ax in fig.axes:
        if not ax.axison or ax.name != "rectilinear" or ax.get_images():
            continue
        box = ax.get_window_extent(r)
        from matplotlib.text import Text as _T
        texts = [(t.get_text()[:28], _T.get_window_extent(t, r)) for t in ax.texts if t.get_visible() and t.get_text().strip()]
        lg = ax.get_legend()
        if lg is not None and lg.get_visible():
            texts.append(("[legend]", lg.get_window_extent(r)))
        if not texts:
            continue
        pts = []
        for ln in ax.lines:
            if not ln.get_visible() or ln.get_label().startswith("_child") and ln.get_linestyle() in ("None", "") and ln.get_marker() in ("None", "", None):
                continue
            xy = ln.get_transform().transform(np.column_stack(ln.get_data(orig=False)).astype(float))
            xy = xy[np.all(np.isfinite(xy), axis=1)]
            if len(xy) == 0:
                continue
            rad = ln.get_markersize()*fig.dpi/72/2 if ln.get_marker() not in ("None", "", None, ",") else 0
            if ln.get_linestyle() not in ("None", "", " ") and len(xy) > 1:
                seg = [np.linspace(xy[i], xy[i + 1], max(2, int(np.hypot(*(xy[i + 1] - xy[i]))/1.5) + 1)) for i in range(len(xy) - 1)]
                pts.append((np.vstack(seg), ln.get_linewidth()*fig.dpi/72/2))
            if rad:
                pts.append((xy, rad))
        for col in ax.collections:
            off = col.get_offsets()
            if len(off):
                xy = col.get_offset_transform().transform(np.asarray(off, float))
                pts.append((xy, 3*fig.dpi/72))
        bars = [p.get_window_extent(r) for p in ax.patches if isinstance(p, _Rect) and p.get_visible() and p.get_alpha() != 0 and p.get_facecolor()[3] > 0]
        for name, tb in texts:
            grown = tb.padded(pad)
            hit = False
            for xy, rad in pts:
                inside = (xy[:, 0] > box.x0) & (xy[:, 0] < box.x1) & (xy[:, 1] > box.y0) & (xy[:, 1] < box.y1)
                q = xy[inside]
                if len(q) and np.any((q[:, 0] > grown.x0 - rad) & (q[:, 0] < grown.x1 + rad) & (q[:, 1] > grown.y0 - rad) & (q[:, 1] < grown.y1 + rad)):
                    hit = True; break
            if not hit and any(grown.overlaps(b) for b in bars):
                hit = True
            if hit:
                out.append(name)
    return out


def auto_legends(fig):
    """Move each in-axes legend to the first standard location where it clears the data; add headroom if needed."""
    import numpy as np
    locs = ["upper right", "upper left", "lower left", "lower right", "center right", "center left", "upper center",
            "lower center", "center"]
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is None or not ax.axison or ax.get_images():
            continue
        others = [a for a in fig.axes if a is not ax and a.get_legend() is not None]
        def clear():
            for a in others:                        # test this legend alone
                a.get_legend().set_visible(False)
            hidden = [t for t in ax.texts if t.get_visible()]
            for t in hidden:
                t.set_visible(False)
            ok = "[legend]" not in text_feature_clashes(fig)
            for t in hidden:
                t.set_visible(True)
            for a in others:
                a.get_legend().set_visible(True)
            return ok
        if clear():
            continue
        lg.set_bbox_to_anchor(None)                 # positions relative to the axes, not a manual anchor
        done = False
        for attempt in range(4):
            for loc in locs if attempt == 0 else locs[:2] + ["upper center"]:
                lg.set_loc(loc)
                if clear():
                    done = True; break
            if done:
                break
            lo, hi = ax.get_ylim()                  # headroom above the data for the legend
            if ax.get_yscale() == "log":
                ax.set_ylim(lo, hi*(hi/lo)**0.25)
            else:
                ax.set_ylim(lo, hi + 0.25*(hi - lo))


def save(fig, name):
    auto_legends(fig)
    near = text_feature_clashes(fig)
    if near:
        print(f"[near feature] {name}: " + "; ".join(f"'{n}'" for n in near))
    clashes = text_clashes(fig)
    if clashes:
        print(f"[text clash] {name}: " + "; ".join(f"'{a}' / '{b}'" for a, b in clashes[:12])
              + (f" (+{len(clashes) - 12} more)" if len(clashes) > 12 else ""))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def crossover():
    """(a) Section 67 speedup vs active fraction; (b) adaptive advantage vs workers (Sections 55, 61)."""
    rows = load("crossover-map-2026-09-27/report.json")["rows"]
    threads = load("parallel-scaling-2026-09-26/report.json")["adaptive_over_uniform_speedup"]
    procs = load("amr-message-passing-2026-09-26/message-passing/report.json")["adaptive_over_uniform"]

    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 4.6), layout="constrained")
    f = np.array([r["active_fraction"] for r in rows])
    g = np.linspace(0.2, 1.0, 200)
    a.plot(g**2, 1 / (g + (1 - g) / 4) ** 2, color=MUTED, lw=1.0, ls="--")
    a.plot(f, [r["speedup"] for r in rows], "o-", color=BLUE)
    a.axhline(2.0, color=GRID, lw=0.8, zorder=0)
    a.annotate("predicted cell ratio", (0.47, 1 / (0.686 + 0.314 / 4) ** 2), xytext=(0.5, 3.3),
               color=MUTED, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a.annotate("measured speedup", (f[1], rows[1]["speedup"]), xytext=(0.3, 4.6), color=BLUE, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    a.set(xlabel="Fine-band area fraction", ylabel="Uniform / adaptive time", xlim=(0, 1.02), ylim=(0.8, 5.5))
    label(a, "(a)")

    p = [1, 2, 4, 8]
    b.plot(p, [threads[str(k)] for k in p], "o-", color=BLUE)
    b.plot(p, [procs[str(k)] for k in p], "s-", color=ORANGE)
    b.text(8.9, threads["8"], "threads", color=INK, fontsize=8.5, va="center")
    b.text(8.9, procs["8"], "processes", color=INK, fontsize=8.5, va="center")
    b.set(xscale="log", xlabel="Workers", ylabel="Uniform / adaptive time", ylim=(0, 11), xlim=(0.85, 18))
    b.set_xticks(p, [str(k) for k in p])
    b.minorticks_off()
    label(b, "(b)")
    save(fig, "fig_crossover")


def allocator():
    """(a) Section 92 sparse-region error vs threshold; (b) CPU cost per step vs threshold."""
    rep = load("allocator-control-2026-09-28/report.json")
    cpu = load("allocator-control-2026-09-28/cpu-time-post-hoc.json")["minimum"]
    nc = rep["thresholds"]
    s91, s92 = rep["study"]["91"], rep["study"]["92"]

    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 4.6), layout="constrained")
    a.plot(nc, [100 * s91[str(n)]["error"] for n in nc], "o-", color=BLUE)
    a.plot(nc, [100 * s92[str(n)]["error"] for n in nc], "s--", color=ORANGE)
    a.axhline(100 * s92["C"]["error"], color=MUTED, lw=1.0, ls=":")
    a.text(nc[-1], 100 * s92["C"]["error"] + 0.08, "continuum C", color=MUTED, fontsize=8.5, ha="right", va="bottom")
    a.text(22, 100 * s91["64"]["error"] - 0.45, "calibration seed", color=BLUE, fontsize=8.5, ha="center")
    a.text(64, 100 * s92["64"]["error"] + 0.18, "test seed", color=ORANGE, fontsize=8.5, ha="center")
    a.set(xscale="log", xlabel=r"Continuum threshold $n_c$ (lines per block)",
          ylabel="Sparse-region error vs D (%)", ylim=(0, 2.6))
    a.set_xticks(nc, [str(n) for n in nc])
    a.minorticks_off()
    label(a, "(a)")

    b.plot(nc, [1e3 * cpu[f"H{n}"] for n in nc], "o-", color=BLUE)
    b.axhline(1e3 * cpu["D"], color=ORANGE, lw=1.2)
    b.axhline(1e3 * cpu["C"], color=AQUA, lw=1.2)
    b.text(nc[0], 1e3 * cpu["D"] * 1.3, "discrete D", color=INK, fontsize=8.5, va="bottom")
    b.text(nc[0], 1e3 * cpu["C"] * 0.7, "continuum C", color=INK, fontsize=8.5, va="top")
    b.text(nc[-1], 1e3 * cpu["H512"] * 1.45, "allocator H", color=INK, fontsize=8.5, ha="right", va="bottom")
    b.set(xscale="log", yscale="log", xlabel=r"Continuum threshold $n_c$ (lines per block)",
          ylabel="CPU time per step (ms)", ylim=(0.7, 120))
    b.set_xticks(nc, [str(n) for n in nc])
    b.minorticks_off()
    label(b, "(b)")
    save(fig, "fig_allocator")


if __name__ == "__main__":
    crossover()
    allocator()
    print("wrote", sorted(p.name for p in OUT.glob("fig_*")))
