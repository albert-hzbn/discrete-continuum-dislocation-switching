"""Topic figures assembled from each study's data-generated report figures (positive outcomes only).

Run from the repository root:
    python manuscript/figures/make_composites.py

Each panel is a report figure written by that study's script from its saved results (see the table in
Appendix B); an optional crop box (left, top, right, bottom, in source pixels) keeps only the relevant part.
Rows are scaled to a common height and to the full figure width, and panels are labelled (a), (b), ...
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
R = ROOT / "results"

FIGS = {
    "fig_pf_convergence": [
        [("figures/convergence.png", None), ("microstructure-2026-09-13/figures/triple-junctions.png", None)],
    ],
    "fig_pf_kinetics": [
        [("kinetic-laws-2026-09-26/figures/kinetic-laws.png", None)],
        [("microstructure-2026-09-13/figures/grain-statistics.png", None),
         ("precipitate-interactions-2026-09-13/figures/precipitate-pair-convergence.png", None)],
    ],
    "fig_coarsening": [
        [("coarsening-conditioning-2026-09-26/figures/coarsening-conditioning.png", None)],
        [("grain-mesh-2026-09-13/figures/grain-mesh.png", None),
         ("spherical-sinks-2026-09-13/figures/spherical-storage.png", None)],
    ],
    "fig_sinks_pores": [
        [("spherical-sinks-2026-09-13/figures/spherical-capture.png", None),
         ("microstructure-2026-09-13/figures/pore-kinetics.png", None)],
    ],
    "fig_segregation_growth": [
        [("segregation-2026-09-26/figures/segregation.png", None),
         ("pore-growth-2026-09-26/figures/pore-growth.png", (0, 0, .5, 1))],
    ],
    "fig_loops": [
        [("curved-loops-fcc-2026-09-12/figures/loop-convergence.png", None)],
        [("frank-read-elastic-2026-09-28/figures/frank-read-elastic.png", None),
         ("disclination-dynamics-2026-09-28/figures/disclination-dynamics.png", None)],
    ],
    "fig_interfaces3d": [
        [("interface-graphs-2026-09-12/figures/spatial-branches.png", None),
         ("three-dimensional-2026-09-12/figures/convergence-cost.png", None)],
    ],
    "fig_transfer": [
        [("representation-transfer-2026-09-13/figures/transfer-convergence.png", None)],
    ],
    "fig_linefeedback": [
        [("curved-loops-fcc-2026-09-12/figures/coupled-loop-feedback.png", None),
         ("interface-graphs-2026-09-12/figures/reaction-kinetics.png", None)],
    ],
    "fig_fracture": [
        [("transition-elastic-notch-2026-09-27/figures/transition-notch.png", None)],
        [("dynamic-branching-2026-09-27/figures/dynamic-branching-refinement.png", None)],
        [("thermofracture-2026-09-22/figures/thermofracture-verification.png", None)],
    ],
    "fig_fracture2": [
        [("spatial-fracture-complete-2026-09-22/figures/fracture-verification.png", (0, 0, 1, .5))],
        [("vacancy-fracture-2026-09-26/figures/vacancy-fracture.png", None)],
    ],
    "fig_finitestrain": [
        [("finite-elasticity-complete-2026-09-22/figures/finite-verification.png", None)],
        [("thermoplastic-2026-09-22/figures/thermoplastic-verification.png", None)],
        [("quadratic-mechanics-2026-09-22/figures/quadratic-verification.png", None)],
    ],
    "fig_copper": [
        [("calibration-copper-2026-09-27/figures/copper-self-diffusion.png", None),
         ("embryo-atomistics-mishin-2026-09-28/figures/embryo-emt-mishin.png", None),
         ("vacancy-loop-2026-09-28/figures/vacancy-loop.png", None),
         ("screw-core-2026-09-28/figures/screw-zjw-mishin.png", None)],
    ],
    "fig_clusters": [
        [("sft-2026-09-28/figures/sft.png", None)],
        [("critical-nucleus-2026-09-28/figures/critical-nucleus.png", None)],
    ],
    "fig_gridcost": [
        [("performance-complete-2026-09-13/figures/accuracy-cost.png", (0, 0, .5, 1)),
         ("graded-cost-2026-09-26/figures/graded-cost.png", None)],
        [("amr-message-passing-2026-09-26/figures/amr-message-passing.png", None)],
    ],
    "fig_costs2": [
        [("local-plasticity-2026-09-26/figures/local-plasticity.png", None)],
        [("thermochemical-cost-2026-09-22/figures/thermochemical-cost.png", None)],
        [("response-cache-optimized-2026-09-13/figures/cache-cost-accuracy.png", None)],
    ],
}

WIDTH_IN, GAP = 7.2, 0.012   # figure width (inches); horizontal gap as a fraction of the width
PRINT_IN = 7.0               # printed width of a full-width composite in the manuscript
LABELS = "abcdefghijklmnopqrstuvwxyz"
SRC = OUT / "src"            # figures regenerated with print-sized fonts by regen_sources.py


def load(src, box):
    """Open a source figure, preferring the regenerated copy; box is a fractional crop (x0, y0, x1, y1)."""
    path = SRC / src if (SRC / src).exists() else R / src
    im = Image.open(path).convert("RGB")
    if box:
        w, h = im.size
        im = im.crop((int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)))
    return im


def row_heights(rows):
    """Common height of each row (in units of the figure width) so that the row spans the full width."""
    return [(1 - GAP*(len(r) - 1))/sum(im.size[0]/im.size[1] for im in r) for r in rows]


def display_widths():
    """Printed width (inches) of each full source figure, keyed by file stem, for regen_sources.py."""
    out = {}
    for rows in FIGS.values():
        ims = [[load(src, box) for src, box in row] for row in rows]
        for row, h, ims_row in zip(rows, row_heights(ims), ims):
            for (src, box), im in zip(row, ims_row):
                frac = im.size[0]/im.size[1]*h/((box[2] - box[0]) if box else 1)
                stem = Path(src).stem
                out[stem] = min(out.get(stem, 99), frac*PRINT_IN)
    return out


NATIVE_DPI, MIN_SCALE = 160, 0.6   # report figures: ~10 pt text at 160 dpi; print at >= 6.5 pt


# Subplot grids (rows, columns) of the original report figures that are split into panels.
GRID = {
    "grain-mesh": (2, 2), "precipitate-pair-convergence": (2, 2), "frank-read-elastic": (1, 2),
    "spatial-branches": (2, 2), "graded-cost": (1, 2), "cache-cost-accuracy": (1, 2),
    "convergence": (1, 2), "triple-junctions": (1, 4), "grain-statistics": (2, 2), "coarsening-conditioning": (1, 3),
    "spherical-storage": (2, 2), "spherical-capture": (2, 2), "pore-kinetics": (2, 2), "segregation": (1, 3),
    "disclination-dynamics": (1, 2), "convergence-cost": (1, 3), "transfer-convergence": (1, 3),
    "coupled-loop-feedback": (2, 3), "reaction-kinetics": (2, 2), "transition-notch": (1, 3),
    "dynamic-branching-refinement": (1, 3), "fracture-verification": (1, 2), "vacancy-fracture": (1, 3),
    "sft": (1, 3), "critical-nucleus": (1, 3), "amr-message-passing": (1, 3), "local-plasticity": (1, 3),
}


# Panels kept (row-major indices) where only part of an original figure is needed, and sources whose
# panels are mostly field images with little text (a lower minimum print scale suffices).
KEEP = {
    "graded-cost": [0],
    "disclination-dynamics": [0],
    "grain-mesh": [3], "precipitate-pair-convergence": [1], "spatial-branches": [1],
    "frank-read-elastic": [0], "cache-cost-accuracy": [0],
    "triple-junctions": [2, 3], "grain-statistics": [1], "coarsening-conditioning": [1, 2],
    "spherical-storage": [0], "spherical-capture": [0, 1], "pore-kinetics": [0, 2], "segregation": [0, 2],
    "convergence-cost": [0], "coupled-loop-feedback": [1, 2], "reaction-kinetics": [0, 3],
    "transition-notch": [0, 1], "vacancy-fracture": [0, 2], "amr-message-passing": [1], "local-plasticity": [2],
}
# Figures whose first text line is a figure-wide title; it is removed before splitting.
SUPTITLE = {"triple-junctions", "grain-statistics", "spherical-capture", "spherical-storage", "pore-kinetics",
            "coupled-loop-feedback", "reaction-kinetics", "grain-mesh", "precipitate-pair-convergence",
            "spatial-branches", "loop-convergence"}
FIELD_SCALE = {"dynamic-branching-refinement": 0.3, "sft": 0.35, "critical-nucleus": 0.45,
               # small native figures (6-7 in wide) whose text is already large relative to the figure
               "copper-self-diffusion": 0.42, "embryo-emt-mishin": 0.42, "vacancy-loop": 0.42, "screw-zjw-mishin": 0.42}


def _cuts(profile, n, length, min_gap=8):
    """Cut positions for n equal-ish panels, snapped to the centre of the nearest white gap."""
    blank = ~(profile > 0.004*length)
    gaps, start = [], None
    for i, v in enumerate(blank):
        if v and start is None:
            start = i
        if not v and start is not None:
            if i - start >= min_gap:
                gaps.append((start + i)//2)
            start = None
    size = len(profile)
    cuts = [0]
    for k in range(1, n):
        target = k*size//n
        near = [g for g in gaps if abs(g - target) < 0.12*size]
        cuts.append(min(near, key=lambda g: abs(g - target)) if near else target)
    return cuts + [size]


def split_panels(im, grid):
    """Split a report figure into its subplot grid, cutting along the white gutters."""
    import numpy as np
    a = np.asarray(im.convert("L")) < 245
    H, W = a.shape
    nr, nc = grid
    ys = _cuts(a.sum(axis=1), nr, W)
    panels = []
    for y0, y1 in zip(ys, ys[1:]):
        xs = _cuts(a[y0:y1].sum(axis=0), nc, y1 - y0)
        panels += [im.crop((x0, y0, x1, y1)) for x0, x1 in zip(xs, xs[1:])]
    return panels


def drop_suptitle(im):
    """Remove the first line of text (a figure-wide title) above the panels."""
    import numpy as np
    a = np.asarray(im.convert("L")) < 245
    ink = a.sum(axis=1) > 0.004*a.shape[1]
    i = int(np.argmax(ink))                       # first inked row
    j = i + int(np.argmax(~ink[i:]))              # end of that text line
    return im.crop((0, j, im.size[0], im.size[1])) if j - i < 0.08*a.shape[0] else im


def items_of(src, box):
    """Return panels and the printed width each needs (inches) for one source figure."""
    im = load(src, box)
    stem = Path(src).stem
    if stem in SUPTITLE:
        im = drop_suptitle(im)
    grid = GRID.get(stem)
    panels = split_panels(im, grid) if grid else [im]
    if stem in KEEP:
        panels = [panels[i] for i in KEEP[stem]]
    if (SRC / src).exists():             # regenerated for the full printed width: keep that scale
        full = im.size[0]/((box[2] - box[0]) if box else 1)   # width of the uncropped regenerated figure
        return panels, [PRINT_IN*p.size[0]/full for p in panels]
    scale = FIELD_SCALE.get(stem, MIN_SCALE)
    return panels, [scale*p.size[0]/NATIVE_DPI for p in panels]


def flow(entries, width=PRINT_IN, gap_in=0.08):
    """Fill rows greedily. Each row is as tall as its most demanding panel requires (required printed
    width over aspect ratio), no taller, and holds panels while they fit the printed width."""
    rows, cur = [], []

    def height(row):
        return max(req/(p.size[0]/p.size[1]) for p, req, _ in row)

    def fits(row):
        h = height(row)
        return sum(p.size[0]/p.size[1]*h for p, _, _ in row) + gap_in*(len(row) - 1) <= 1.04*width   # rows may shrink 4%
    for item in entries:
        if cur and not fits(cur + [item]):
            rows.append(cur); cur = []
        cur.append(item)
    if cur:
        rows.append(cur)
    return rows


def compose(name, rows):
    entries = []
    for row in rows:
        for src, box in row:
            panels, reqs = items_of(src, box)
            entries += [(p, r, j == 0) for j, (p, r) in enumerate(zip(panels, reqs))]
    rows = flow(entries)
    label_in, vgap_in, gap_in = 0.16, 0.06, 0.08
    heights = []
    for r in rows:
        h = max(req/(p.size[0]/p.size[1]) for p, req, _ in r)
        w = sum(p.size[0]/p.size[1]*h for p, _, _ in r) + gap_in*(len(r) - 1)
        heights.append(h*min(1., PRINT_IN/w))          # a single oversized panel is capped at the width
    total_in = sum(heights) + (label_in + vgap_in)*len(rows)
    fig = plt.figure(figsize=(PRINT_IN, total_in))
    y, k = total_in, 0
    for r, h in zip(rows, heights):
        y -= label_in
        widths = [p.size[0]/p.size[1]*h for p, _, _ in r]
        x = (PRINT_IN - sum(widths) - gap_in*(len(r) - 1))/2
        for (im, _, first), w in zip(r, widths):
            ax = fig.add_axes((x/PRINT_IN, (y - h)/total_in, w/PRINT_IN, h/total_in))
            ax.imshow(im, interpolation="lanczos"); ax.set_axis_off()
            if first:
                fig.text(x/PRINT_IN, (y + 0.02)/total_in, f"({LABELS[k]})", fontsize=9, fontweight="bold", va="bottom")
                k += 1
            x += w + gap_in
        y -= h + vgap_in
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(name, k, "sources,", sum(len(r) for r in rows), "panels, height", round(total_in, 2), "in")


if __name__ == "__main__":
    from make_si_figs import FIGS as REPLOTTED     # figures replotted from data replace their composites
    for name, rows in FIGS.items():
        if name not in REPLOTTED:
            compose(name, rows)
