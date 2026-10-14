"""Use-case gallery: one field panel from each study's own data-generated report figure.

Run from the repository root:
    python manuscript/figures/make_usecases.py

Each panel is cropped (pixel box in the source image) from a figure written by that study's report script from its
saved results; the caption in sections/usecases.tex gives the study and quantities. Output: fig_usecases.pdf/.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
R = ROOT / "results"

# (label, source figure, crop box in source pixels (left, top, right, bottom), panel title)
PANELS = [
    ("(a)", "dynamic-branching-2026-09-27/figures/dynamic-branching-thermal.png", (7, 73, 723, 431),
     "Dynamic branching: damage"),
    ("(b)", "dynamic-branching-2026-09-27/figures/dynamic-branching-thermal.png", (843, 73, 1559, 431),
     "Dynamic branching: fracture heat"),
    ("(c)", "crack-inclusion-2026-09-26/figures/crack-inclusion.png", (1904, 40, 2453, 589),
     "Crack arrest at a tough inclusion"),
    ("(d)", "precipitate-interactions-2026-09-13/figures/precipitate-pair-fields.png", (784, 202, 1226, 495),
     "Coherent precipitate pair: $\\sigma_{xx}$"),
    ("(e)", "pore-topology-2026-09-26/figures/pore-topology.png", (924, 43, 1359, 600),
     "Pore coalescence"),
    ("(f)", "microstructure-2026-09-13/figures/triple-junctions.png", (1241, 96, 1671, 523),
     "Grain triple junction"),
]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), layout="constrained")
    for ax, (label, src, box, title) in zip(axes.ravel(), PANELS):
        im = Image.open(R / src).convert("RGB").crop(box)
        ax.imshow(im, interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(f"{label} {title}", fontsize=8, loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_usecases.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote fig_usecases")


if __name__ == "__main__":
    main()
