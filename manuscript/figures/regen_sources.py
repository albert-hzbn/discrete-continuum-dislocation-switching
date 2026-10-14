"""Regenerate the study figures used by the manuscript composites with print-sized fonts.

Run from the repository root:
    python manuscript/figures/regen_sources.py

Each report script is run from its saved results in an isolated copy of the repository (manuscript/figures/_work),
so results/ and docs/ are not modified. Before every save, a wrapper raises each text element to FONT_PT at the
width at which the figure is printed in the manuscript (from make_composites.display_widths()). The regenerated
PNGs are copied to manuscript/figures/src/<results dir>/figures/, where make_composites.py looks for them first.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = HERE / "_work"
SRC = HERE / "src"
FONT_PT = 8.0

# (report script, arguments, results directories it reads)
JOBS = [
    ("microstructure_report.py", ["--input", "results/microstructure-2026-09-13"], ["microstructure-2026-09-13"]),
    ("kinetic_laws_report.py", ["--input", "results/kinetic-laws-2026-09-26"], ["kinetic-laws-2026-09-26"]),
    ("precipitate_interactions_report.py", ["--input", "results/precipitate-interactions-2026-09-13"], ["precipitate-interactions-2026-09-13"]),
    ("coarsening_conditioning_report.py", [], ["coarsening-conditioning-2026-09-26"]),
    ("grain_mesh_report.py", ["--input", "results/grain-mesh-2026-09-13"], ["grain-mesh-2026-09-13", "microstructure-2026-09-13"]),
    ("spherical_sinks_report.py", ["--input", "results/spherical-sinks-2026-09-13"], ["spherical-sinks-2026-09-13"]),
    ("pore_growth_report.py", ["--input", "results/pore-growth-2026-09-26"], ["pore-growth-2026-09-26"]),
    ("curved_loops_report.py", ["--input", "results/curved-loops-fcc-2026-09-12"], ["curved-loops-fcc-2026-09-12"]),
    ("frank_read_elastic_report.py", [], ["frank-read-elastic-2026-09-28"]),
    ("interface_graphs_report.py", ["--input", "results/interface-graphs-2026-09-12"], ["interface-graphs-2026-09-12"]),
    ("three_dimensional_report.py", ["--input", "results/three-dimensional-2026-09-12"], ["three-dimensional-2026-09-12"]),
    ("representation_transfer_report.py", ["--input", "results/representation-transfer-2026-09-13"], ["representation-transfer-2026-09-13"]),
    ("transition_notch_report.py", ["--input", "results/transition-elastic-notch-2026-09-27", "--elastic"], ["transition-elastic-notch-2026-09-27"]),
    ("dynamic_branching_report.py", ["--input", "results/dynamic-branching-2026-09-27"], ["dynamic-branching-2026-09-27"]),
    ("thermofracture_report.py", ["--input", "results/thermofracture-2026-09-22"], ["thermofracture-2026-09-22"]),
    ("spatial_fracture_report.py", ["--input", "results/spatial-fracture-complete-2026-09-22"], ["spatial-fracture-complete-2026-09-22"]),
    ("vacancy_fracture_report.py", ["--input", "results/vacancy-fracture-2026-09-26"], ["vacancy-fracture-2026-09-26"]),
    ("finite_elasticity_report.py", ["--input", "results/finite-elasticity-complete-2026-09-22"], ["finite-elasticity-complete-2026-09-22"]),
    ("thermoplastic_report.py", ["--input", "results/thermoplastic-2026-09-22", "--nonaffine", "results/thermoplastic-nonaffine-2026-09-22"],
     ["thermoplastic-2026-09-22", "thermoplastic-nonaffine-2026-09-22"]),
    ("quadratic_mechanics_report.py", ["--input", "results/quadratic-mechanics-2026-09-22", "--q1-manufactured", "results/finite-elasticity-complete-2026-09-22",
                                       "--q1-base", "results/thermochemical-2026-09-22", "--q1-fine", "results/thermochemical-refinement-2026-09-22"],
     ["quadratic-mechanics-2026-09-22", "finite-elasticity-complete-2026-09-22", "thermochemical-2026-09-22", "thermochemical-refinement-2026-09-22"]),
    ("performance_audit_report.py", ["--input", "results/performance-complete-2026-09-13"], ["performance-complete-2026-09-13"]),
    ("graded_cost_report.py", ["--input", "results/graded-cost-2026-09-26"], ["graded-cost-2026-09-26"]),
    ("amr_message_passing_report.py", ["--input", "results/amr-message-passing-2026-09-26"], ["amr-message-passing-2026-09-26"]),
    ("local_plasticity_report.py", ["--input", "results/local-plasticity-2026-09-26"], ["local-plasticity-2026-09-26"]),
    ("thermochemical_cost_report.py", ["--input", "results/thermochemical-cost-2026-09-22"], ["thermochemical-cost-2026-09-22"]),
    ("response_cache_report.py", ["--input", "results/response-cache-optimized-2026-09-13"], ["response-cache-optimized-2026-09-13", "response-cache-2026-09-13"]),
    ("sft_report.py", [], ["sft-2026-09-28"]),
    ("screw_core_report.py", [], ["screw-core-2026-09-28"]),
]

WRAPPER = r'''
import json, os, runpy, sys
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
DISPLAY = json.loads(os.environ["FIG_DISPLAY"]); FONT_PT = float(os.environ["FIG_FONT_PT"])
# Manuscript style (make_figures.py): validated palette, recessive axes and grid, thicker marks.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2", "#c9a227", "#d6457a", "#4f9fb8", "#6b6a64"]
INK, MUTED, GRIDC = "#1f1f1e", "#6b6a64", "#dddcd6"
plt.rcParams.update({
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.5, "lines.linewidth": 1.8, "lines.markersize": 6,
    "legend.frameon": False, "axes.linewidth": 0.9, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
})
TAB = {mcolors.to_hex(c): PALETTE[i] for i, c in enumerate(["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:olive", "tab:pink", "tab:cyan", "tab:gray"])}
TAB.update({mcolors.to_hex("tab:red"): "#d6457a", mcolors.to_hex("tab:brown"): "#8a6a4a"})
def _remap(c):
    try:
        h = mcolors.to_hex(c)
    except Exception:
        return None
    return TAB.get(h)
# Field files that no longer exist become NaN placeholders, so the rest of a report still renders; any figure that
# touched a placeholder is recorded as incomplete and is not used by the composites.
import numpy as np
_TOUCHED = [False]
class _Missing(np.ndarray):
    def __new__(cls):
        return np.full((8, 8), np.nan).view(cls)
    def reshape(self, *shape, **k):
        _TOUCHED[0] = True
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return np.full(shape, np.nan)
    def __getitem__(self, key):
        _TOUCHED[0] = True
        return np.asarray(self)[key] if not isinstance(key, str) else _Missing()
class _MissingFile(dict):
    files = ["0"]
    def __getitem__(self, key):
        _TOUCHED[0] = True
        return _Missing()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def close(self): pass
_load = np.load
def _safe_load(f, *a, **k):
    try:
        return _load(f, *a, **k)
    except FileNotFoundError:
        _TOUCHED[0] = True
        return _MissingFile() if str(f).endswith(".npz") else _Missing()
np.load = _safe_load
INCOMPLETE = []
_save = Figure.savefig
def savefig(self, fname, *a, **k):
    if _TOUCHED[0]:
        INCOMPLETE.append(os.path.basename(str(fname)))
        _TOUCHED[0] = False
        with open(os.environ["FIG_INCOMPLETE"], "a") as fh:
            fh.write(os.path.abspath(str(fname)) + chr(10))
    for ax in self.axes:
        for ln in ax.get_lines():
            r = _remap(ln.get_color())
            if r: ln.set_color(r)
            if ln.get_linewidth() < 1.4 and ln.get_linestyle() not in ("None", "", " "): ln.set_linewidth(1.6)
            if ln.get_marker() not in (None, "None", "", " ", ",") and ln.get_markersize() < 5: ln.set_markersize(5.5)
        for coll in ax.collections:
            try:
                fc = coll.get_facecolors()
                if len(fc):
                    coll.set_facecolors([_remap(c) or c for c in fc])
            except Exception:
                pass
        for pt in ax.patches:
            r = _remap(pt.get_facecolor())
            if r: pt.set_facecolor(r)
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "log":
                lo, hi = sorted(ax.get_xlim() if axis is ax.xaxis else ax.get_ylim())
                if lo > 0 and hi/lo > 12:           # at least one labelled decade: drop the minor-tick labels
                    axis.set_minor_formatter(NullFormatter())
        for sp in ("top", "right"):
            if ax.spines[sp].get_visible() and ax.name == "rectilinear" and not ax.get_images():
                ax.spines[sp].set_visible(False)
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_linewidth(0); leg.get_frame().set_alpha(0.85)
    name = os.path.basename(str(fname)); stem = os.path.splitext(name)[0]
    shown = DISPLAY.get(stem, 7.2)
    size = FONT_PT*self.get_figwidth()/shown
    for t in self.findobj(Text):
        if t.get_text() and t.get_fontsize() < size:
            t.set_fontsize(size)
    import textwrap
    for ax in self.axes:
        width_pt = ax.get_position().width*self.get_figwidth()*72
        chars = max(12, int(width_pt/(0.55*size)))
        for t in (ax.title, ax._left_title, ax._right_title):
            if t.get_text() and len(t.get_text()) > chars:
                t.set_text(textwrap.fill(t.get_text(), chars))
        ax.tick_params(labelsize=size)
        ax.xaxis.get_offset_text().set_fontsize(size); ax.yaxis.get_offset_text().set_fontsize(size)
    if self.get_layout_engine() is None:
        try:
            self.set_layout_engine("constrained")
        except Exception:
            pass
    return _save(self, fname, *a, **k)
Figure.savefig = savefig
script = sys.argv[1]; sys.argv = sys.argv[1:]
# Resilient mode: load the report without running it, wrap each of its plotting functions so that one that fails
# for lack of field data is skipped, then call main(). Scripts without main() run as usual.
import types, functools, traceback
ns = runpy.run_path(script, run_name="__report__")
main = ns.get("main")
if isinstance(main, types.FunctionType):
    g = main.__globals__
    def tolerant(f):
        @functools.wraps(f)
        def call(*a, **k):
            try:
                return f(*a, **k)
            except Exception as e:
                print("skipped", f.__name__, type(e).__name__, str(e)[:120], file=sys.stderr)
                plt.close("all"); _TOUCHED[0] = False
                return None
        return call
    for key, val in list(g.items()):
        if isinstance(val, types.FunctionType) and key != "main" and val.__code__.co_filename == main.__code__.co_filename:
            g[key] = tolerant(val)
    main()
else:
    runpy.run_path(script, run_name="__main__")
'''


def mirror(dirs):
    for sub in ("src", "benchmarks"):
        if not (WORK / sub).exists():
            shutil.copytree(ROOT / sub, WORK / sub, ignore=shutil.ignore_patterns("__pycache__"))
    (WORK / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "docs" / "moves.json", WORK / "docs" / "moves.json")
    for d in dirs:
        if not (WORK / "results" / d).exists():
            shutil.copytree(ROOT / "results" / d, WORK / "results" / d)


def main():
    # Every source is regenerated for the full printed width; make_composites.py splits it into panels
    # and prints each panel at the width it had in the full figure, so text keeps FONT_PT.
    display = {}
    WORK.mkdir(exist_ok=True)
    (WORK / "wrapper.py").write_text(WRAPPER)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(WORK / "src"), str(WORK / "benchmarks")]),
               FIG_DISPLAY=json.dumps(display), FIG_FONT_PT=str(FONT_PT), MPLBACKEND="Agg",
               FIG_INCOMPLETE=str(WORK / "incomplete.txt"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    failed = []
    (WORK / "incomplete.txt").write_text("")
    started = __import__("time").time()
    for script, args, dirs in JOBS:
        mirror(dirs)
        r = subprocess.run([sys.executable, str(WORK / "wrapper.py"), str(WORK / "benchmarks" / script), *args],
                           cwd=WORK, env=env, capture_output=True, text=True, timeout=1800)
        ok = r.returncode == 0
        print(("ok    " if ok else "FAIL  ") + script, flush=True)
        if not ok:
            failed.append(script); print(r.stderr[-1500:], flush=True)
    incomplete = {Path(x).resolve() for x in (WORK / "incomplete.txt").read_text().splitlines() if x}
    skipped = []
    for png in (WORK / "results").rglob("figures/*.png"):
        if png.stat().st_mtime < started:   # copied original, not regenerated
            continue
        if png.resolve() in incomplete:     # drew from a missing field file: keep the original figure
            skipped.append(png.name); continue
        dest = SRC / png.relative_to(WORK / "results")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, dest)
    print("failed:", failed)
    print("incomplete (original kept):", skipped)


if __name__ == "__main__":
    main()
