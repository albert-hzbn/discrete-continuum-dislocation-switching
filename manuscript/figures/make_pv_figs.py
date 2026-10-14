"""PyVista 3D renderings for Sections 98 (creep cavitation in copper) and 99 (ExaDiS vein/channel allocator).

Run from the repository root:
    python manuscript/figures/make_pv_figs.py creep     # needs results/creep-cavitation-3d-2026-09-30/A_30_96_3d.npz
    python manuscript/figures/make_pv_figs.py network   # needs results/allocator-3d-2026-09-30/{D,H}_s2*.vtk
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
pv.global_theme.font.family = "arial"
pv.global_theme.font.size = 22
pv.global_theme.font.label_size = 20
pv.global_theme.font.title_size = 24
pv.global_theme.font.color = "#1f1f1e"

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
R3 = ROOT / "results" / "creep-cavitation-3d-2026-09-30"
RA = ROOT / "results" / "allocator-3d-2026-09-30"
BLUE, ORANGE, AQUA, INK = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1e"


def grid(vol, spacing):
    g = pv.ImageData(dimensions=np.array(vol.shape) + 1, spacing=(spacing,)*3)
    g.cell_data["v"] = np.asarray(vol, dtype=np.float32).ravel(order="F")
    return g.cell_data_to_point_data()


def creep(S=30):
    d = np.load(R3 / f"A_{S}_48_3d.npz")
    i = 0                                               # last saved state before rupture
    p, szz = d["p"][i].astype(np.float32), d["szz"][i]
    n = p.shape[0]; h = 10.0/n                         # micrometres
    mu = 4.21e10*(1 - 0.54*(773 - 300)/1356); Su = S*1e6/mu
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1250))
    pl.set_background("white")
    cav = grid(p, h).contour([0.5], scalars="v")
    pl.add_mesh(cav, color="#b8b4a8", smooth_shading=True, opacity=0.35)
    sl = grid(szz/Su, h).slice(normal="z", origin=(5, 5, 5.0))
    pl.add_mesh(sl, scalars="v", cmap="RdBu_r", clim=(-10, 10),
                scalar_bar_args=dict(title="sigma_zz / S", vertical=True, position_x=0.86, position_y=0.2,
                                     height=0.6, title_font_size=40, label_font_size=36, fmt="%.0f"))
    pl.add_mesh(pv.Box(bounds=(0, 10, 0, 10, 0, 10)).outline(), color=INK, line_width=2)
    pl.camera_position = [(27, -23, 19), (5, 5, 4.6), (0, 0, 1)]
    pl.screenshot(OUT / "fig_creep3d_render.png", transparent_background=False)
    print("wrote fig_creep3d_render.png", "t =", float(d["t"][i]))


def network(seed=1, alpha=None):
    alpha = alpha if alpha is not None else json.loads((RA / "calibration.json").read_text())["alpha"]
    L = 8000*0.255e-3                                   # micrometres per b: 2.04 um box
    s = 0.255e-3
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2600, 1250), border=False)
    for col, (tag, title) in enumerate(((f"D_s{seed}", "Full DDD (ExaDiS)"), (f"H_s{seed}_a{alpha:g}", "Allocator"))):
        pl.subplot(0, col)
        pl.set_background("white")
        m = pv.read(RA / f"{tag}.vtk")
        m = m.extract_cells(np.where(m.celltypes == pv.CellType.POLY_LINE)[0])      # drop the simulation-box cell
        m.points *= s
        b = np.abs(np.asarray(m.cell_data["Burgers"])); fam = np.argmax(b, axis=1) + 3*(np.sum(b > 1e-6, axis=1) == 2)
        m.cell_data["family"] = fam % 6
        pl.add_mesh(m, scalars="family", cmap=["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2", "#c9a227", "#d6457a"], line_width=2, clim=(0, 5),
                    render_lines_as_tubes=True, show_scalar_bar=False)
        if col == 1:
            wall = pv.Box(bounds=(0, L, 0, L, (4000 - 800)*s, (4000 + 800)*s))
            pl.add_mesh(wall, color=ORANGE, opacity=0.18)
            pl.add_mesh(wall.outline(), color=ORANGE, line_width=2)
        pl.add_mesh(pv.Box(bounds=(0, L, 0, L, 0, L)).outline(), color=INK, line_width=2)
        pl.add_text(title, position="upper_left", font_size=30, color=INK)
        pl.camera_position = [(5.2, -4.2, 3.2), (L/2, L/2, L/2), (0, 0, 1)]
    pl.screenshot(OUT / "fig_alloc3d_render.png")
    print("wrote fig_alloc3d_render.png")


if __name__ == "__main__":
    {"creep": creep, "network": network}[sys.argv[1]]()
