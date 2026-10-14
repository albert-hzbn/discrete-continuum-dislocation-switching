"""Nested tensor refinement, conservative overlap transfer, and explicit audits."""

import numpy as np
from .grid import Grid


def focused_grid(length, h, bounds, factor=4):
    n = int(round(length / h))
    if not np.isclose(n * h, length):
        raise ValueError("Length must be an integer multiple of fine spacing.")
    edges = np.linspace(-length / 2, length / 2, n + 1)
    axes = []
    for low, high in bounds:
        indices = np.arange(n + 1)
        keep = (indices % factor == 0) | ((edges >= low) & (edges <= high))
        keep[[0, -1]] = True
        axes.append(edges[keep])
    return Grid(*axes)


def overlap_matrix(old_edges, new_edges):
    left = np.maximum(new_edges[:-1, None], old_edges[None, :-1])
    right = np.minimum(new_edges[1:, None], old_edges[None, 1:])
    return np.maximum(0, right - left) / np.diff(new_edges)[:, None]


def transfer(old, new, field):
    wx, wy = overlap_matrix(old.xe, new.xe), overlap_matrix(old.ye, new.ye)
    return wy @ field @ wx.T


def bounds_for(grid, field, points=(), threshold=0.999, buffer=3.0):
    active = np.abs(field) < threshold
    if np.any(active):
        xb = [float(grid.X[active].min()), float(grid.X[active].max())]
        yb = [float(grid.Y[active].min()), float(grid.Y[active].max())]
    else:
        xb, yb = [0.0, 0.0], [0.0, 0.0]
    for x, y in points:
        xb = [min(xb[0], x), max(xb[1], x)]
        yb = [min(yb[0], y), max(yb[1], y)]
    return ((xb[0] - buffer, xb[1] + buffer), (yb[0] - buffer, yb[1] + buffer))


def adapt(grid, fields, h, length, energy, points=(), tolerance=1e-3):
    """Reject transfers that increase energy or materially change reconstruction."""
    candidate = focused_grid(length, h, bounds_for(grid, fields[0], points))
    if np.array_equal(candidate.xe, grid.xe) and np.array_equal(candidate.ye, grid.ye):
        return grid, fields, {"changed": False, "reason": "same_grid"}
    mapped = [transfer(grid, candidate, f) for f in fields]
    old_e, new_e = energy(grid, fields), energy(candidate, mapped)
    # This common fine grid is an occasional diagnostic, not persistent solve state.
    audit = Grid.uniform(length, int(round(length / h)))
    errs = []
    audit_old, audit_new = [], []
    mass_errors = []
    for old_f, new_f in zip(fields, mapped):
        a, b = transfer(grid, audit, old_f), transfer(candidate, audit, new_f)
        audit_old.append(a)
        audit_new.append(b)
        errs.append(float(np.sqrt(audit.integrate((a - b) ** 2) / max(audit.integrate(a * a), 1e-30))))
        mass_errors.append(abs(grid.integrate(old_f) - candidate.integrate(new_f)))
    audit_jump = energy(audit, audit_new) - energy(audit, audit_old)
    e_tol = 1e-10 * max(1, abs(old_e))
    m_tol = 1e-10 * max(1, *(abs(grid.integrate(f)) for f in fields))
    accepted = max(errs) <= tolerance and new_e <= old_e + e_tol and audit_jump <= e_tol and max(mass_errors) <= m_tol
    record = {"changed": bool(accepted), "reason": "accepted" if accepted else "audit_rejected",
              "relative_l2": max(errs), "energy_jump": float(new_e - old_e),
              "common_grid_energy_jump": float(audit_jump), "mass_error": max(mass_errors),
              "old_cells": grid.size, "proposed_cells": candidate.size}
    return (candidate, mapped, record) if accepted else (grid, fields, record)
