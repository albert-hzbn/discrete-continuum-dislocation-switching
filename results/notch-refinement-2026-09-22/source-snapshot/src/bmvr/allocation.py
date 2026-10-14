"""Joint two-grid indicators and greedy tensor-grid allocation.

The indicator measures operator discrepancy, not an exact Galerkin gain.
All full-grid audits belong in allocation cost when benchmarking.
"""

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix
from .grid import Grid
from .adaptivity import overlap_matrix
from .defects import core_kernel


class TransferCache:
    def __init__(self):
        self.maps = {}

    def axis(self, old, new):
        key = (old.tobytes(), new.tobytes())
        if key not in self.maps:
            self.maps[key] = csr_matrix(overlap_matrix(old, new))
        return self.maps[key]

    def project(self, old, new, field):
        wx, wy = self.axis(old.xe, new.xe), self.axis(old.ye, new.ye)
        return (wx @ (wy @ field).T).T


def inverse_diffusion_map(grid, rate, mobility=1.0):
    """Positive cell shares of ||rate||^2_{(-mobility*Laplacian)^dagger}."""
    if grid.periodic or not np.allclose(grid.dx, grid.dx[0]) or not np.allclose(grid.dy, grid.dy[0]):
        raise ValueError("A uniform no-flux audit grid is required.")
    if mobility <= 0:
        raise ValueError("Mobility must be positive.")
    mass_rate = grid.integrate(rate)
    scale = max(1.0, grid.integrate(np.abs(rate)))
    if abs(mass_rate) > 1e-10 * scale:
        raise ValueError("The conserved residual has a nonzero inventory rate.")
    centered = rate - mass_rate / grid.area
    nx, ny = len(grid.x), len(grid.y)
    lam = (4 * np.sin(np.pi * np.arange(nx) / (2 * nx))[None, :] ** 2 / grid.dx[0] ** 2
           + 4 * np.sin(np.pi * np.arange(ny) / (2 * ny))[:, None] ** 2 / grid.dy[0] ** 2)
    coeff = dctn(centered, type=2, norm="ortho")
    potential = idctn(np.divide(coeff, lam, out=np.zeros_like(coeff), where=lam > 0), type=2, norm="ortho")
    cell = np.zeros(grid.shape)
    fx = np.diff(potential, axis=1) ** 2 * grid.dy[:, None] / np.diff(grid.x) / mobility
    fy = np.diff(potential, axis=0) ** 2 * grid.dx[None, :] / np.diff(grid.y)[:, None] / mobility
    cell[:, :-1] += fx / 2
    cell[:, 1:] += fx / 2
    cell[:-1] += fy / 2
    cell[1:] += fy / 2
    return cell, potential, float(mass_rate)


def two_grid_indicator(grid, fields, audit, evaluate, cache=None,
                       relative=(.05, .05, .05), absolute=1e-7,
                       phase_mobility=1., diffusion_mobility=1.):
    cache = cache or TransferCache()
    lifted = [cache.project(grid, audit, f) for f in fields]
    coarse, fine = evaluate(grid, fields), evaluate(audit, lifted)
    re = fine["rates"][0] - cache.project(grid, audit, coarse["rates"][0])
    rc = fine["rates"][1] - cache.project(grid, audit, coarse["rates"][1])
    eta_map = audit.vol * re ** 2 / phase_mobility
    c_map, _, inventory_rate = inverse_diffusion_map(audit, rc, diffusion_mobility)
    phase_norm = audit.integrate(fine["rates"][0] ** 2) / phase_mobility
    c_norm = float(inverse_diffusion_map(audit, fine["rates"][1], diffusion_mobility)[0].sum())
    line_map = np.zeros(audit.shape)
    line_norm = 0.
    points = fine.get("line_points", np.empty((0, 2)))
    if len(points):
        mobility = float(fine["line_mobility"])
        difference = fine["line_velocity"] - coarse["line_velocity"]
        for i, point in enumerate(points):
            rho, _, _ = core_kernel(audit, point, fine["line_core"])
            line_map += audit.vol * rho * float(difference[i] @ difference[i]) / mobility
        line_norm = float(np.sum(fine["line_velocity"] ** 2) / mobility)
    maps = [eta_map, c_map, line_map]
    norms = np.array([phase_norm, c_norm, line_norm])
    squared = np.array([m.sum() for m in maps])
    budgets = np.asarray(relative) ** 2 * norms + absolute ** 2
    ratios = np.sqrt(squared / budgets)
    ranking_map = sum(m / b for m, b in zip(maps, budgets))
    return {"squared": squared.tolist(), "full_norm_squared": norms.tolist(),
            "budget_squared": budgets.tolist(), "budget_ratios": ratios.tolist(),
            "met": bool(np.all(ratios <= 1)), "inventory_rate_roundoff": inventory_rate,
            "line_block_present": bool(len(points)), "ranking_map": ranking_map}


def audit_grid(grid, h):
    nx = int(round((grid.xe[-1] - grid.xe[0]) / h))
    ny = int(round((grid.ye[-1] - grid.ye[0]) / h))
    return Grid(np.linspace(grid.xe[0], grid.xe[-1], nx + 1),
                np.linspace(grid.ye[0], grid.ye[-1], ny + 1))


def seed_grid(source, fields, audit, cache, points=(), core=1., factor=4,
              geometry=False, buffer=1.5):
    v, c = [cache.project(source, audit, f) for f in fields]
    mask = np.abs(v) < .99999
    if geometry:
        mask |= np.abs(c - np.median(c)) > 1e-5
    for x, y in points:
        mask |= (np.abs(audit.X - x) <= 4 * core) & (np.abs(audit.Y - y) <= 4 * core)
    masks = [mask.any(axis=0), mask.any(axis=1)]
    edges = []
    for full, active in zip((audit.xe, audit.ye), masks):
        active = binary_dilation(active, iterations=max(1, int(np.ceil(buffer / audit.hmin))))
        if geometry and active.any():
            lo, hi = np.flatnonzero(active)[[0, -1]]
            active[lo:hi + 1] = True
        keep = np.arange(len(full)) % factor == 0
        keep[:-1] |= active
        keep[1:] |= active
        keep[[0, -1]] = True
        edges.append(full[keep])
    return Grid(*edges)


def split_candidates(grid, audit, indicator, fields_count=2, mode="cost"):
    """Strip indicator mass is a benefit proxy; actual residual is recomputed."""
    candidates = []
    for axis, edges, full, centers, other in (
        ("x", grid.xe, audit.xe, audit.x, len(grid.y)),
        ("y", grid.ye, audit.ye, audit.y, len(grid.x)),
    ):
        strip_mass = indicator["ranking_map"].sum(axis=0 if axis == "x" else 1)
        for low, high in zip(edges[:-1], edges[1:]):
            extra = full[(full > low + 1e-12) & (full < high - 1e-12)]
            if not len(extra):
                continue
            benefit = float(strip_mass[(centers > low) & (centers < high)].sum())
            cost = fields_count * len(extra) * other
            score = benefit / cost if mode == "cost" else benefit
            candidates.append({"axis": axis, "low": float(low), "high": float(high),
                               "extra": extra, "benefit": benefit, "added_dofs": int(cost), "score": score})
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def plan_grid(source, fields, evaluate, h, mode="cost", relative=(.05, .05, .05),
              absolute=1e-7, max_splits=256, max_cells=None, refine_only=False):
    if mode not in ("cost", "residual", "geometry"):
        raise ValueError("Unknown allocation strategy.")
    cache = TransferCache()
    audit = audit_grid(source, h)
    source_info = evaluate(source, fields)
    grid = seed_grid(source, fields, audit, cache, source_info.get("line_points", ()),
                     source_info.get("line_core", 1.), geometry=mode == "geometry")
    if refine_only:
        grid = Grid(np.union1d(grid.xe, source.xe), np.union1d(grid.ye, source.ye))
    if max_cells is not None and grid.size > max_cells:
        raise ValueError("Mandatory interface/core resolution exceeds the cell budget.")
    selected = []
    reason = "geometry_baseline"
    for iteration in range(max_splits + 1):
        mapped = [cache.project(source, grid, f) for f in fields]
        indicator = two_grid_indicator(grid, mapped, audit, evaluate, cache, relative, absolute)
        if indicator["met"] or mode == "geometry":
            reason = "budgets_met" if indicator["met"] else "geometry_budgets_unmet"
            break
        if iteration == max_splits:
            reason = "split_limit"
            break
        choices = split_candidates(grid, audit, indicator, len(fields), mode)
        choices = [c for c in choices if max_cells is None or grid.size + c["added_dofs"] / len(fields) <= max_cells]
        if not choices:
            reason = "no_admissible_split"
            break
        chosen = choices[0]
        xe, ye = grid.xe, grid.ye
        if chosen["axis"] == "x":
            xe = np.sort(np.r_[xe, chosen["extra"]])
        else:
            ye = np.sort(np.r_[ye, chosen["extra"]])
        grid = Grid(xe, ye)
        selected.append({k: v for k, v in chosen.items() if k != "extra"})
    record = {k: v for k, v in indicator.items() if k != "ranking_map"}
    record.update({"strategy": mode, "reason": reason, "selected": selected,
                   "candidate_cells": grid.size, "audit_cells": audit.size,
                   "full_operator_audits": iteration + 1, "refine_only": bool(refine_only)})
    return grid, mapped, record


def allocate(source, fields, evaluate, h, mode="cost", relative=(.05, .05, .05),
             transfer_tolerance=1e-3, max_splits=256, refine_only=False):
    candidate, mapped, record = plan_grid(source, fields, evaluate, h, mode, relative,
                                         max_splits=max_splits, refine_only=refine_only)
    cache, audit = TransferCache(), audit_grid(source, h)
    old_fine = [cache.project(source, audit, f) for f in fields]
    new_fine = [cache.project(candidate, audit, f) for f in mapped]
    reconstruction = [np.sqrt(audit.integrate((a - b) ** 2) / max(audit.integrate(a * a), 1e-30))
                      for a, b in zip(old_fine, new_fine)]
    mass = [abs(source.integrate(a) - candidate.integrate(b)) for a, b in zip(fields, mapped)]
    before = evaluate(source, fields)["energy"]
    jump = evaluate(candidate, mapped)["energy"] - before
    common_jump = evaluate(audit, new_fine)["energy"] - evaluate(audit, old_fine)["energy"]
    energy_tolerance = 1e-10 * max(1., abs(before))
    mass_tolerance = 1e-10 * max(1., *(abs(source.integrate(f)) for f in fields))
    admissible = (max(reconstruction) <= transfer_tolerance and max(mass) <= mass_tolerance
                  and jump <= energy_tolerance and common_jump <= energy_tolerance
                  and (record["met"] or mode == "geometry"))
    record.update({"accepted": bool(admissible), "relative_transfer_error": [float(x) for x in reconstruction],
                   "mass_error": [float(x) for x in mass], "energy_jump": float(jump),
                   "common_grid_energy_jump": float(common_jump), "source_cells": source.size})
    if not admissible:
        return source, fields, record
    return candidate, mapped, record
