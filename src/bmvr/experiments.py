"""Executable cases for the explicitly restricted models in the theory document."""

from time import perf_counter
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import factorized
from scipy.optimize import minimize
from scipy.integrate import trapezoid
from .grid import Grid, circle
from .energy import DoubleWell, grain_energy, grain_force, pore_energy, pore_forces
from .solvers import ac_step, ch_step, grain_step, project_simplex
from .adaptivity import focused_grid, adapt, transfer
from .diagnostics import radius, energy_increase, structure_factor
from .elasticity import coherent_elasticity
from .defects import Screws, binding, screw_stress, coupled_energy_forces
from .reactions import annihilate


def result(case, name, variant="reference"):
    return {"case": case, "name": name, "variant": variant, "metrics": {},
            "checks": {}, "limitations": [], "timings": {"initialization_s": 0.0,
            "solver_s": 0.0, "adaptation_s": 0.0, "diagnostics_s": 0.0,
            "compilation_s": 0.0}, "status": "partial"}


def state_bytes(grid, *fields):
    """Stored numerical arrays only; peak RSS is measured separately by the runner."""
    return sum(a.nbytes for a in (grid.xe, grid.ye, grid.dx, grid.dy, grid.x, grid.y,
                                  grid.X, grid.Y, grid.vol, *fields))


def ac_run(n=256, length=128.0, initial_radius=14.0, end=12.0, dt=.02, adaptive=False):
    out = result(1, "Allen-Cahn circular inclusion", "adaptive" if adaptive else "reference")
    start = perf_counter()
    h = length / n
    grid = focused_grid(length, h, ((-25, 25), (-25, 25))) if adaptive else Grid.uniform(length, n)
    v, model = circle(grid, initial_radius), DoubleWell()
    initial = v.copy()
    times, energies, radii, cells, events = [0.0], [model.value(grid, v)], [radius(grid, v)], [grid.size], []
    out["timings"]["initialization_s"] = perf_counter() - start
    steps = int(np.ceil(end / dt))
    dt = end / steps
    for step in range(1, steps + 1):
        start = perf_counter()
        v = ac_step(grid, v, model, dt)
        out["timings"]["solver_s"] += perf_counter() - start
        if adaptive and step % 100 == 0:
            start = perf_counter()
            grid, fields, record = adapt(grid, [v], h, length, lambda g, f: model.value(g, f[0]))
            v = fields[0]
            events.append(record)
            out["timings"]["adaptation_s"] += perf_counter() - start
        if step % 25 == 0 or step == steps:
            start = perf_counter()
            times.append(step * dt)
            energies.append(model.value(grid, v))
            radii.append(radius(grid, v))
            cells.append(grid.size)
            out["timings"]["diagnostics_s"] += perf_counter() - start
    slope = float(np.polyfit(times, np.square(radii), 1)[0])
    # Profile error in a strip through the center, against the planar ansatz.
    rr = radii[-1]
    ansatz = np.tanh((rr - np.hypot(grid.X, grid.Y)) / np.sqrt(2))
    band = np.abs(v) < .9
    profile_error = float(np.sqrt(np.mean((v[band] - ansatz[band]) ** 2)))
    out["metrics"] = {"steps": steps, "dt": dt, "end_time": end, "h": h, "length": length,
                      "physical_profile_scale": float(np.sqrt(2)), "initial_radius": radii[0],
                      "final_radius": radii[-1], "radius_squared_slope": slope,
                      "curvature_slope_relative_error": abs(slope + 2) / 2,
                      "profile_ansatz_rms": profile_error, "max_recorded_energy_increase": energy_increase(energies),
                      "initial_dofs": cells[0], "final_dofs": cells[-1],
                      "mean_dofs": float(np.mean(cells)), "uniform_dofs": n * n,
                      "stored_array_bytes": state_bytes(grid, v), "accepted_transfers": sum(e["changed"] for e in events)}
    out["checks"] = {"energy_decay": energy_increase(energies) <= 1e-9 * max(1, abs(energies[0])),
                     "curvature_slope": abs(slope + 2) / 2 < .1,
                     "finite_values": bool(np.isfinite(v).all())}
    out["transfers"] = events
    out["limitations"] = ["The profile ansatz and curvature law have finite-width errors.",
                          "This adaptation is tensor-grid AMR, not the complete BMVR allocation policy."]
    arrays = {"xe": grid.xe, "ye": grid.ye, "field": v, "time": np.array(times),
              "energy": np.array(energies), "radius": np.array(radii), "dofs": np.array(cells)}
    return out, arrays


def case1(variant):
    return ac_run(adaptive=variant == "adaptive")


def case2(variant):
    out = result(2, "Cahn-Hilliard spinodal decomposition", variant)
    start = perf_counter()
    grid = Grid.uniform(64, 128, periodic=True)
    c = np.random.default_rng(1701).normal(0, .04, grid.shape)
    c -= c.mean()
    initial = c.copy()
    model, lam = DoubleWell(), grid.spectrum()
    times, energies, masses, lengths = [0.], [model.value(grid, c)], [grid.integrate(c)], []
    power0, l0 = structure_factor(grid, c)
    lengths.append(l0)
    out["timings"]["initialization_s"] = perf_counter() - start
    dt, steps = .025, 8000
    for step in range(1, steps + 1):
        start = perf_counter()
        c = ch_step(grid, c, dt, symbol=lam)
        out["timings"]["solver_s"] += perf_counter() - start
        if variant == "adaptive" and step % 20 == 0:
            start = perf_counter()
            # Dense spinodal dynamics: retain the full basis. Audit is charged.
            _ = np.max(np.abs(grid.lap(model.force(grid, c))))
            out["timings"]["adaptation_s"] += perf_counter() - start
        if step % 20 == 0:
            start = perf_counter()
            power, length = structure_factor(grid, c)
            times.append(step * dt)
            energies.append(model.value(grid, c))
            masses.append(grid.integrate(c))
            lengths.append(length)
            out["timings"]["diagnostics_s"] += perf_counter() - start
    fit = np.array(times) >= 80
    exponent = float(np.polyfit(np.log(np.array(times)[fit]), np.log(np.array(lengths)[fit]), 1)[0])
    out["metrics"] = {"steps": steps, "dt": dt, "end_time": steps * dt, "h": .5,
                      "mass_drift": float(np.max(np.abs(np.array(masses) - masses[0]))),
                      "max_recorded_energy_increase": energy_increase(energies),
                      "initial_length": lengths[0], "final_length": lengths[-1],
                      "finite_time_exponent": exponent, "fit_interval": [80, 200],
                      "asymptotic_exponent_verified": False, "final_dofs": grid.size,
                      "stored_array_bytes": state_bytes(grid, c, lam)}
    out["checks"] = {"mass_conservation": out["metrics"]["mass_drift"] < 1e-10,
                     "energy_decay": energy_increase(energies) < 1e-8,
                     "phase_separation": bool(np.std(c) > np.std(initial))}
    out["limitations"] = ["Dense activity forces a full-grid fallback; no DOF saving is expected.",
                          "The finite-time slope is reported, but the asymptotic 1/3 law is inconclusive.",
                          "A separate fixed-width spatial/time study is provided; the long coarsening run is not an asymptotic-exponent certification."]
    return out, {"xe": grid.xe, "ye": grid.ye, "initial": initial, "field": c,
                 "time": np.array(times), "energy": np.array(energies), "mass": np.array(masses),
                 "length": np.array(lengths), "structure_initial": power0, "structure_final": power}


def triple_angles(grid, phases):
    """Find interior three-phase junctions and measure sectors on a finite ring."""
    order = np.sort(phases, axis=0)
    mask = order[-3] > .18
    margin = 6
    mask[:margin] = mask[-margin:] = False
    mask[:, :margin] = mask[:, -margin:] = False
    labels, count = ndimage.label(mask)
    angles = []
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    for label in range(1, count + 1):
        y, x = np.mean(np.argwhere(labels == label), axis=0)
        coords = np.array([y + 3 / grid.dy[0] * np.sin(theta), x + 3 / grid.dx[0] * np.cos(theta)])
        values = np.array([ndimage.map_coordinates(p, coords, order=1, mode="nearest") for p in phases])
        winner = np.argmax(values, axis=0)
        crossings = np.flatnonzero(winner != np.roll(winner, 1))
        if len(crossings) == 3:
            angles.extend((np.diff(np.r_[crossings, crossings[0] + 720]) * .5).tolist())
    return angles


def case3(variant):
    out = result(3, "Symmetric multiphase grain growth", variant)
    start = perf_counter()
    grid = Grid.uniform(32, 64)
    seeds = np.array([[-10., -8.], [0, -10], [10, -7], [-9, 4], [1, 1], [11, 5], [-5, 12], [6, 13],
                      [-2, -1], [4, -1], [1, 4]])
    dist = np.array([(grid.X - x) ** 2 + (grid.Y - y) ** 2 for x, y in seeds])
    scores = -(dist - dist.min(axis=0)) / 4
    phases = np.exp(scores)
    phases /= phases.sum(axis=0)
    initial = phases.copy()
    times, energies, areas = [0.], [grain_energy(grid, phases)], [[grid.integrate(p) for p in phases]]
    out["timings"]["initialization_s"] = perf_counter() - start
    t, steps = 0., 0
    while t < 20 - 1e-12:
        start = perf_counter()
        phases, used = grain_step(grid, phases, min(.025, 20 - t))
        out["timings"]["solver_s"] += perf_counter() - start
        t += used
        steps += 1
        if steps % 40 == 0 or t >= 20 - 1e-12:
            times.append(t)
            energies.append(grain_energy(grid, phases))
            areas.append([grid.integrate(p) for p in phases])
    angles = triple_angles(grid, phases)
    out["metrics"] = {"steps": steps, "end_time": t, "phase_sum_error": float(np.max(np.abs(phases.sum(axis=0) - 1))),
                      "max_recorded_energy_increase": energy_increase(energies),
                      "junction_sector_degrees": angles,
                      "junction_mean_abs_error_degrees": float(np.mean(np.abs(np.array(angles) - 120))) if angles else None,
                      "initial_grains": len(seeds), "remaining_grains": int(np.sum(np.array(areas[-1]) > 1)),
                      "mean_abs_area_change": float(np.mean(np.abs(np.array(areas[-1]) - areas[0]))),
                      "final_dofs": phases.size, "stored_array_bytes": state_bytes(grid, phases)}
    out["checks"] = {"phase_simplex": out["metrics"]["phase_sum_error"] < 1e-12 and bool(phases.min() >= 0),
                     "energy_decay": energy_increase(energies) < 1e-8,
                     "grain_migration": out["metrics"]["mean_abs_area_change"] > .01}
    out["metrics"]["grain_eliminations"] = len(seeds) - out["metrics"]["remaining_grains"]
    out["metrics"]["winning_phase_labels"] = np.unique(np.argmax(phases, axis=0)).tolist()
    out["metrics"]["max_fraction_by_phase"] = phases.max(axis=(1, 2)).tolist()
    out["limitations"] = ["Finite-ring angles are transient estimates, not converged equilibrium validation.",
                          "Statistical grain-growth scaling, mesh convergence and an adaptive comparison remain open."]
    return out, {"xe": grid.xe, "ye": grid.ye, "initial": np.argmax(initial, axis=0),
                 "field": np.argmax(phases, axis=0), "phases": phases,
                 "time": np.array(times), "energy": np.array(energies), "grain_areas": np.array(areas)}


def case4(variant):
    out = result(4, "Coherent precipitate with aligned anisotropy", variant)
    start = perf_counter()
    grid = Grid.uniform(32, 65, periodic=True)
    v, model = circle(grid, 6), DoubleWell(k=(1., 1.4))
    initial = v.copy()
    target = grid.integrate(v)
    elastic = coherent_elasticity(grid, v)
    times, energies, elastic_history = [0.], [model.value(grid, v) + elastic["energy"]], [elastic["energy"]]
    out["timings"]["initialization_s"] = perf_counter() - start
    t, steps, dt = 0., 0, .02
    while t < 8 - 1e-12:
        start = perf_counter()
        force = model.force(grid, v) + elastic["force"]
        force -= grid.integrate(force) / grid.area
        used = min(dt, 8 - t)
        old = model.value(grid, v) + elastic["energy"]
        for _ in range(24):
            trial = v - used * force
            te = coherent_elasticity(grid, trial)
            energy = model.value(grid, trial) + te["energy"]
            if energy <= old + 1e-10:
                break
            used *= .5
        else:
            raise RuntimeError("Precipitate step did not decrease energy.")
        v, elastic = trial, te
        t += used
        steps += 1
        out["timings"]["solver_s"] += perf_counter() - start
        if steps % 20 == 0 or t >= 8 - 1e-12:
            times.append(t)
            energies.append(energy)
            elastic_history.append(elastic["energy"])
    phase_mask = v > 0
    ix = grid.integrate(phase_mask * grid.X ** 2)
    iy = grid.integrate(phase_mask * grid.Y ** 2)
    aspect = float(np.sqrt(iy / ix))
    a, b = circle(grid, 3, center=(-6, 0)), circle(grid, 3, center=(6, 0))
    interaction = coherent_elasticity(grid, a + b + 1)["energy"] - coherent_elasticity(grid, a)["energy"] - coherent_elasticity(grid, b)["energy"]
    final_force = model.force(grid, v) + elastic["force"]
    final_force -= grid.integrate(final_force) / grid.area
    # A separate constrained equilibrium solve; do not assign optimizer iterations a physical time.
    start = perf_counter()
    mean = target / grid.area
    def objective(raw):
        state = raw.reshape(grid.shape)
        state = state - state.mean() + mean
        ee = coherent_elasticity(grid, state)
        ff = model.force(grid, state) + ee["force"]
        grad = (ff - ff.mean()) * grid.vol
        return model.value(grid, state) + ee["energy"], grad.ravel()
    optimum = minimize(objective, v.ravel(), method="L-BFGS-B", jac=True,
                       options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 1000, "maxls": 30})
    equilibrium = optimum.x.reshape(grid.shape)
    equilibrium = equilibrium - equilibrium.mean() + mean
    ee = coherent_elasticity(grid, equilibrium)
    gf = model.force(grid, equilibrium) + ee["force"]
    gf -= gf.mean()
    out["timings"]["equilibrium_optimization_s"] = perf_counter() - start
    equilibrium_mask = equilibrium > 0
    eq_aspect = float(np.sqrt(grid.integrate(equilibrium_mask * grid.Y ** 2) / grid.integrate(equilibrium_mask * grid.X ** 2)))
    out["metrics"] = {"steps": steps, "end_time": t, "volume_constraint_drift": abs(grid.integrate(v) - target),
                      "max_recorded_energy_increase": energy_increase(energies),
                      "elastic_equilibrium_residual": elastic["equilibrium_residual"],
                      "structural_force_rms": float(np.sqrt(np.mean(final_force ** 2))),
                      "transient_y_over_x_aspect": aspect, "two_precipitate_elastic_interaction": interaction,
                      "final_elastic_energy": elastic["energy"], "final_dofs": grid.size,
                      "stored_array_bytes": state_bytes(grid, v, equilibrium, elastic["force"], elastic["sxx"], elastic["syy"], elastic["sxy"]),
                      "equilibrium_force_rms": float(np.sqrt(np.mean(gf ** 2))),
                      "equilibrium_y_over_x_aspect": eq_aspect,
                      "equilibrium_optimizer_success": bool(optimum.success),
                      "equilibrium_optimizer_iterations": int(optimum.nit)}
    out["checks"] = {"elastic_equilibrium": elastic["equilibrium_residual"] < 1e-10,
                     "volume_constraint": out["metrics"]["volume_constraint_drift"] < 1e-9,
                     "energy_decay": energy_increase(energies) < 1e-8,
                     "stationary_shape_residual": out["metrics"]["equilibrium_force_rms"] < 1e-5}
    out["limitations"] = ["A separate constrained equilibrium solve is provided; shape mesh convergence remains open.",
                          "Homogeneous plane-strain elasticity, aligned anisotropy and fixed mean strain only.",
                          "Adaptive elastic discretization and full misfit calibration remain open."]
    return out, {"xe": grid.xe, "ye": grid.ye, "initial": initial, "field": v, "sxx": elastic["sxx"],
                 "syy": elastic["syy"], "exx": elastic["exx"], "eyy": elastic["eyy"], "exy": elastic["exy"],
                 "time": np.array(times), "energy": np.array(energies), "elastic_energy": np.array(elastic_history),
                 "equilibrium_field": equilibrium, "equilibrium_sxx": ee["sxx"]}


def sink_run(n, end=.2, dt=.0005):
    h, D = 1 / n, 1.
    x = (np.arange(n) + .5) * h
    c = np.sin(np.pi * x / 2)
    initial = c.copy()
    diag = -2 * np.ones(n)
    diag[0], diag[-1] = -3, -1
    operator = sparse.diags([np.ones(n-1), diag, np.ones(n-1)], [-1, 0, 1], format="csc") * D / h ** 2
    steps = int(round(end / dt))
    dt = end / steps
    solve = factorized(sparse.eye(n, format="csc") - dt * operator / 2)
    rhs = sparse.eye(n, format="csc") + dt * operator / 2
    captured = 0.
    start = perf_counter()
    for _ in range(steps):
        trial = solve(rhs @ c)
        captured += dt * D * (c[0] + trial[0]) / h
        c = trial
    elapsed = perf_counter() - start
    exact = np.sin(np.pi * x / 2) * np.exp(-np.pi ** 2 * D * end / 4)
    err = float(np.linalg.norm(c - exact) / np.linalg.norm(exact))
    return {"n": n, "h": h, "relative_l2": err, "balance_error": abs(h * c.sum() + captured - h * initial.sum()),
            "captured_inventory": captured, "solver_s": elapsed, "steps": steps}, x, c, exact


def case5(variant):
    out = result(5, "Point-defect diffusion to an absorbing planar sink", variant)
    runs = []
    for n in (64, 128, 256):
        row, x, c, exact = sink_run(n)
        runs.append(row)
    out["timings"]["solver_s"] = sum(r["solver_s"] for r in runs)
    rates = [float(np.log(runs[i]["relative_l2"] / runs[i+1]["relative_l2"]) / np.log(2)) for i in range(2)]
    out["metrics"] = {"spatial_convergence": runs, "observed_orders": rates,
                      "final_relative_l2": runs[-1]["relative_l2"],
                      "mass_plus_sink_balance_error": max(r["balance_error"] for r in runs)}
    out["checks"] = {"analytical_profile": runs[-1]["relative_l2"] < 1e-2,
                     "sink_balance": out["metrics"]["mass_plus_sink_balance_error"] < 1e-10,
                     "spatial_convergence": all(runs[i+1]["relative_l2"] < runs[i]["relative_l2"] for i in range(2))}
    out["limitations"] = ["Planar absorbing-sink reduction; dislocation and pore sink-strength calibration remain open."]
    return out, {"x": x, "field": c, "exact": exact,
                 "resolution": np.array([r["n"] for r in runs]), "error": np.array([r["relative_l2"] for r in runs])}


def case6(variant):
    out = result(6, "Regularized screw dislocation motion and reactions", variant)
    histories = {}
    start = perf_counter()
    for name, signs in (("opposite", [1., -1.]), ("same", [1., 1.])):
        s = Screws(np.array([[-2., 0], [2., 0]]), np.array(signs), core=.75)
        energies, separations, events = [s.pair()[0]], [4.], []
        for step in range(800):
            _, f = s.pair()
            s = s.moved(s.positions + .05 * s.mobility * f)
            s, event = annihilate(s)
            if event["occurred"]:
                events.append({"step": step + 1, **event})
            energies.append(s.pair()[0])
            separations.append(float(np.linalg.norm(s.positions[1] - s.positions[0])) if len(s.burgers) == 2 else 0.)
        histories[name] = {"energy": energies, "separation": separations, "events": events,
                           "burgers": float(s.burgers.sum())}
    out["timings"]["solver_s"] = perf_counter() - start
    r = np.geomspace(.075, 75, 400)
    _, stress = screw_stress(r, np.zeros_like(r))
    classic = 1 / (2 * np.pi * r)
    mask = r >= 7.5
    far_error = float(np.max(np.abs(stress[mask] - classic[mask]) / classic[mask]))
    core, outer = .75, 32.
    exact = (np.log((outer ** 2 + core ** 2) / core ** 2) + core ** 2 / (outer ** 2 + core ** 2) - 1) / (8 * np.pi)
    core_checks = []
    for n in (256, 512, 1024):
        rr = np.linspace(0, outer, n + 1)
        numerical = trapezoid(rr ** 3 / (rr * rr + core * core) ** 2, rr) / (4 * np.pi)
        core_checks.append({"radial_cells": n, "relative_error": abs(numerical - exact) / exact})
    stress_values = np.array([-.04, -.02, .02, .04])
    velocities = []
    for applied in stress_values:
        position = 0.
        for _ in range(200):
            position += .05 * applied
        velocities.append(position / 10.)
    mobility_fit = float(np.polyfit(stress_values, velocities, 1)[0])
    out["metrics"] = {"dt": .05, "end_time": 40., "steps_per_pair": 800,
                      "far_field_relative_error": far_error, "fixed_core_energy_quadrature": core_checks,
                      "same_sign_final_separation": histories["same"]["separation"][-1],
                      "annihilation_events": histories["opposite"]["events"],
                      "max_recorded_energy_increase": max(energy_increase(h["energy"]) for h in histories.values()),
                      "single_line_velocity_stress_slope": mobility_fit, "specified_mobility": 1.}
    out["checks"] = {"far_field_stress": far_error < .01,
                     "same_sign_repulsion": histories["same"]["separation"][-1] > 4,
                     "opposite_sign_annihilation": bool(histories["opposite"]["events"]),
                     "burgers_balance": histories["opposite"]["burgers"] == 0 and histories["same"]["burgers"] == 2,
                     "pair_energy_decay": out["metrics"]["max_recorded_energy_increase"] < 1e-10,
                     "linear_velocity_stress": abs(mobility_fit - 1) < 1e-12,
                     "core_quadrature_convergence": all(core_checks[i+1]["relative_error"] < core_checks[i]["relative_error"] for i in range(2))}
    out["limitations"] = ["Straight screw sections and a softened pair closure; curved-line dynamics are not implemented.",
                          "Isolated stress energy and pair energy have distinct near-core closures.",
                          "The velocity-stress test verifies the specified linear drag law; a material-calibrated curve remains open."]
    return out, {"r": r, "stress": stress, "classical_stress": classic, "time": np.arange(801) * .05,
                 "opposite_separation": np.array(histories["opposite"]["separation"]),
                 "same_separation": np.array(histories["same"]["separation"]),
                 "opposite_energy": np.array(histories["opposite"]["energy"]),
                 "same_energy": np.array(histories["same"]["energy"]),
                 "applied_stress": stress_values, "single_line_velocity": np.array(velocities)}


def case7(variant):
    out = result(7, "Screw interaction with a fixed diffuse interface", variant)
    grid = Grid.uniform(24, 96)
    v, c = np.tanh(grid.X / np.sqrt(2)), np.full(grid.shape, .2)
    rows, arrays = [], {"xe": grid.xe, "ye": grid.ye, "field": v}
    start = perf_counter()
    for tag, stress in (("weak_drive", .005), ("strong_drive", .08)):
        s = Screws(np.array([[-3., 0.]]), np.array([1.]))
        xs, energies = [-3.], []
        for step in range(1601):
            eb, _, _, f = binding(grid, v, c, s, chemical=0., interface=.5)
            energies.append(eb - stress * s.positions[0, 0])
            if step == 1600:
                break
            next_x = s.positions.copy()
            next_x[0, 0] += .05 * (f[0, 0] + stress)
            s = s.moved(next_x)
            xs.append(float(s.positions[0, 0]))
        rows.append({"drive": stress, "final_x": xs[-1], "crossed_midplane": bool(max(xs) > 0),
                     "escaped_binding_region": bool(xs[-1] > 3),
                     "max_energy_increase": energy_increase(energies)})
        arrays[tag + "_x"], arrays[tag + "_energy"] = np.array(xs), np.array(energies)
    out["timings"]["solver_s"] = perf_counter() - start
    arrays["time"] = np.arange(1601) * .05
    out["metrics"] = {"trajectories": rows, "interface_profile_scale": float(np.sqrt(2)),
                      "absorption": None, "crystallographic_transmission": None,
                      "reflection": None, "residual_burgers_vector": None}
    out["checks"] = {"binding_dynamics": all(r["final_x"] > -3 for r in rows),
                     "driven_potential_decay": all(r["max_energy_increase"] < 1e-9 for r in rows)}
    out["limitations"] = ["Binding and glide across a prescribed diffuse profile are demonstrated.",
                          "Midplane crossing is not crystallographic transmission.",
                          "Absorption, reflection, slip-system selection, residual interfacial Burgers content and their barriers are uncalibrated and disabled."]
    return out, arrays


def case8(variant):
    out = result(8, "Vacancy-coupled pore growth and shrinkage", variant)
    grid = Grid.uniform(40, 80)
    arrays, rows = {"xe": grid.xe, "ye": grid.ye}, []
    for tag, supersaturation in (("low", .005), ("high", .15)):
        p = (circle(grid, 6) + 1) / 2
        n = np.full(grid.shape, supersaturation)
        inventory = grid.integrate(n + p)
        times, energies, mass, volumes = [0.], [pore_energy(grid, n, p)], [inventory], [grid.integrate(p)]
        t, steps = 0., 0
        while t < 20 - 1e-12:
            start = perf_counter()
            un, up = pore_forces(grid, n, p)
            rate = 10 * p ** 2 * (1 - p) ** 2 * (up - un)
            dn, dp = grid.lap(un) + rate, -rate
            used = min(.02, 20 - t)
            old = pore_energy(grid, n, p)
            for _ in range(30):
                nn, pp = n + used * dn, p + used * dp
                en = pore_energy(grid, nn, pp)
                if pp.min() >= 0 and pp.max() <= 1 and nn.min() >= 0 and en <= old + 1e-11:
                    break
                used *= .5
            else:
                raise RuntimeError("Pore step could not preserve bounds and energy.")
            n, p = nn, pp
            t += used
            steps += 1
            if steps > 20000:
                raise RuntimeError("Pore integration stalled.")
            out["timings"]["solver_s"] += perf_counter() - start
            if steps % 25 == 0 or t >= 20 - 1e-12:
                times.append(t)
                energies.append(en)
                mass.append(grid.integrate(n + p))
                volumes.append(grid.integrate(p))
        rows.append({"supersaturation": supersaturation, "steps": steps,
                     "initial_pore_area": volumes[0], "final_pore_area": volumes[-1],
                     "area_change": volumes[-1] - volumes[0],
                     "inventory_drift": float(np.max(np.abs(np.array(mass) - inventory))),
                     "max_energy_increase": energy_increase(energies),
                     "minimum_inventory_density": float(n.min())})
        for key, value in (("p", p), ("n", n), ("time", times), ("energy", energies), ("mass", mass), ("area", volumes)):
            arrays[tag + "_" + key] = np.asarray(value)
    out["metrics"] = {"runs": rows, "end_time": 20., "physical_profile_scale": float(np.sqrt(2))}
    out["checks"] = {"inventory_conservation": all(r["inventory_drift"] < 1e-8 for r in rows),
                     "energy_decay": all(r["max_energy_increase"] < 1e-8 for r in rows),
                     "low_supersaturation_shrinks": rows[0]["area_change"] < 0,
                     "high_supersaturation_grows": rows[1]["area_change"] > 0}
    out["limitations"] = ["Two-dimensional closed-inventory analogue; no gas, elasticity, nucleation or fracture.",
                          "Quantitative Gibbs-Thomson and growth-law convergence, and an adaptive comparison remain open."]
    return out, arrays


def case9(variant):
    out = result(9, "Coupled defect population, screw dipole and diffuse inclusion", variant)
    start = perf_counter()
    length, h = 64., .5
    grid = focused_grid(length, h, ((-24, 24), (-24, 24))) if variant == "adaptive" else Grid.uniform(length, 128)
    v, c = circle(grid, 8), np.full(grid.shape, .2)
    screws = Screws(np.array([[-10., 0], [10., 0]]), np.array([1., -1.]), core=1.)
    origins = screws.positions.copy()
    inventory = grid.integrate(c)
    params = {"imposed": .15, "chemical": .3, "interface": .2}
    current = coupled_energy_forces(grid, v, c, screws, origins, **params)
    times, energies, masses = [0.], [current["energy"]], [inventory]
    positions, plastic, stress, dofs, events = [origins.copy()], [0.], [current["mean_stress"]], [2 * grid.size + 2], []
    out["timings"]["initialization_s"] = perf_counter() - start
    t, steps = 0., 0
    while t < 8 - 1e-12:
        start = perf_counter()
        dv, dc = -current["gv"], grid.lap(current["mu"])
        velocity = screws.mobility * current["force"]
        used = min(.02, 8 - t)
        for _ in range(24):
            nv, nc, ns = v + used * dv, c + used * dc, screws.moved(screws.positions + used * velocity)
            candidate = coupled_energy_forces(grid, nv, nc, ns, origins, **params)
            if nc.min() > 0 and candidate["energy"] <= current["energy"] + 1e-10:
                break
            used *= .5
        else:
            raise RuntimeError("Coupled step failed its energy/bounds check.")
        v, c, screws, current = nv, nc, ns, candidate
        t += used
        steps += 1
        out["timings"]["solver_s"] += perf_counter() - start
        if variant == "adaptive" and steps % 100 == 0:
            start = perf_counter()
            energy = lambda g, f: coupled_energy_forces(g, f[0], f[1], screws, origins, **params)["energy"]
            grid, fields, record = adapt(grid, [v, c], h, length, energy, points=screws.positions)
            v, c = fields
            events.append(record)
            current = coupled_energy_forces(grid, v, c, screws, origins, **params)
            out["timings"]["adaptation_s"] += perf_counter() - start
        if steps % 20 == 0 or t >= 8 - 1e-12:
            times.append(t)
            energies.append(current["energy"])
            masses.append(grid.integrate(c))
            positions.append(screws.positions.copy())
            plastic.append(current["plastic_shear"])
            stress.append(current["mean_stress"])
            dofs.append(2 * grid.size + 2)
    out["metrics"] = {"end_time": t, "steps": steps, "h": h, "physical_profile_scale": float(np.sqrt(2)),
                      "inventory_drift": float(np.max(np.abs(np.array(masses) - inventory))),
                      "burgers_content": float(screws.burgers.sum()), "final_plastic_shear": plastic[-1],
                      "initial_stress": stress[0], "final_stress": stress[-1],
                      "dislocation_displacement": float(np.linalg.norm(screws.positions - origins)),
                      "concentration_range": float(np.ptp(c)),
                      "max_recorded_energy_increase": energy_increase(energies),
                      "initial_dofs": dofs[0], "final_dofs": dofs[-1],
                      "stored_array_bytes": state_bytes(grid, v, c, screws.positions, screws.burgers),
                      "accepted_transfers": sum(e["changed"] for e in events)}
    out["checks"] = {"mass_conservation": out["metrics"]["inventory_drift"] < 1e-8,
                     "burgers_conservation": screws.burgers.sum() == 0,
                     "energy_decay": energy_increase(energies) < 1e-7,
                     "reciprocal_coupling_evolves": out["metrics"]["dislocation_displacement"] > .01 and np.ptp(c) > 1e-5,
                     "plastic_relaxation": abs(stress[-1]) < abs(stress[0])}
    out["transfers"] = events
    out["limitations"] = ["0D defects are populations, lines are straight sections, and plasticity is one analytical mean-shear mode.",
                          "No local 3D crystal plasticity, curved-line topology, disclinations or crystallographic transmission.",
                          "The reference uses the same sparse lines with uniform fields; it is a uniform-field hybrid reference, not an all-field dislocation PF model.",
                          "The full residual-per-cost allocation policy is not implemented."]
    return out, {"xe": grid.xe, "ye": grid.ye, "field": v, "concentration": c,
                 "time": np.array(times), "energy": np.array(energies), "mass": np.array(masses),
                 "positions": np.array(positions), "plastic_shear": np.array(plastic),
                 "stress": np.array(stress), "dofs": np.array(dofs)}


CASES = {1: case1, 2: case2, 3: case3, 4: case4, 5: case5, 6: case6, 7: case7, 8: case8, 9: case9}
