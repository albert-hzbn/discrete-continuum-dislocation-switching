import numpy as np
from bmvr.frank_read import FrankRead, signed_area


def test_nodal_rates_are_exact_energy_gradients_for_line_and_loop():
    fr = FrankRead(stress=.8)
    line = fr.initial(bow=.1)["line"]
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    loop = np.stack((2 * np.cos(theta), -2 * np.sin(theta)), 1)
    for nodes, closed, i in ((line, False, 5), (loop, True, 3)):
        rates, _ = fr._rates(nodes, closed)
        ell = fr.lengths(nodes, closed)
        measure = (ell[i] + ell[i - 1]) / 2
        for axis in (0, 1):
            plus, minus = nodes.copy(), nodes.copy()
            plus[i, axis] += 1e-7
            minus[i, axis] -= 1e-7
            state = lambda n: dict(line=n, loops=[]) if not closed else dict(line=line, loops=[n])
            fd = -(fr.energy(state(plus)) - fr.energy(state(minus))) / 2e-7
            assert np.isclose(fd, rates[i, axis] * measure, rtol=1e-6, atol=1e-10)


def test_remeshing_never_raises_energy_and_insertion_is_neutral():
    fr = FrankRead(stress=1.2)
    x = np.linspace(-1, 1, 6)
    state = dict(line=np.stack((x, .3 * (1 - x * x)), 1), loops=[])
    meshed, removed = fr.remesh(state)
    assert removed == 0 and len(meshed["line"]) > 6
    assert abs(fr.energy(meshed) - fr.energy(state)) < 1e-13
    crowded = np.stack((np.linspace(-1, 1, 101), .3 * (1 - np.linspace(-1, 1, 101) ** 2)), 1)
    before = fr.energy(dict(line=crowded, loops=[]))
    meshed, removed = fr.remesh(dict(line=crowded, loops=[]))
    assert removed <= 0 and abs(fr.energy(meshed) - (before + removed)) < 1e-12


def polyline(corners, spacing=.05):
    points = [np.array(corners[0], float)]
    for a, b in zip(corners[:-1], corners[1:]):
        a, b = np.array(a, float), np.array(b, float)
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        points.extend(a + (b - a) * np.arange(1, k + 1)[:, None] / k)
    return np.array(points)


def test_reconnection_emits_a_closed_loop_and_repins_the_source():
    fr = FrankRead(stress=1.2)
    # A wrapped source: the A-side arm runs right along y=-1.02, the B-side arm runs left along y=-0.98 (antiparallel).
    line = polyline([(-1, 0), (-1, -1.02), (.5, -1.02), (.5, -2), (-2, -2), (-2, 2), (2, 2), (2, -.98), (-.5, -.98), (-.5, -.5), (1, -.5), (1, 0)])
    state = dict(line=line, loops=[])
    new, info = fr.reconnect(state)
    assert np.array_equal(new["line"][0], line[0]) and np.array_equal(new["line"][-1], line[-1])
    assert len(new["loops"]) == 1 and len(new["line"]) + len(new["loops"][0]) == len(line)
    assert info["distance"] < fr.capture
    assert signed_area(new["loops"][0]) > 0
    # Long antiparallel overlap: the strip between the arms is swept area that the swap loses, so the declared rule
    # can raise F (the Section 60 F3 failure); the post hoc rule defers exactly such events.
    assert fr.energy(new) > fr.energy(state)
    assert fr.reconnect(state, require_descent=True) is None
    # Parallel arms (same direction) are never reconnected.
    parallel = polyline([(-1, 0), (-1, -1.02), (.5, -1.02), (.5, -2), (-2, -2), (-2, -.98), (1, -.98), (1, 0)])
    assert fr.reconnect(dict(line=parallel, loops=[])) is None
