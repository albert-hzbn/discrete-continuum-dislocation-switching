"""Section 60: a Frank-Read source in the line-tension approximation, with loop emission by reconnection."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


def signed_area(nodes):
    """Area swept in the positive (tau-driven) sense: minus the shoelace sum of the closed polygon."""
    x, y = nodes[:, 0], nodes[:, 1]
    return -.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def area_gradient(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    return -.5 * np.stack((np.roll(y, -1) - np.roll(y, 1), np.roll(x, 1) - np.roll(x, -1)), 1)


def segment_distance(p0, p1, q0, q1):
    """Minimum distance between two segments in the plane (sampled endpoint projections)."""
    def point_segment(p, a, b):
        d = b - a
        s = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-300), 0, 1)
        return np.linalg.norm(p - a - s * d)
    return min(point_segment(p0, q0, q1), point_segment(p1, q0, q1), point_segment(q0, p0, p1), point_segment(q1, p0, p1))


@dataclass
class FrankRead:
    length: float = 2.0
    tension: float = 1.0
    burgers: float = 1.0
    mobility: float = 1.0
    stress: float = 1.2
    lmin: float = .03
    lmax: float = .1
    capture: float = .05

    @property
    def critical_stress(self):
        return 2 * self.tension / (self.burgers * self.length)

    def initial(self, bow=1e-3):
        n = int(np.ceil(self.length / self.lmax)) + 1
        x = np.linspace(-self.length / 2, self.length / 2, n)
        return dict(line=np.stack((x, bow * (1 - (2 * x / self.length) ** 2)), 1), loops=[])

    @staticmethod
    def lengths(nodes, closed):
        seg = (np.roll(nodes, -1, axis=0) - nodes) if closed else np.diff(nodes, axis=0)
        return np.linalg.norm(seg, axis=1)

    def energy(self, state):
        total = self.tension * self.lengths(state["line"], False).sum() - self.stress * self.burgers * signed_area(state["line"])
        for loop in state["loops"]:
            total += self.tension * self.lengths(loop, True).sum() - self.stress * self.burgers * signed_area(loop)
        return float(total)

    def _rates(self, nodes, closed):
        seg = (np.roll(nodes, -1, axis=0) - nodes) if closed else np.diff(nodes, axis=0)
        ell = np.linalg.norm(seg, axis=1)
        t = seg / ell[:, None]
        force = self.stress * self.burgers * area_gradient(nodes)
        if closed:
            force += self.tension * (t - np.roll(t, 1, axis=0))
            measure = (ell + np.roll(ell, 1)) / 2
        else:
            force[1:-1] += self.tension * (t[1:] - t[:-1])
            measure = np.ones(len(nodes))
            measure[1:-1] = (ell[1:] + ell[:-1]) / 2
        rates = self.mobility * force / measure[:, None]
        if not closed:
            rates[[0, -1]] = 0.0  # pinned ends
        return rates, float(np.sum(force * rates))

    def step(self, state, dt, energy=None):
        energy = self.energy(state) if energy is None else energy
        pieces = [self._rates(state["line"], False)] + [self._rates(loop, True) for loop in state["loops"]]
        dissipation = sum(d for _, d in pieces)
        used = dt
        for _ in range(30):
            trial = dict(line=state["line"] + used * pieces[0][0], loops=[l + used * r for l, (r, _) in zip(state["loops"], pieces[1:])])
            value = self.energy(trial)
            if value <= energy - .1 * used * dissipation + 1e-13 * max(1.0, abs(energy)):
                return trial, value, used
            used *= .5
        raise RuntimeError("Line step failed its energy check.")

    def _remesh_curve(self, nodes, closed):
        """Insert midpoints (energy neutral); remove short-segment nodes only when that does not raise F."""
        ell = self.lengths(nodes, closed)
        if ell.max() <= self.lmax and ell.min() >= self.lmin:
            return nodes, 0.0
        out = []
        count = len(nodes)
        for i in range(count):
            out.append(nodes[i])
            if i < len(ell) and ell[i] > self.lmax:
                k = int(np.ceil(ell[i] / self.lmax))
                nxt = nodes[(i + 1) % count]
                out.extend(nodes[i] + (nxt - nodes[i]) * np.arange(1, k)[:, None] / k)
        nodes = np.array(out)
        removed = 0.0
        i = 1
        while i < len(nodes) - (0 if closed else 1) and len(nodes) > (4 if closed else 3):
            prev, nxt = nodes[i - 1], nodes[(i + 1) % len(nodes)]
            if min(np.linalg.norm(nodes[i] - prev), np.linalg.norm(nxt - nodes[i])) < self.lmin:
                old = self.tension * (np.linalg.norm(nodes[i] - prev) + np.linalg.norm(nxt - nodes[i]))
                new = self.tension * np.linalg.norm(nxt - prev)
                tri = .5 * ((nodes[i][0] - prev[0]) * (nxt[1] - prev[1]) - (nxt[0] - prev[0]) * (nodes[i][1] - prev[1]))
                delta = new - old - self.stress * self.burgers * tri  # removing node i removes -tri from the swept area
                if delta <= 0:
                    nodes = np.delete(nodes, i, axis=0)
                    removed += delta
                    continue
            i += 1
        return nodes, removed

    def remesh(self, state):
        line, d0 = self._remesh_curve(state["line"], False)
        loops, removed = [], d0
        for loop in state["loops"]:
            l, d = self._remesh_curve(loop, True)
            loops.append(l)
            removed += d
        return dict(line=line, loops=loops), removed

    def reconnect(self, state, gap=5, require_descent=False):
        """Swap connections of the closest antiparallel non-adjacent segment pair of the line within the capture distance.

        require_descent (post hoc, Section 60 outcome): defer the event while it would raise F.
        """
        line = state["line"]
        if len(line) < gap + 3:
            return None
        seg = np.diff(line, axis=0)
        t = seg / np.linalg.norm(seg, axis=1)[:, None]
        mid = (line[:-1] + line[1:]) / 2
        best = None
        for p, q in cKDTree(mid).query_pairs(self.capture + self.lmax):
            p, q = min(p, q), max(p, q)
            if q - p < gap or np.dot(t[p], t[q]) >= -.5:
                continue
            d = segment_distance(line[p], line[p + 1], line[q], line[q + 1])
            if d < self.capture and (best is None or d < best[0]):
                best = (d, p, q)
        if best is None:
            return None
        _, p, q = best
        new_line = np.concatenate([line[:p + 1], line[q + 1:]])
        loop = line[p + 1:q + 1].copy()
        new = dict(line=new_line, loops=state["loops"] + [loop])
        if require_descent and self.energy(new) > self.energy(state):
            return None
        return new, dict(p=int(p), q=int(q), distance=float(best[0]))
