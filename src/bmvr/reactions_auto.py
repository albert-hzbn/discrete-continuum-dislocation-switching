"""Section 78: automatic detection, Frank classification and execution of line-line junction reactions on graphs."""
import numpy as np
from .graphs3d import DislocationGraph


def segment_closest(p0, p1, q0, q1):
    """Closest points of segments [p0,p1] and [q0,q1]: returns (s, t, distance) with parameters in [0, 1]."""
    d1, d2, r = p1-p0, q1-q0, p0-q0
    a, e, f = d1@d1, d2@d2, d2@r
    if a <= 1e-300 and e <= 1e-300:return 0., 0., float(np.linalg.norm(r))
    if a <= 1e-300:s, t = 0., np.clip(f/e, 0, 1)
    else:
        c = d1@r
        if e <= 1e-300:t, s = 0., np.clip(-c/a, 0, 1)
        else:
            b = d1@d2; den = a*e-b*b
            s = np.clip((b*f-c*e)/den, 0, 1) if den > 1e-300 else 0.
            t = (b*s+f)/e
            if t < 0:t, s = 0., np.clip(-c/a, 0, 1)
            elif t > 1:t, s = 1., np.clip((b-c)/a, 0, 1)
    return float(s), float(t), float(np.linalg.norm(p0+s*d1-(q0+t*d2)))


def line_labels(graph):
    """Connected-component label of every edge (union-find over nodes)."""
    parent = list(range(len(graph.nodes)))
    def find(i):
        while parent[i] != i:parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in graph.edges:parent[find(i)] = find(j)
    return np.array([find(i) for i, _ in graph.edges])


def detect(graph, capture):
    """Pairs of edges on different lines whose closest approach is below `capture`."""
    labels = line_labels(graph); out = []
    for e1 in range(len(graph.edges)):
        for e2 in range(e1+1, len(graph.edges)):
            if labels[e1] == labels[e2]:continue
            (i, j), (k, l) = graph.edges[e1], graph.edges[e2]
            s, t, d = segment_closest(graph.nodes[i], graph.nodes[j], graph.nodes[k], graph.nodes[l])
            if d < capture:out.append(dict(edges=(e1, e2), s=s, t=t, distance=d))
    return out


def crossing_nodes(graph, candidates, tol=1e-12):
    """For candidates whose closest points coincide with nodes, return the node of each line (one pair)."""
    for c in candidates:
        (i, j), (k, l) = graph.edges[c['edges'][0]], graph.edges[c['edges'][1]]
        a = i if c['s'] < tol else j if c['s'] > 1-tol else None
        b = k if c['t'] < tol else l if c['t'] > 1-tol else None
        if a is not None and b is not None and np.linalg.norm(graph.nodes[a]-graph.nodes[b]) <= max(c['distance'], tol)+tol:
            return a, b
    return None


def classify(graph, a, b, normal_a, normal_b):
    """Frank classification of a crossing of lines through nodes a and b (glide-plane normals given)."""
    t = np.cross(normal_a, normal_b); t /= np.linalg.norm(t)
    def arms(node):
        ein = int(np.flatnonzero(graph.edges[:, 1] == node)[0]); eout = int(np.flatnonzero(graph.edges[:, 0] == node)[0])
        return ein, eout
    (ain, aout), (bin_, bout) = arms(a), arms(b)
    if (graph.nodes[graph.edges[aout, 1]]-graph.nodes[graph.edges[ain, 0]])@t < 0:t = -t  # line a runs along +t
    sb = 1. if (graph.nodes[graph.edges[bout, 1]]-graph.nodes[graph.edges[bin_, 0]])@t > 0 else -1.
    b1, b2 = graph.burgers[ain], graph.burgers[bin_]; bj = b1+sb*b2
    gain = b1@b1+b2@b2-bj@bj
    return dict(direction=t, sense_b=sb, junction_burgers=bj, frank_gain=float(gain), favorable=bool(gain > 0),
                arms=dict(a=(ain, aout), b=(bin_, bout)))


def form_junction(graph, a, b, info, length):
    """Replace crossing nodes a (line A) and b (line B) by a junction J1 -> J2 of the given length along t."""
    t = info['direction']; X = graph.nodes[a]; J1, J2 = X-length/2*t, X+length/2*t
    keep = [n for n in range(len(graph.nodes)) if n not in (a, b)]; remap = {n: i for i, n in enumerate(keep)}
    j1, j2 = len(keep), len(keep)+1
    nodes = np.vstack([graph.nodes[keep], J1, J2])
    projectors = np.concatenate([graph.projectors[keep], np.outer(t, t)[None], np.outer(t, t)[None]])
    (ain, aout), (bin_, bout) = info['arms']['a'], info['arms']['b']
    edges, burgers = [], []
    for e, (i, j) in enumerate(graph.edges):
        if e in (ain, aout, bin_, bout):continue
        edges.append((remap[i], remap[j])); burgers.append(graph.burgers[e])
    b1, b2 = graph.burgers[ain], graph.burgers[bin_]
    edges += [(remap[graph.edges[ain, 0]], j1), (j2, remap[graph.edges[aout, 1]])]; burgers += [b1, b1]
    if info['sense_b'] > 0:edges += [(remap[graph.edges[bin_, 0]], j1), (j2, remap[graph.edges[bout, 1]])]
    else:edges += [(remap[graph.edges[bin_, 0]], j2), (j1, remap[graph.edges[bout, 1]])]
    burgers += [b2, b2]
    edges.append((j1, j2)); burgers.append(info['junction_burgers'])
    return DislocationGraph(nodes, np.array(edges), np.array(burgers), core=graph.core, projectors=projectors, mobility=graph.mobility)
