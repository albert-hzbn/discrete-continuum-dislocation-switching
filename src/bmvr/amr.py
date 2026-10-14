"""Section 61.1: an independent block-structured quadtree AMR baseline for the case 1 Allen-Cahn problem.

Leaf blocks of B x B cells on levels 0..L (h_l = h_0 / 2^l) are stacked into one array with a one-cell ghost layer.
Ghost values come from a precomputed sparse gather: a same-level neighbour (copy), a coarser neighbour (covering
cell, piecewise constant), a finer neighbour (mean of the two adjacent fine cells) or the domain boundary (mirror,
no flux). The explicit AC update is vectorized over all blocks. Regridding refines blocks that contain the diffuse
interface, adds a one-block buffer, enforces 2:1 face balance, and transfers the field conservatively through the
finest-level projection.
"""

import numpy as np
from scipy import sparse
from scipy.ndimage import maximum_filter


class QuadtreeAMR:
    def __init__(self, length=128.0, base_blocks=2, block=16, levels=3, threshold=.995, W=1.0, kappa=1.0):
        self.length, self.nb0, self.B, self.L, self.threshold, self.W, self.kappa = length, base_blocks, block, levels, threshold, W, kappa
        self.nf = base_blocks * 2 ** levels            # finest blocks per side
        self.n = self.nf * block                        # finest cells per side
        self.h_fine = length / self.n

    # ---- geometry -------------------------------------------------------------------------------------------
    def h(self, level):
        return self.h_fine * 2 ** (self.L - level)

    def span(self, level):
        return 2 ** (self.L - level)                    # finest blocks covered per side

    def finest_centers(self):
        x = (np.arange(self.n) + .5) * self.h_fine - self.length / 2
        return np.meshgrid(x, x)

    # ---- tree construction ----------------------------------------------------------------------------------
    def build(self, mask):
        """Leaves refining every block whose area meets `mask` (finest-block booleans), then 2:1 balanced."""
        leaves = []

        def visit(level, i, j):
            s = self.span(level)
            if level < self.L and mask[j * s:(j + 1) * s, i * s:(i + 1) * s].any():
                for di in (0, 1):
                    for dj in (0, 1):
                        visit(level + 1, 2 * i + di, 2 * j + dj)
            else:
                leaves.append((level, i, j))
        for i in range(self.nb0):
            for j in range(self.nb0):
                visit(0, i, j)
        while True:
            levelmap = self.level_map(leaves)
            refine = []
            for leaf in leaves:
                level, i, j = leaf
                s = self.span(level)
                i0, j0 = i * s, j * s
                neighbours = []
                if i0 > 0: neighbours.append(levelmap[j0:j0 + s, i0 - 1])
                if i0 + s < self.nf: neighbours.append(levelmap[j0:j0 + s, i0 + s])
                if j0 > 0: neighbours.append(levelmap[j0 - 1, i0:i0 + s])
                if j0 + s < self.nf: neighbours.append(levelmap[j0 + s, i0:i0 + s])
                if any(n.max() > level + 1 for n in neighbours):
                    refine.append(leaf)
            if not refine:
                break
            refined = set(refine)
            leaves = [l for l in leaves if l not in refined] + [(l + 1, 2 * i + di, 2 * j + dj) for (l, i, j) in refine for di in (0, 1) for dj in (0, 1)]
        self.leaves = sorted(leaves)
        self.index = {leaf: k for k, leaf in enumerate(self.leaves)}
        self.levels = np.array([l for l, _, _ in self.leaves])
        self._ghost_operator()
        return self.leaves

    def level_map(self, leaves):
        m = np.zeros((self.nf, self.nf), dtype=int)
        for level, i, j in leaves:
            s = self.span(level)
            m[j * s:(j + 1) * s, i * s:(i + 1) * s] = level
        return m

    def _leaf_at(self, level_map_leafs, fi, fj):
        return level_map_leafs[fj, fi]

    def _ghost_operator(self):
        """Sparse G with ghost = G @ interior.ravel(), interior shape (nblocks, B, B)."""
        B = self.B
        owner = np.full((self.nf, self.nf), -1)
        for k, (level, i, j) in enumerate(self.leaves):
            s = self.span(level)
            owner[j * s:(j + 1) * s, i * s:(i + 1) * s] = k
        rows, cols, vals = [], [], []
        nb = len(self.leaves)
        # ghost layout: (nblocks, 4 sides, B): side 0 = west, 1 = east, 2 = south, 3 = north
        def gid(k, side, m): return (k * 4 + side) * B + m
        def cid(k, r, c): return (k * B + r) * B + c
        for k, (level, i, j) in enumerate(self.leaves):
            s = self.span(level)
            for side in range(4):
                for m in range(B):
                    # finest-resolution coordinates (in finest cells) of the ghost cell centre
                    scale = 2 ** (self.L - level)
                    if side in (0, 1):
                        r, c = m, (-1 if side == 0 else B)
                    else:
                        r, c = (-1 if side == 2 else B), m
                    gx = (i * B + c + .5) * scale   # in finest-cell units
                    gy = (j * B + r + .5) * scale
                    if gx < 0 or gx >= self.n or gy < 0 or gy >= self.n:
                        # domain boundary: mirror the adjacent own cell (no flux)
                        rr, cc = min(max(r, 0), B - 1), min(max(c, 0), B - 1)
                        rows.append(gid(k, side, m)); cols.append(cid(k, rr, cc)); vals.append(1.0)
                        continue
                    fb_i, fb_j = int(gx // B), int(gy // B)
                    q = owner[fb_j, fb_i]
                    ql, qi, qj = self.leaves[q]
                    qscale = 2 ** (self.L - ql)
                    if ql <= level:
                        # same or coarser: the neighbour cell containing the ghost centre
                        cc, rr = int(gx // qscale) - qi * B, int(gy // qscale) - qj * B
                        rows.append(gid(k, side, m)); cols.append(cid(q, rr, cc)); vals.append(1.0)
                    else:
                        # finer (one level): the two fine cells adjacent to the shared face, averaged
                        half = scale / 2
                        if side in (0, 1):
                            fx = gx + (half / 2 if side == 0 else -half / 2)
                            pts = [(fx, gy - half / 2), (fx, gy + half / 2)]
                        else:
                            fy = gy + (half / 2 if side == 2 else -half / 2)
                            pts = [(gx - half / 2, fy), (gx + half / 2, fy)]
                        for px, py in pts:
                            q2 = owner[int(py // B), int(px // B)]
                            l2, i2, j2 = self.leaves[q2]
                            s2 = 2 ** (self.L - l2)
                            rows.append(gid(k, side, m)); cols.append(cid(q2, int(py // s2) - j2 * B, int(px // s2) - i2 * B)); vals.append(.5)
        self.G = sparse.csr_matrix((vals, (rows, cols)), shape=(nb * 4 * B, nb * B * B))
        self.hblock = np.array([self.h(l) for l in self.levels])

    # ---- data transfer --------------------------------------------------------------------------------------
    def restrict(self, fine):
        """Leaf data from a finest-level field by conservative block averaging."""
        B = self.B
        data = np.empty((len(self.leaves), B, B))
        for k, (level, i, j) in enumerate(self.leaves):
            s = 2 ** (self.L - level)
            patch = fine[j * B * s:(j + 1) * B * s, i * B * s:(i + 1) * B * s]
            data[k] = patch.reshape(B, s, B, s).mean(axis=(1, 3))
        return data

    def prolong(self, data):
        """Finest-level piecewise-constant projection of leaf data."""
        B = self.B
        fine = np.empty((self.n, self.n))
        for k, (level, i, j) in enumerate(self.leaves):
            s = 2 ** (self.L - level)
            fine[j * B * s:(j + 1) * B * s, i * B * s:(i + 1) * B * s] = np.kron(data[k], np.ones((s, s)))
        return fine

    def regrid(self, data=None, fine=None):
        fine = self.prolong(data) if fine is None else fine
        active = (np.abs(fine) < self.threshold).reshape(self.nf, self.B, self.nf, self.B).any(axis=(1, 3))
        self.build(maximum_filter(active, size=3, mode="constant"))
        return self.restrict(fine)

    # ---- evolution ------------------------------------------------------------------------------------------
    def step(self, data, dt):
        B = self.B
        ghost = (self.G @ data.ravel()).reshape(len(self.leaves), 4, B)
        west, east = np.concatenate([ghost[:, 0, :, None], data[:, :, :-1]], 2), np.concatenate([data[:, :, 1:], ghost[:, 1, :, None]], 2)
        south, north = np.concatenate([ghost[:, 2, None, :], data[:, :-1, :]], 1), np.concatenate([data[:, 1:, :], ghost[:, 3, None, :]], 1)
        lap = (west + east + south + north - 4 * data) / self.hblock[:, None, None] ** 2
        return data - dt * (self.W * (data ** 3 - data) - self.kappa * lap)

    def cells(self):
        return len(self.leaves) * self.B * self.B
