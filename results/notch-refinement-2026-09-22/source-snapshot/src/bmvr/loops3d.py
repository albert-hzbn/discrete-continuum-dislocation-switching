"""Conserved closed-loop sources and energy-consistent regularized PK forces.

Gaussian cores, homogeneous periodic mechanics, contractible polygons only.
Direct Fourier sampling is intended for verification rather than large networks.
"""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .continuum3d import Cube, CoupledCrystal, Parameters3D, PeriodicElasticity


@dataclass
class Loop:
    nodes: np.ndarray
    burgers: np.ndarray
    core: float = .8
    core_tension: float = .01
    mobility: float = 1.
    glide_normal: np.ndarray | None = None

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.burgers = np.asarray(self.burgers, dtype=float)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3 or len(self.nodes) < 3:
            raise ValueError('A closed polygon requires at least three 3D nodes, without a repeated endpoint.')
        if self.burgers.shape != (3,) or np.linalg.norm(self.burgers) == 0:
            raise ValueError('The Burgers vector must be a nonzero 3D vector.')
        if not np.isfinite(self.nodes).all() or not np.isfinite(self.burgers).all():
            raise ValueError('Loop geometry and charge must be finite.')
        if min(self.core, self.mobility) <= 0 or self.core_tension < 0 or self.lengths.min() <= 1e-12:
            raise ValueError('Positive core, mobility and segment lengths, and nonnegative core tension, are required.')
        if self.glide_normal is not None:
            self.glide_normal = np.asarray(self.glide_normal, dtype=float)
            if self.glide_normal.shape != (3,) or not np.isclose(np.linalg.norm(self.glide_normal), 1):
                raise ValueError('The glide normal must be a unit 3D vector.')
            if abs(self.burgers @ self.glide_normal) > 1e-12:
                raise ValueError('Burgers vector must lie in the fixed glide plane.')
            if np.ptp(self.nodes @ self.glide_normal) > 1e-10:
                raise ValueError('All nodes must lie in the declared glide plane.')

    @property
    def segments(self):
        return np.roll(self.nodes, -1, axis=0) - self.nodes

    @property
    def lengths(self):
        return np.linalg.norm(self.segments, axis=1)

    @property
    def area(self):
        return .5 * np.sum(np.cross(self.nodes, np.roll(self.nodes, -1, axis=0)), axis=0)

    @property
    def core_energy(self):
        return float(self.core_tension * self.lengths.sum())

    def moved(self, nodes):
        return replace(self, nodes=np.array(nodes, copy=True))

    def subdivide(self, max_length):
        if max_length <= 0:
            raise ValueError('Maximum segment length must be positive.')
        points = []
        for node, segment, length in zip(self.nodes, self.segments, self.lengths):
            count = int(np.ceil(length / max_length))
            points.extend(node + np.arange(count)[:, None] / count * segment)
        return self.moved(np.array(points))

    def to_dict(self):
        return {'nodes': self.nodes.tolist(), 'burgers': self.burgers.tolist(), 'core': self.core,
                'core_tension': self.core_tension, 'mobility': self.mobility,
                'glide_normal': None if self.glide_normal is None else self.glide_normal.tolist()}

    def coarsen_collinear(self):
        """Remove only straight-segment subdivision nodes within roundoff tolerance."""
        incoming = self.nodes - np.roll(self.nodes, 1, axis=0)
        outgoing = self.segments
        product = np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
        straight = (np.linalg.norm(np.cross(incoming, outgoing), axis=1) <= 1e-12 * product)
        straight &= np.sum(incoming * outgoing, axis=1) > 0
        if np.count_nonzero(~straight) < 3:
            raise ValueError('Collinear removal would produce an invalid closed polygon.')
        return self.moved(self.nodes[~straight])


def circular_loop(radius=3., count=12, center=(0., 0., 0.), burgers=(.2, 0., 0.), **kwargs):
    theta = np.arange(count) * (2 * np.pi / count)
    nodes = np.array([radius*np.cos(theta), radius*np.sin(theta), np.zeros(count)]).T + center
    return Loop(nodes, np.array(burgers), **kwargs)


class LoopMechanics:
    def __init__(self, grid, lam=2., shear=1., quadrature_order=8):
        if quadrature_order < 1:
            raise ValueError('Segment quadrature order must be positive.')
        self.grid = grid
        self.elastic = PeriodicElasticity(grid, lam, shear)
        self.k = self.elastic.k
        self.k2 = np.sum(self.k ** 2, axis=0)
        self.origin = np.full(3, -grid.length/2 + grid.h/2)
        abscissa, weight = np.polynomial.legendre.leggauss(quadrature_order)
        self.quadrature_order = quadrature_order
        self.s, self.weight = (abscissa + 1)/2, weight/2

    def source(self, loops):
        started = perf_counter()
        shape = (3, 3) + self.grid.shape
        alpha, beta = np.zeros(shape, complex), np.zeros(shape, complex)
        for loop in loops:
            line = np.zeros((3,) + self.grid.shape, complex)
            for point, segment in zip(loop.nodes, loop.segments):
                midpoint = point + segment/2 - self.origin
                phase = np.exp(-1j * np.einsum('i...,i->...', self.k, midpoint))
                integral = np.sinc(np.einsum('i...,i->...', self.k, segment)/(2*np.pi))
                line += segment[:, None, None, None] * phase * integral
            a = loop.burgers[:, None, None, None, None] * line[None] * np.exp(-loop.core**2*self.k2/2) / self.grid.h**3
            alpha += a
            for i in range(3):
                beta[i] += -1j * np.cross(self.k, a[i], axisa=0, axisb=0, axisc=0) * self.elastic.inverse_k2
            beta[:, :, 0, 0, 0] -= np.outer(loop.burgers, loop.area) / self.grid.h**3
        distortion = np.fft.ifftn(beta, axes=(-3, -2, -1)).real
        div = 1j * np.einsum('j...,ij...->i...', self.k, alpha)
        curl = np.array([-1j*np.cross(self.k, row, axisa=0, axisb=0, axisc=0) for row in beta])
        scale = max(1., float(np.linalg.norm(alpha.ravel())))
        return {'eigenstrain': (distortion + distortion.swapaxes(0, 1))/2, 'alpha_hat': alpha,
                'distortion': distortion, 'beta_hat': beta,
                'divergence_relative': float(np.linalg.norm(div.ravel())/scale),
                'curl_relative': float(np.linalg.norm((curl-alpha).ravel())/scale),
                'source_assembly_s': perf_counter()-started}

    def forces(self, loops, stress):
        """Galerkin nodal PK force from core-smoothed stress, plus residual core force."""
        stress_hat = np.fft.fftn(stress, axes=(-3, -2, -1)) / self.grid.n**3
        forces = []
        for loop in loops:
            filtered = (stress_hat * np.exp(-loop.core**2*self.k2/2)).reshape(3, 3, -1)
            force = np.zeros_like(loop.nodes)
            tangents = loop.segments / loop.lengths[:, None]
            for q, (node, segment) in enumerate(zip(loop.nodes, loop.segments)):
                for s, weight in zip(self.s, self.weight):
                    point = node + s*segment - self.origin
                    phase = np.exp(1j * np.einsum('i...,i->...', self.k, point)).ravel()
                    value = np.einsum('ijk,k->ij', filtered, phase).real
                    # t ds = segment ds_parameter; segment length is already included.
                    pk = np.cross(value @ loop.burgers, segment)
                    force[q] += weight*(1-s)*pk
                    force[(q+1) % len(force)] += weight*s*pk
            force += loop.core_tension * (tangents - np.roll(tangents, 1, axis=0))
            forces.append(force)
        return forces

    def evaluate(self, loops, mean_strain=None, background=None, need_forces=True):
        source = self.source(loops)
        eigenstrain = source['eigenstrain']
        if background is not None:
            eigenstrain = eigenstrain + background
        started = perf_counter()
        elastic = self.elastic.solve(eigenstrain, mean_strain)
        mechanics_s = perf_counter()-started
        started = perf_counter()
        forces = self.forces(loops, elastic['stress']) if need_forces else []
        force_s = perf_counter()-started
        return {'energy': elastic['energy'] + sum(loop.core_energy for loop in loops),
                'elastic_energy': elastic['energy'], 'core_energy': sum(loop.core_energy for loop in loops),
                'stress': elastic['stress'], 'strain': elastic['strain'], 'forces': forces,
                'equilibrium_rms': elastic['equilibrium_rms'],
                'divergence_relative': source['divergence_relative'], 'curl_relative': source['curl_relative'],
                'timings': {'source_s': source['source_assembly_s'], 'mechanics_s': mechanics_s, 'force_s': force_s}}


def velocities(loops, forces):
    rates, dissipation = [], 0.
    for loop, force in zip(loops, forces):
        projected = force.copy()
        if loop.glide_normal is not None:
            projected -= np.outer(projected @ loop.glide_normal, loop.glide_normal)
        measure = (loop.lengths + np.roll(loop.lengths, 1))/2
        rate = loop.mobility * projected / measure[:, None]
        rates.append(rate)
        dissipation += float(np.sum(force * rate))
    return rates, dissipation


def coupled_state(crystal, fields, loops, mechanics, need_forces=True):
    source = mechanics.source(loops)
    current = crystal.evaluate(*fields, extra_eigenstrain=source['eigenstrain'])
    current['components']['loop_core'] = sum(loop.core_energy for loop in loops)
    current['energy'] += current['components']['loop_core']
    current['source_divergence_relative'] = source['divergence_relative']
    current['source_curl_relative'] = source['curl_relative']
    if need_forces:
        current['line_forces'] = mechanics.forces(loops, current['stress'])
        current['line_rates'], line_dissipation = velocities(loops, current['line_forces'])
        current['dissipation'] += line_dissipation
    return current


def coupled_step(crystal, fields, loops, mechanics, dt, current=None, pinned=False):
    if dt <= 0:
        raise ValueError('Timestep must be positive.')
    current = coupled_state(crystal, fields, loops, mechanics) if current is None else current
    if pinned:
        current = dict(current)
        _, line_dissipation = velocities(loops, current['line_forces'])
        current['dissipation'] -= line_dissipation
        current['line_rates'] = [np.zeros_like(loop.nodes) for loop in loops]
    used = dt
    for trial in range(1, 25):
        new_fields = tuple(f + used*r for f, r in zip(fields, current['rates']))
        if new_fields[1].min() <= 0 or new_fields[1].max() >= 1:
            used *= .5
            continue
        new_loops = [loop.moved(loop.nodes+used*rate) for loop,rate in zip(loops,current['line_rates'])]
        new = coupled_state(crystal, new_fields, new_loops, mechanics, need_forces=False)
        bound = current['energy'] - .1*used*current['dissipation'] + 1e-12*max(1.,abs(current['energy']))
        if np.isfinite(new['energy']) and new['energy'] <= bound:
            new['line_forces'] = mechanics.forces(new_loops,new['stress'])
            new['line_rates'], line_dissipation = velocities(new_loops,new['line_forces'])
            new['dissipation'] += line_dissipation
            return new_fields, new_loops, new, used, trial
        used *= .5
    raise RuntimeError('Coupled loop step failed positivity or energy descent.')


def annihilate_coincident(first, second):
    """Exact-overlap opposite-charge reaction only; nearby loops are not removed."""
    matched = (first.nodes.shape == second.nodes.shape and first.core == second.core
               and np.array_equal(first.nodes, second.nodes) and np.array_equal(first.burgers, -second.burgers))
    if not matched:
        return [first, second], {'accepted': False, 'reason': 'complete sources do not cancel exactly'}
    return [], {'accepted': True, 'released_core_energy': first.core_energy+second.core_energy,
                'reason': 'coincident opposite Burgers sources and mean distortions cancel'}


def save_checkpoint(path, crystal, fields, loops, mechanics, time, dt, step, pinned=False):
    """Persist deterministic fields, line graph, constitutive state and integrator state."""
    metadata = {'grid': {'length': crystal.grid.length, 'n': crystal.grid.n},
                'parameters': asdict(crystal.p), 'mean_strain': crystal.mean_strain.tolist(),
                'schmid': crystal.schmid.tolist(), 'lam': crystal.elastic.lam, 'shear': crystal.elastic.shear,
                'loops': [loop.to_dict() for loop in loops], 'quadrature_order': mechanics.quadrature_order,
                'time': float(time), 'dt': float(dt), 'step': int(step), 'pinned': bool(pinned)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, phase=fields[0], population=fields[1], slip=fields[2], metadata=json.dumps(metadata))


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        fields = tuple(data[name].copy() for name in ('phase', 'population', 'slip'))
    grid = Cube(**metadata['grid'])
    crystal = CoupledCrystal(grid, Parameters3D(**metadata['parameters']),
                             np.array(metadata['mean_strain']), np.array(metadata['schmid']),
                             metadata['lam'], metadata['shear'])
    mechanics = LoopMechanics(grid, metadata['lam'], metadata['shear'], metadata['quadrature_order'])
    loops = [Loop(**value) for value in metadata['loops']]
    return crystal, fields, loops, mechanics, {key: metadata[key] for key in ('time', 'dt', 'step', 'pinned')}
