"""Shared-energy loop/Nye-field promotion and audited known-topology reconstruction.

Small strain, periodic homogeneous elasticity, quadratic density core energy.
Line and continuum drag are separate closures; equal state energy does not imply
kinetic equivalence. No unknown-topology or nucleation reconstruction is claimed.
"""
import json
import numpy as np
from scipy.optimize import least_squares
from .continuum3d import Cube
from .loops3d import Loop, LoopMechanics, velocities


class NyeTransfer:
    def __init__(self, grid, core_coefficient=.3, field_mobility=10., glide_normal=None,
                 mean_strain=None, background=None, lam=2., shear=1., quadrature_order=10):
        if core_coefficient <= 0 or field_mobility <= 0:
            raise ValueError('Positive density-core coefficient and field mobility are required.')
        self.grid = grid
        self.core_coefficient, self.field_mobility = core_coefficient, field_mobility
        self.mechanics = LoopMechanics(grid, lam, shear, quadrature_order)
        self.mean_strain = np.zeros((3, 3)) if mean_strain is None else np.asarray(mean_strain, dtype=float)
        self.background = np.zeros((3, 3)+grid.shape) if background is None else np.asarray(background, dtype=float)
        self.glide_normal = None if glide_normal is None else np.asarray(glide_normal, dtype=float)
        if self.glide_normal is not None and (self.glide_normal.shape != (3,) or not np.isclose(np.linalg.norm(self.glide_normal), 1)):
            raise ValueError('Glide normal must be a unit vector.')

    def curl(self, tensor):
        hat = np.fft.fftn(tensor, axes=(-3, -2, -1))
        return np.fft.ifftn(np.array([1j*np.cross(self.mechanics.k, row, axisa=0, axisb=0, axisc=0) for row in hat]),
                           axes=(-3, -2, -1)).real

    def evaluate(self, beta):
        beta = np.asarray(beta)
        if beta.shape != (3, 3)+self.grid.shape or not np.isfinite(beta).all():
            raise ValueError('A finite full plastic-distortion tensor field is required.')
        alpha = -self.curl(beta)
        elastic = self.mechanics.elastic.solve((beta+beta.swapaxes(0, 1))/2+self.background, self.mean_strain)
        core = self.core_coefficient/2*self.grid.integrate(alpha*alpha)
        H = elastic['stress']+self.core_coefficient*self.curl(alpha)
        force = sum(np.cross(H[i], alpha[i], axisa=0, axisb=0, axisc=0) for i in range(3))
        projected = force.copy()
        if self.glide_normal is not None:
            projected -= self.glide_normal[:, None, None, None]*np.einsum('i,i...->...', self.glide_normal, force)
        velocity = self.field_mobility*projected
        rate = np.array([np.cross(row, velocity, axisa=0, axisb=0, axisc=0) for row in alpha])
        ahat = np.fft.fftn(alpha, axes=(-3, -2, -1))
        divergence = np.einsum('j...,ij...->i...', self.mechanics.k, ahat)
        return {'energy': elastic['energy']+core, 'elastic_energy': elastic['energy'], 'core_energy': core,
                'alpha': alpha, 'stress': elastic['stress'], 'strain': elastic['strain'], 'H': H,
                'body_force': force, 'velocity': velocity, 'rate': rate,
                'dissipation': self.grid.integrate(force*velocity),
                'divergence_relative': float(np.linalg.norm(divergence.ravel())/max(np.linalg.norm(ahat.ravel()), 1e-30)),
                'mean_beta': beta.mean(axis=(2, 3, 4)), 'equilibrium_rms': elastic['equilibrium_rms']}

    def promote(self, loops, mean_correction=None):
        if any(loop.core_tension != 0 for loop in loops):
            raise ValueError('Separate line tension must be zero for the shared density-core closure.')
        beta = self.mechanics.source(loops)['distortion']
        if mean_correction is not None:
            correction = np.asarray(mean_correction)
            if correction.shape != (3, 3) or not np.isfinite(correction).all():
                raise ValueError('Mean plastic correction must be a finite 3x3 tensor.')
            beta += correction[..., None, None, None]
        return beta

    def line_state(self, loops, mean_correction=None):
        beta = self.promote(loops, mean_correction)
        state = self.evaluate(beta)
        # LoopMechanics contracts matrix @ b; row-wise beta work requires H.T @ b.
        state['line_forces'] = self.mechanics.forces(loops, state['H'].swapaxes(0, 1))
        state['line_rates'], state['line_dissipation'] = velocities(loops, state['line_forces'])
        return state

    def field_step(self, beta, dt):
        if dt <= 0:
            raise ValueError('Positive timestep required.')
        current = self.evaluate(beta)
        for trials in range(1, 31):
            candidate = beta+dt*current['rate']
            following = self.evaluate(candidate)
            if following['energy'] <= current['energy']-.1*dt*current['dissipation']+1e-12*max(1., abs(current['energy'])):
                return candidate, following, dt, trials
            dt *= .5
        raise RuntimeError('Continuum defect step failed energy descent.')

    def line_step(self, loops, mean_correction, dt):
        if dt <= 0:
            raise ValueError('Positive timestep required.')
        current = self.line_state(loops, mean_correction)
        for trials in range(1, 31):
            candidate = [loop.moved(loop.nodes+dt*rate) for loop, rate in zip(loops, current['line_rates'])]
            following = self.line_state(candidate, mean_correction)
            if following['energy'] <= current['energy']-.1*dt*current['line_dissipation']+1e-12*max(1., abs(current['energy'])):
                return candidate, following, dt, trials
            dt *= .5
        raise RuntimeError('Line-chart step failed energy descent.')

    def recover(self, beta, hints, source_tolerance=.01, stress_tolerance=.01,
                force_tolerance=.01, energy_tolerance=.001, max_evaluations=50):
        """Fit known connectivity; retain the field unchanged if any audit refuses."""
        if not hints:
            raise ValueError('A known-topology chart is required; unknown topology is not inferred.')
        if min(source_tolerance, stress_tolerance, force_tolerance, energy_tolerance) < 0:
            raise ValueError('Reconstruction budgets must be nonnegative.')
        reference = self.evaluate(beta)
        target_hat = np.fft.fftn(reference['alpha'], axes=(-3, -2, -1))
        scale = max(float(np.linalg.norm(target_hat.ravel())), 1e-12)
        bases, slices, offset = [], [], 0
        for loop in hints:
            if loop.core_tension != 0:
                raise ValueError('Use zero separate core tension for reconstruction.')
            if loop.glide_normal is None:
                basis = np.eye(3)
            else:
                _, _, vt = np.linalg.svd(loop.glide_normal[None])
                basis = vt[1:].T
            bases.append(basis)
            count = len(loop.nodes)*basis.shape[1]
            slices.append(slice(offset, offset+count))
            offset += count
        def construct(x):
            return [loop.moved(loop.nodes+x[sl].reshape(len(loop.nodes), basis.shape[1])@basis.T)
                    for loop, basis, sl in zip(hints, bases, slices)]
        def residual(x):
            candidate = construct(x)
            difference = (self.mechanics.source(candidate)['alpha_hat']-target_hat)/scale
            return np.r_[difference.real.ravel(), difference.imag.ravel()]
        try:
            fit = least_squares(residual, np.zeros(offset), ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=max_evaluations)
            loops = construct(fit.x)
        except (ValueError, np.linalg.LinAlgError) as error:
            return {'accepted': False, 'reason': 'invalid_or_unresolved_chart', 'detail': str(error),
                    'loops': None, 'mean_correction': None, 'beta': np.array(beta, copy=True),
                    'errors': {}, 'optimizer_success': False, 'optimizer_evaluations': None, 'fitted_nodes': []}
        base = self.promote(loops)
        correction = reference['mean_beta']-base.mean(axis=(2, 3, 4))
        candidate_beta = self.promote(loops, correction)
        candidate = self.evaluate(candidate_beta)
        initial_fit_energy_jump = candidate['energy']-reference['energy']
        energy_roundoff = 1e-12*max(1., abs(reference['energy']))
        corrections = 0
        # Fit source first, then descend within the chart at fixed mean plastic history.
        # This cannot manufacture missing topology: the final source/force budgets still apply.
        for corrections in range(7):
            excess = candidate['energy']-reference['energy']
            if excess <= energy_roundoff or corrections == 6:
                break
            fixed_mean_H = candidate['H']-candidate['H'].mean(axis=(2, 3, 4))[..., None, None, None]
            nodal = self.mechanics.forces(loops, fixed_mean_H.swapaxes(0, 1))
            rates, decrease = velocities(loops, nodal)
            if decrease <= 1e-30:
                break
            used = 1.05*excess/decrease
            for _ in range(20):
                try:
                    trial = [loop.moved(loop.nodes+used*rate) for loop,rate in zip(loops,rates)]
                    base = self.promote(trial)
                    trial_correction = reference['mean_beta']-base.mean(axis=(2,3,4))
                    trial_beta = self.promote(trial, trial_correction)
                    trial_state = self.evaluate(trial_beta)
                    if trial_state['energy'] < candidate['energy']:
                        loops, correction, candidate_beta, candidate = trial, trial_correction, trial_beta, trial_state
                        break
                except ValueError:
                    pass
                used *= .5
            else:
                break
        norm = lambda value: float(np.linalg.norm(value.ravel()))
        relative = lambda a, b: norm(a-b)/max(norm(a), 1e-12)
        field_forces = np.concatenate(self.mechanics.forces(loops, reference['H'].swapaxes(0, 1)))
        candidate_forces = np.concatenate(self.mechanics.forces(loops, candidate['H'].swapaxes(0, 1)))
        errors = {'alpha_relative': relative(reference['alpha'], candidate['alpha']),
                  'stress_relative': relative(reference['stress'], candidate['stress']),
                  'force_relative': relative(field_forces, candidate_forces),
                  'energy_relative': abs(candidate['energy']-reference['energy'])/max(abs(reference['energy']), 1e-12),
                  'energy_jump': candidate['energy']-reference['energy'],
                  'mean_beta_absolute': float(np.max(np.abs(candidate['mean_beta']-reference['mean_beta']))),
                  'field_rate_relative': relative(reference['rate'], candidate['rate'])}
        accepted = bool(fit.success and errors['alpha_relative'] <= source_tolerance
                        and errors['stress_relative'] <= stress_tolerance and errors['force_relative'] <= force_tolerance
                        and errors['energy_relative'] <= energy_tolerance and errors['mean_beta_absolute'] <= 1e-10
                        and errors['energy_jump'] <= energy_roundoff)
        return {'accepted': accepted, 'reason': 'audits_met' if accepted else 'fit_or_representation_budget_failed',
                'loops': loops if accepted else None, 'mean_correction': correction if accepted else None,
                'beta': candidate_beta if accepted else np.array(beta, copy=True), 'errors': errors,
                'optimizer_success': bool(fit.success), 'optimizer_evaluations': fit.nfev,
                'initial_fit_energy_jump': initial_fit_energy_jump, 'energy_corrections': corrections,
                'fitted_nodes': [loop.nodes.tolist() for loop in loops]}

    def save(self, path, beta, hints, time, step):
        metadata = {'length': self.grid.length, 'n': self.grid.n, 'core_coefficient': self.core_coefficient,
                    'field_mobility': self.field_mobility, 'glide_normal': None if self.glide_normal is None else self.glide_normal.tolist(),
                    'mean_strain': self.mean_strain.tolist(), 'lam': self.mechanics.elastic.lam,
                    'shear': self.mechanics.elastic.shear, 'quadrature_order': self.mechanics.quadrature_order,
                    'hints': [loop.to_dict() for loop in hints], 'time': time, 'step': step}
        np.savez_compressed(path, beta=beta, background=self.background, metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            m = json.loads(str(data['metadata']))
            grid = Cube(m.pop('length'), m.pop('n'))
            hints = [Loop(**item) for item in m.pop('hints')]
            clock = {'time': m.pop('time'), 'step': m.pop('step')}
            model = cls(grid, background=data['background'].copy(), **m)
            return model, data['beta'].copy(), hints, clock
