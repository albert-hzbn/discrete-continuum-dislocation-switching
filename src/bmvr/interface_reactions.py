"""Synthetic orientation-resolved interface kinetics with exact charge bookkeeping.

This finite-state closure does not replace a spatial reactive line-network solve.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from scipy.special import xlogy


NAMES = ("incident", "transmitted", "absorbed", "reflected")


@dataclass(frozen=True)
class InterfaceParameters:
    angle_degrees: float = 20.
    core_coefficient: float = 1.
    residual_stiffness: float = 2.
    binding_energy: float = .4
    drive_work: float = .8
    thermal_energy: float = .2
    attempt_rate: float = 1.
    transmission_barrier: float = .15
    absorption_barrier: float = .5
    reflection_barrier: float = .3
    misalignment_barrier: float = .4


class InterfaceReactionModel:
    def __init__(self, parameters=InterfaceParameters()):
        self.parameters = p = parameters
        if p.thermal_energy <= 0 or p.attempt_rate <= 0:
            raise ValueError("Thermal energy and attempt rate must be positive.")
        if min(p.transmission_barrier, p.absorption_barrier, p.reflection_barrier,
               p.misalignment_barrier, p.core_coefficient, p.residual_stiffness) < 0:
            raise ValueError("Barrier and quadratic energy coefficients must be nonnegative.")
        theta = np.deg2rad(p.angle_degrees)
        rotation = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                             [0, np.sin(theta), np.cos(theta)]])
        self.incident_burgers = np.array([0., 1., -1.]) / np.sqrt(2)
        self.incident_plane = np.ones(3) / np.sqrt(3)
        self.outgoing_burgers = rotation @ self.incident_burgers
        self.outgoing_plane = rotation @ self.incident_plane
        self.alignment = abs(self.incident_burgers @ self.outgoing_burgers) * abs(self.incident_plane @ self.outgoing_plane)
        zero = np.zeros(3)
        self.mobile = np.array([self.incident_burgers, self.outgoing_burgers, zero, self.incident_burgers])
        self.residual = self.incident_burgers[None, :] - self.mobile
        self.mobile_count = np.array([1, 1, 0, 1])
        self.normal_direction = np.array([1, 1, 0, -1])
        b2 = np.sum(self.mobile * self.mobile, axis=1)
        rb2 = np.sum(self.residual * self.residual, axis=1)
        work = np.array([0., p.drive_work * np.cos(theta), 0., -p.drive_work])
        self.energies = p.core_coefficient * b2 + .5 * p.residual_stiffness * rb2 - work
        self.energies[2] -= p.binding_energy
        barriers = [p.transmission_barrier + p.misalignment_barrier * (1 - self.alignment),
                    p.absorption_barrier, p.reflection_barrier]
        self.rates = np.zeros((4, 4))
        for j, barrier in enumerate(barriers, 1):
            saddle = max(self.energies[0], self.energies[j]) + barrier
            self.rates[0, j] = p.attempt_rate * np.exp(-(saddle - self.energies[0]) / p.thermal_energy)
            self.rates[j, 0] = p.attempt_rate * np.exp(-(saddle - self.energies[j]) / p.thermal_energy)
        self.generator = self.rates.T.copy()
        np.fill_diagonal(self.generator, -self.rates.sum(axis=1))
        weights = np.exp(-(self.energies - self.energies.min()) / p.thermal_energy)
        self.stationary = weights / weights.sum()

    def charge(self, state):
        return self.mobile[state] + self.residual[state]

    def event(self, previous, following):
        if self.rates[previous, following] <= 0:
            raise ValueError("This transition is not in the reaction graph.")
        error = float(np.linalg.norm(self.charge(following) - self.charge(previous)))
        if error > 1e-12:
            raise ValueError("The reaction would violate Burgers conservation.")
        return {"from": NAMES[previous], "to": NAMES[following],
                "mobile_burgers": self.mobile[following].tolist(),
                "residual_burgers": self.residual[following].tolist(),
                "mobile_count": int(self.mobile_count[following]),
                "normal_travel_direction": int(self.normal_direction[following]),
                "burgers_error": error,
                "configuration_energy_change": float(self.energies[following] - self.energies[previous])}

    def first_event_probabilities(self):
        rate = self.rates[0].sum()
        if rate == 0:
            raise ValueError("No representable first-event rate.")
        return self.rates[0] / rate

    def sample_first_events(self, count=20000, seed=20260912, overlap=1.):
        if overlap <= 0:
            raise ValueError("Positive encounter overlap is required for finite waiting times.")
        rng = np.random.default_rng(seed)
        waiting = rng.exponential(1 / (overlap * self.rates[0].sum()), count)
        states = rng.choice(4, size=count, p=self.first_event_probabilities())
        return waiting, states

    def propagate(self, exposure, initial=None):
        p0 = np.array([1., 0, 0, 0]) if initial is None else np.asarray(initial, dtype=float)
        if p0.shape != (4,) or p0.min() < 0 or not np.isclose(p0.sum(), 1):
            raise ValueError("Initial probabilities must form a four-state distribution.")
        exposures = np.asarray(exposure)
        if exposures.min() < 0:
            raise ValueError("Integrated exposure cannot be negative.")
        return np.array([expm(self.generator * t) @ p0 for t in exposures])

    def ensemble_free_energy(self, probabilities):
        probabilities = np.asarray(probabilities)
        if probabilities.min() < -1e-12:
            raise ValueError("Negative probability exceeds roundoff tolerance.")
        safe = np.maximum(probabilities, 0)
        return safe @ self.energies + self.parameters.thermal_energy * np.sum(xlogy(safe, safe), axis=-1)

    def entropy_production(self, probabilities):
        p = np.asarray(probabilities)
        total = 0.
        for i in range(4):
            for j in range(i + 1, 4):
                if self.rates[i, j] == 0:
                    continue
                a, b = p[i] * self.rates[i, j], p[j] * self.rates[j, i]
                if max(a, b) == 0:
                    continue
                if min(a, b) <= 0:
                    return np.inf
                total += (a - b) * np.log(a / b)
        return float(total)  # Entropy rate in units of k_B.


def approach_exposure(times, start=-6., speed=.5, width=np.sqrt(2)):
    """Exact integral of sech^2(s/width), approaching the boundary then holding."""
    times = np.asarray(times)
    if start >= 0 or speed <= 0 or width <= 0 or times.min() < 0:
        raise ValueError("Use a positive-width approach starting on the incident side.")
    arrival = -start / speed
    moving_time = np.minimum(times, arrival)
    distance = start + speed * moving_time
    exposure = width / speed * (np.tanh(distance / width) - np.tanh(start / width))
    exposure += np.maximum(times - arrival, 0)
    return distance, exposure
