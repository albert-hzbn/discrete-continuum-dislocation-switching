"""Charge- and energy-checked reactions for compatible straight screw dipoles."""

import numpy as np
from .defects import Screws


def annihilate(screws, capture=None):
    """Only the pair/core closure is included; not a general interface reaction."""
    capture = screws.core if capture is None else capture
    before, _ = screws.pair()
    for i in range(len(screws.burgers)):
        for j in range(i + 1, len(screws.burgers)):
            same_glide = abs(screws.positions[i, 1] - screws.positions[j, 1]) < 1e-12
            neutral = abs(screws.burgers[i] + screws.burgers[j]) < 1e-12
            if neutral and same_glide and np.linalg.norm(screws.positions[i] - screws.positions[j]) <= capture:
                keep = np.ones(len(screws.burgers), dtype=bool)
                keep[[i, j]] = False
                candidate = Screws(screws.positions[keep], screws.burgers[keep], screws.core,
                                   screws.shear, screws.reference_radius, screws.mobility)
                after, _ = candidate.pair()
                if after <= before + 1e-12:
                    return candidate, {"occurred": True, "released_energy": before - after,
                                       "burgers_error": float(candidate.burgers.sum() - screws.burgers.sum())}
    return screws, {"occurred": False, "released_energy": 0.0, "burgers_error": 0.0}
