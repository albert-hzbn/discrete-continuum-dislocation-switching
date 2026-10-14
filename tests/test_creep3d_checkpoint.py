"""A run resumed from a checkpoint reproduces the uninterrupted run (Section 98 restart support)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from bmvr.creep3d import Copper773, Creep3D, Params3D, to_host  # noqa: E402


@pytest.mark.parametrize('model', ['A', 'C'])
def test_checkpoint_restart_is_exact(tmp_path, model):
    p = Params3D(n=16)
    full = Creep3D(Copper773(), 60., model, p)
    full.run(0.3, dt_mech=.1)
    first = Creep3D(Copper773(), 60., model, p)
    first.run(0.1, dt_mech=.1)
    first.checkpoint(tmp_path/'s.ckpt')
    resumed = Creep3D.restore(tmp_path/'s.ckpt')
    resumed.run(0.3, dt_mech=.1, resume=True)
    assert resumed.t == pytest.approx(full.t)
    assert np.array_equal(resumed.c, full.c) and np.array_equal(resumed.R, full.R)
    assert len(resumed.history) == len(full.history)
    if full.mech_on:
        assert np.array_equal(to_host(resumed.q), to_host(full.q))
        assert np.array_equal(to_host(resumed.H), to_host(full.H))
