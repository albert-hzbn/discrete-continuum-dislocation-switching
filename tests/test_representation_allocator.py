import numpy as np
from bmvr.representation_allocator import GlideEnsemble, Run


def test_conversions_and_transport_conserve_burgers_content_exactly():
    rng = np.random.default_rng(3); e = GlideEnsemble(64, b=1e-4, M=1e4, tau=.5, block=8, n_c=16)
    x = np.r_[.5 + .05*rng.standard_normal(3000), rng.random(40)] % 1; y = np.r_[.5 + .05*rng.standard_normal(3000), rng.random(40)] % 1
    s = np.where(rng.random(3040) < .6, 1, -1); ids = np.arange(3040)
    r = Run(e, x, y, s, ids, 'H', realloc=5); n0, b0 = r.totals()
    assert r.cont.any() and 0 < len(r.x) < 3040
    for _ in range(12): r.step(.2*e.h/(e.M*e.b*e.tau))
    n1, b1 = r.totals(); assert abs(n1 - n0) < 1e-9 and abs(b1 - b0) < 1e-9
    assert max(l['count_error'] for l in r.ledger) < 1e-9


def test_single_dislocation_stress_has_the_edge_sign_and_magnitude():
    e = GlideEnsemble(256, b=1.); s = e.stress(e.deposit(np.array([.5]), np.array([.5]), np.array([1.])))
    i = int(round(.56/e.h - .5)); j = int(round(.5/e.h - .5)); x = (i + .5)*e.h - .5
    assert .9 < s[i, j]/(1/(2*np.pi*.7*x)) < 1.


def test_a_higher_threshold_never_makes_more_blocks_continuum():
    rng = np.random.default_rng(5); x = (.5 + .06*rng.standard_normal(4000)) % 1; y = (.5 + .06*rng.standard_normal(4000)) % 1
    s = np.where(rng.random(4000) < .5, 1, -1); ids = np.arange(4000); counts = []
    for nc in (4, 16, 64, 256):
        e = GlideEnsemble(64, b=1e-4, M=1e4, tau=.5, block=8, n_c=nc); counts.append(int(Run(e, x, y, s, ids, 'H').cont.sum()))
    assert all(a >= b for a, b in zip(counts, counts[1:])) and counts[0] > counts[-1]
