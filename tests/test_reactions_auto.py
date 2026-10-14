import numpy as np
from bmvr import graphs3d as G
from bmvr import reactions_auto as R


def test_automatic_junction_equals_hand_built_and_rejects_unfavourable():
    theta = np.arccos(1/3); nA = np.array([0, 0, 1.]); nB = np.array([np.sin(theta), 0, np.cos(theta)])
    pair = G.junction_pair('favorable'); c = pair['crossed']
    a, b = R.crossing_nodes(c, R.detect(c, .05))
    info = R.classify(c, a, b, nA, nB)
    assert info['favorable'] and np.allclose(info['junction_burgers'], pair['b1']+pair['b2'])
    auto = R.form_junction(c, a, b, info, .4)
    assert auto.core_energy == pair['junction'].core_energy
    bad = G.junction_pair('unfavorable')['crossed']; a, b = R.crossing_nodes(bad, R.detect(bad, .05))
    assert not R.classify(bad, a, b, nA, nB)['favorable']


def test_segment_distance_is_exact_for_known_cases():
    s, t, d = R.segment_closest(np.zeros(3), np.array([1., 0, 0]), np.array([.5, 1, 1]), np.array([.5, -1, 1]))
    assert np.isclose(d, 1.) and np.isclose(s, .5) and np.isclose(t, .5)
