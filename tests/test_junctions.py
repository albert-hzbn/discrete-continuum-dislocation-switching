"""Section 54: junction formation/dissolution between dislocation lines on intersecting glide planes."""
import numpy as np
import pytest
from bmvr.continuum3d import Cube
from bmvr.graphs3d import (GraphMechanics,junction_pair,dissolve_junction,junction_length,line_state,line_step,
                           DislocationGraph)


@pytest.mark.parametrize('case',['favorable','neutral','unfavorable'])
def test_junction_and_dissolution_are_charge_exact(case):
    pair=junction_pair(case);crossed,junction=pair['crossed'],pair['junction'];mechanics=GraphMechanics(Cube(12,13))
    assert crossed.balance_error<1e-14 and junction.balance_error<1e-14
    np.testing.assert_allclose(junction.burgers[2],pair['b1']+pair['b2'],atol=1e-14)
    for graph in (crossed,junction,dissolve_junction(junction,crossed)):
        source=mechanics.source(graph)
        assert source['divergence_relative']<1e-10 and source['curl_relative']<1e-10
    t=pair['t'];off=junction.nodes[6:]-np.outer(junction.nodes[6:]@t,t)
    assert np.max(abs(off))<1e-12
    expected={'favorable':-.02,'neutral':0.,'unfavorable':.02}[case]
    assert pair['b1']@pair['b2']==pytest.approx(expected,abs=1e-14)
    with pytest.raises(ValueError,match='balance'):
        DislocationGraph(junction.nodes,junction.edges,np.vstack([junction.burgers[:2],junction.burgers[1:2],junction.burgers[3:]]),core=junction.core)


def test_line_tension_junction_equilibrium_matches_closed_form():
    """Core energy only, pinned anchors: Gamma_3 = sum Gamma_i cos beta_i gives y_P = -1 + sqrt(0.140625/0.75)."""
    junction=junction_pair('favorable')['junction'];exact=-1+np.sqrt(.140625/.75)
    # The implemented force vanishes along the line at the closed-form equilibrium and not at the embryo.
    at=junction.nodes.copy();at[6,1],at[7,1]=exact,-exact
    projected=np.einsum('nij,nj->ni',junction.projectors,line_state(junction.moved(at))['forces'])
    assert np.max(abs(projected))<1e-14 and np.max(abs(np.einsum('nij,nj->ni',junction.projectors,line_state(junction)['forces'])))>1e-3
    graph,state=junction,None
    # Slow mode (rate ~2.5e-3): large steps, still energy-backtracked.
    for _ in range(2000):graph,state,_=line_step(graph,20.,None,state)
    assert graph.nodes[6,1]==pytest.approx(exact,rel=1e-6) and graph.nodes[7,1]==pytest.approx(-exact,rel=1e-6)


def test_unfavorable_line_tension_shrinks_and_energy_decreases():
    junction=junction_pair('unfavorable')['junction'];graph,state=junction,line_state(junction);energy=[state['energy']]
    for _ in range(20):graph,state,_=line_step(graph,.05,None,state);energy.append(state['energy'])
    assert junction_length(graph)<junction_length(junction) and np.all(np.diff(energy)<=1e-12)
