"""Independent Jacobian, estimator consistency and correction audit checks."""
import numpy as np
import pytest
from bmvr.continuum3d import Cube
from bmvr.coupled_transport import CoupledTransport
from bmvr.localized_transport import LocalizedTransport
from bmvr.trajectory_control import tangent_rate,TrajectoryControl


def example():
    g=Cube(12,7);x,y,z=g.coordinates();q=2*np.pi/12
    fields=(.3*np.cos(q*x)*np.sin(q*y),.2+.02*np.cos(q*(y+z)),.1*np.sin(q*(x+y))+.04*np.cos(q*z))
    return CoupledTransport(g,burgers=(.2,.1,0)),fields


def test_analytic_rate_jacobian_matches_independent_centered_differences():
    model,fields=example();rng=np.random.default_rng(35);direction=tuple(rng.normal(size=model.grid.shape) for _ in range(3))
    exact=tangent_rate(model,fields,direction)['rates'];delta=1e-6
    plus=model.evaluate(*(a+delta*b for a,b in zip(fields,direction)))['rates']
    minus=model.evaluate(*(a-delta*b for a,b in zip(fields,direction)))['rates']
    for a,b,c in zip(exact,plus,minus):assert np.max(abs(a-(b-c)/(2*delta)))<1e-8
    assert abs(model.grid.integrate(exact[1]))<1e-11


def test_taylor_remainder_scales_quadratically_and_energy_hessian_is_reciprocal():
    model,fields=example();rng=np.random.default_rng(351);directions=[tuple(rng.normal(size=model.grid.shape) for _ in range(3)) for _ in range(2)]
    tangent=[tangent_rate(model,fields,d) for d in directions];state=model.evaluate(*fields);errors=[]
    for epsilon in (.001,.0005):
        shifted=model.evaluate(*(a+epsilon*b for a,b in zip(fields,directions[0])))
        errors.append(np.sqrt(sum(np.sum((a-b-epsilon*j)**2) for a,b,j in zip(shifted['rates'],state['rates'],tangent[0]['rates']))))
    assert 3.9<errors[0]/errors[1]<4.1
    ab=sum(model.grid.integrate(a*b) for a,b in zip(directions[0],tangent[1]['force_variation']))
    ba=sum(model.grid.integrate(a*b) for a,b in zip(directions[1],tangent[0]['force_variation']))
    assert ab==pytest.approx(ba,rel=1e-12,abs=1e-10)


def test_first_step_error_is_exact_for_same_accepted_timestep():
    model,fields=example();control=TrajectoryControl(LocalizedTransport(model,fields,reset_at_cap=True),correct=False)
    reduced,used,_,_=control.step(.01)
    full,_,dt,_=model.step(fields,.01);assert dt==used
    for e,a,b in zip(control.error,full,reduced['fields']):assert np.max(abs(e-(a-b)))<1e-15


def test_zero_cap_full_evolution_has_only_roundoff_estimated_error():
    model,fields=example();control=TrajectoryControl(LocalizedTransport(model,fields,caps=(0,0,0)))
    for _ in range(4):control.step(.01)
    assert max(np.max(abs(e)) for e in control.error)<1e-14
    assert control.corrections==control.rejections==0


def test_energy_decreasing_transfer_rebases_and_preserves_inventory():
    model,fields=example();control=TrajectoryControl(LocalizedTransport(model,fields))
    before=control.chart.state(project=False);candidate,reference,_,_=model.step(fields,.001)
    control.error=tuple(a-b for a,b in zip(candidate,fields))
    following,event=control.transfer(before,candidate,reference)
    assert event['accepted'] and control.corrections==1
    assert all(np.array_equal(a,b) for a,b in zip(following['fields'],candidate))
    assert all(np.count_nonzero(e)==0 for e in control.error)
    assert abs(model.grid.integrate(candidate[1]-fields[1]))<1e-12
    assert all(len(m)==0 for m in control.chart.modes)


@pytest.mark.parametrize('violation',['energy','inventory','positivity'])
def test_rejected_transfer_retains_fields_and_promotes(violation):
    model,fields=example();control=TrajectoryControl(LocalizedTransport(model,fields));before=control.chart.state(project=False)
    candidate=list(fields)
    if violation=='energy':candidate[0]=fields[0]+.001*before['forces'][0]
    elif violation=='inventory':candidate[1]=fields[1]+.001
    else:candidate[1]=-fields[1]
    following,event=control.transfer(before,tuple(candidate),model.evaluate(*candidate))
    assert not event['accepted'] and control.disabled and control.rejections==1
    assert control.chart.full==[True]*3
    assert all(np.array_equal(a,b) for a,b in zip(following['fields'],fields))


def test_controller_restart_preserves_error_and_next_correction(tmp_path):
    model,fields=example();control=TrajectoryControl(LocalizedTransport(model,fields,reset_at_cap=True),field_limits=(1e-7,)*3,energy_limit=1e-7)
    for _ in range(3):control.step(.03)
    path=tmp_path/'control.npz';control.save(path);restored=TrajectoryControl.load(path)
    a,dt,_,event=control.step(.03);b,rdt,_,revent=restored.step(.03)
    assert dt==rdt and event==revent
    assert control.corrections==restored.corrections
    for x,y in zip(a['fields'],b['fields']):assert np.array_equal(x,y)
    for x,y in zip(control.error,restored.error):assert np.array_equal(x,y)
