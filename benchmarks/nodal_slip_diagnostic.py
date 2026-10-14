"""Section 45.6 diagnostics: slip-error location and event-timing convergence."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from bmvr.nodal_damage import NodalThermalFracture2D


def cross(t,y,level):
    i=np.flatnonzero(y>=level)
    if not len(i):return None
    i=int(i[0]);return float(t[i-1]+(level-y[i-1])/(y[i]-y[i-1])*(t[i]-t[i-1]))


def slip_location(base):
    def load(n):return NodalThermalFracture2D.load(base/f'nodal-tensile-{n}-0.00025-final.npz')[:2]
    (ma,sa),(mb,sb)=load(24),load(32)
    def grid(m,n):return m.mechanics.mesh.quadrature_points.reshape(n,n,3,3,2).transpose(0,2,1,3,4).reshape(3*n,3*n,2)
    xb=grid(mb,32);gb=sb.gamma.reshape(32,32,3,3,2).transpose(0,2,1,3,4).reshape(96,96,2)
    target=RegularGridInterpolator((xb[:,0,0],xb[0,:,1]),gb)(grid(ma,24)).reshape(24,3,24,3,2).transpose(0,2,1,3,4).reshape(576,9,2)
    w=ma.mechanics.mesh.weights[None,:,None];e=(w*(sa.gamma-target)**2).sum(-1);p=ma.mechanics.mesh.quadrature_points
    mismatch=np.any(sa.gamma!=0,-1)^np.any(target!=0,-1);notch=(abs(p[...,1]-.5)<.16)&(p[...,0]<.75)
    d=ma.damage_quadrature(sa.damage)
    result=dict(relative_slip_24_32=float(np.sqrt(e.sum()/np.sum(w*target**2))),
        squared_error_fraction_active_set_mismatch=float(e[mismatch].sum()/e.sum()),squared_error_fraction_notch=float(e[notch].sum()/e.sum()))
    for lo,hi in ((0,.05),(.05,.3),(.3,.7),(.7,1.01)):result[f'squared_error_fraction_damage_{lo}_{hi}']=float(e[(d>=lo)&(d<hi)].sum()/e.sum())
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,default=Path('results/nodal-damage-constrained-2026-09-26'))
    parser.add_argument('--refinement',type=Path,default=Path('results/nodal-refinement-2026-09-26'));args=parser.parse_args()
    runs={n:json.loads((args.base/f'nodal-tensile-{n}-0.00025.json').read_text()) for n in (16,24,32)}
    refinement=json.loads((args.refinement/'report.json').read_text());runs[48]=refinement['run']
    timing={}
    for n,r in runs.items():
        rows=r['rows'];t=np.array([x['time'] for x in rows]);D=np.array([x['damage_integral'] for x in rows])
        md=np.array([x['max_damage'] for x in rows]);ke=np.array([x['kinetic_energy'] for x in rows])
        timing[str(n)]=dict(distributed_damage_plus_0005=cross(t,D-D[0],.005),distributed_damage_plus_001=cross(t,D-D[0],.01),
            tip_max_damage_09=cross(t,md,.9),tip_max_damage_099=cross(t,md,.99),kinetic_energy_2e4=cross(t,ke,2e-4),
            first_saturation=next((x['time'] for x in rows[1:] if x['newly_saturated_damage_nodes']),None),
            final_kinetic_energy=float(ke[-1]),final_max_slip=float(rows[-1]['max_slip']))
    h=np.array([1/24,1/32,1/48]);tip=np.array([timing[k]['tip_max_damage_09'] for k in ('24','32','48')])
    ratio=(tip[0]-tip[1])/(tip[1]-tip[2]);orders=np.linspace(.2,4,3801)
    fit=orders[np.argmin(abs((h[0]**orders-h[1]**orders)/(h[1]**orders-h[2]**orders)-ratio))]
    C=(tip[1]-tip[2])/(h[1]**fit-h[2]**fit)
    estimate=dict(observed_order=float(fit),extrapolated_tip_time=float(tip[2]-C*h[2]**fit),n48_relative_timing_error=float(C*h[2]**fit/(tip[2]-C*h[2]**fit)),
        caution='Three-point Richardson estimate from successive differences; not an established asymptotic rate.')
    report=dict(slip_location=slip_location(args.base),timing=timing,tip_timing_estimate=estimate,slip_measures=refinement['slip_measures'])
    (args.refinement/'timing-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=1))


if __name__=='__main__':main()
