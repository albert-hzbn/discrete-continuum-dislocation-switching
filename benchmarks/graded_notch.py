"""Section 46.4-46.6 graded notch refinement with the stiff-block preconditioned midpoint solve."""
import argparse,json,os,platform
from bmvr import _resource as resource
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
from bmvr.quadratic_mesh import graded_edges
from bmvr.nodal_damage import NodalThermalFracture2D
from thermofracture import run
from thermochemical import write_json
import finite_damage_split
from nodal_damage import make_model as uniform_model

# Section 46.5 revision: near-isotropic cells along the whole damaged band (first design retained as a failure).
BANDS=dict(x=(-.10,.55),y=(.25,.75),ratio=3.,width=.03)


def edges(n):
    return (graded_edges(n,band=BANDS['x'],ratio=BANDS['ratio'],width=BANDS['width']),
            graded_edges(n,band=BANDS['y'],ratio=BANDS['ratio'],width=BANDS['width']))


def make_model(n,variant):
    def model_class(n_,**kwargs):return NodalThermalFracture2D(n_,edges=edges(n_),preconditioned=True,**kwargs)
    return finite_damage_split.make_model(n,variant,model_class=model_class)


def axes(mesh):
    return (np.linspace(0,mesh.length,mesh.n+1),)*2 if mesh.uniform else mesh.edges


def gauss_axes(mesh):
    q=(np.polynomial.legendre.leggauss(3)[0]+1)/2
    return tuple((e[:-1,None]+np.diff(e)[:,None]*q[None,:]).ravel() for e in axes(mesh))


def load(path,loader=None):return (loader or NodalThermalFracture2D).load(path)[:2]


def cellwise_slip(A,B,gamma):
    """Evaluate B's quadrature-point slip at A's quadrature points by B's cell-local Q2 interpolant (Section 47 GLL fields
    are discontinuous across cells; boundary points are assigned to the B cell overlapping A's cell interior)."""
    from bmvr.quadratic_mesh import quadratic_shape
    centers=A.coordinates[A.cells].mean(axis=1)[:,None,:];points=A.quadrature_points
    probe=points+1e-9*(centers-points);(xe,ye)=axes(B)
    i=np.clip(np.searchsorted(xe,probe[...,0],side='right')-1,0,B.n-1);j=np.clip(np.searchsorted(ye,probe[...,1],side='right')-1,0,B.n-1)
    xi=2*(points[...,0]-xe[i])/(xe[i+1]-xe[i])-1;eta=2*(points[...,1]-ye[j])/(ye[j+1]-ye[j])-1
    rule=np.array([-1.,0.,1.]) if getattr(B,'quadrature','gauss')=='lobatto' else np.polynomial.legendre.leggauss(3)[0]
    ref=np.stack(np.meshgrid(rule,rule,indexing='ij'),axis=-1).reshape(-1,2)
    # Lagrange basis through B's nine quadrature points, evaluated at (xi, eta).
    def lagrange(z,nodes,k):
        others=[m for m in range(3) if m!=k];return np.prod([(z-nodes[m])/(nodes[k]-nodes[m]) for m in others],axis=0)
    basis=np.stack([lagrange(xi,rule,a//3)*lagrange(eta,rule,a%3) for a in range(9)],axis=-1)
    cell=i*B.n+j
    return np.einsum('ega,egas->egs',basis,gamma[cell])


def compare(a_path,b_path,loader=None,cell_fields=('c','eta','T')):
    """Errors of state a against source b at a's sites; volume-weighted, no extrapolation."""
    ma,sa=load(a_path,loader);mb,sb=load(b_path,loader);A,B=ma.mechanics.mesh,mb.mechanics.mesh
    V=np.broadcast_to(ma.V,(A.n**2,));centers=A.coordinates[A.cells].mean(axis=1)
    xb,yb=(( e[:-1]+e[1:])/2 for e in axes(B))
    # Same-mesh comparisons place boundary centers outside B's center grid by roundoff only.
    lo,hi=np.array([xb[0],yb[0]]),np.array([xb[-1],yb[-1]])
    assert np.all(centers>=lo-1e-12) and np.all(centers<=hi+1e-12),'Comparison would extrapolate.'
    centers=np.clip(centers,lo,hi)
    errors={}
    for key in cell_fields:
        target=RegularGridInterpolator((xb,yb),getattr(sb,key).reshape(B.n,B.n))(centers)
        mean=np.sum(V*target)/V.sum()
        errors[key]=float(np.sqrt(np.sum(V*(getattr(sa,key)-target)**2)/np.sum(V*(target-mean)**2)))
    points=A.quadrature_points;w=A.cell_weights
    if 'lobatto' in (getattr(A,'quadrature','gauss'),getattr(B,'quadrature','gauss')):target=cellwise_slip(A,B,sb.gamma)
    else:
        gb=sb.gamma.reshape(B.n,B.n,3,3,2).transpose(0,2,1,3,4).reshape(3*B.n,3*B.n,2)
        target=RegularGridInterpolator(gauss_axes(B),gb)(points)
    errors['slip']=float(np.sqrt(np.sum(w[...,None]*(sa.gamma-target)**2)/np.sum(w[...,None]*target**2)))
    target=RegularGridInterpolator(axes(B),sb.damage.reshape(B.n+1,B.n+1))(points)
    actual=ma.damage_quadrature(sa.damage)
    errors['damage']=float(np.sqrt(np.sum(w*(actual-target)**2)/np.sum(w*target**2)))
    return errors


def front(model,damage,threshold=.4):
    x,y=axes(model.mechanics.mesh);n=model.mechanics.mesh.n
    values=np.array([np.interp(.5,y,row) for row in damage.reshape(n+1,n+1)])
    if values[0]<threshold:return None
    below=np.flatnonzero(values<threshold)
    if not len(below):return 1.
    j=int(below[0]);return float(x[j-1]+(threshold-values[j-1])/(values[j]-values[j-1])*(x[j]-x[j-1]))


def geometry(record,out,suffix='final'):
    m,a=load(out/f'{record["name"]}-initial.npz');_,b=load(out/f'{record["name"]}-{suffix}.npz');mesh=m.mechanics.mesh
    delta=np.sum(mesh.cell_weights*(m.damage_quadrature(b.damage)-m.damage_quadrature(a.damage)),axis=1)
    x,y=mesh.coordinates[mesh.cells].mean(axis=1).T;first,last=front(m,a.damage),front(m,b.damage)
    return dict(initial_front=first,final_front=last,advance=None if first is None or last is None else last-first,
        top_fraction=None if delta.sum()<=0 else float(delta[y>.85].sum()/delta.sum()),
        notch_fraction=None if delta.sum()<=0 else float(delta[(abs(y-.5)<.16)&(x<.75)].sum()/delta.sum()))


def cross(t,y,level):
    i=np.flatnonzero(y>=level)
    if not len(i):return None
    i=int(i[0]);return float(t[i-1]+(level-y[i-1])/(y[i]-y[i-1])*(t[i]-t[i-1]))


def events(record):
    rows=record['rows'];t=np.array([x['time'] for x in rows]);D=np.array([x['damage_integral'] for x in rows])
    return dict(distributed_damage_plus_001=cross(t,D-D[0],.01),tip_max_damage_09=cross(t,np.array([x['max_damage'] for x in rows]),.9),
        first_saturation=next((x['time'] for x in rows[1:] if x['newly_saturated_damage_nodes']),None),
        final_kinetic_energy=rows[-1]['kinetic_energy'],max_midpoint_iterations=max(x['midpoint_iterations'] for x in rows[1:]))


def slip_measures(path):
    m,s=load(path);w=m.mechanics.mesh.cell_weights[...,None]
    return dict(L1=float(np.sum(w*abs(s.gamma))),L2=float(np.sqrt(np.sum(w*s.gamma**2))),max=float(abs(s.gamma).max()))


CHECKPOINTS=(.1,.15,.2,.25,.3)
FAMILIES=dict(graded=(make_model,'graded-notch',(32,48,64)),uniform=(uniform_model,'nodal-tensile',(32,48)))
# Section 45/45.5 saved finals that the uniform rerun must reproduce exactly.
SAVED={32:Path('results/nodal-damage-constrained-2026-09-26'),48:Path('results/nodal-refinement-2026-09-26')}


def label(t):return f't{t:g}'


def reached(record):
    return [t for t in CHECKPOINTS if record['failure'] is None or t<=record['failure']['last_accepted_time']+1e-12]


def reproduction(record,out):
    """Bitwise comparison of a uniform rerun with the saved Section 45 final state."""
    source=SAVED[record['n']]/f'{record["name"]}-final.npz'
    if record['failure'] is not None or not source.exists():return None
    with np.load(source) as a,np.load(out/f'{record["name"]}-final.npz') as b:
        return all(np.array_equal(a[k],b[k]) for k in ('u','v','T','Fp','gamma','c','eta','damage'))


def pair(coarse,fine,out,times):
    path=lambda r,t:out/f'{r["name"]}-{label(t)}.npz'
    rows={label(t):compare(path(coarse,t),path(fine,t)) for t in times}
    if coarse['failure'] is None and fine['failure'] is None:
        rows['final']=compare(out/f'{coarse["name"]}-final.npz',out/f'{fine["name"]}-final.npz')
    return rows


def cross_family(out,a,b,times):
    """Same-cell-count diagnostic; skipped where a's sites lie outside b's cell centers (no extrapolation)."""
    try:return {label(t):compare(out/f'{a["name"]}-{label(t)}.npz',out/f'{b["name"]}-{label(t)}.npz') for t in times}
    except ValueError as error:return dict(skipped=str(error))


def trajectory(family,n,out):
    builder,variant,_=FAMILIES[family]
    return run(n,.00025,variant,out,builder=builder,end_time=.4,checkpoints=CHECKPOINTS,on_failure='record')


def assemble(out,start):
    records={f:{n:json.loads((out/f'{v}-{n}-0.00025.json').read_text()) for n in ns} for f,(_,v,ns) in FAMILIES.items()}
    everyone=[r for f in records.values() for r in f.values()]
    common=[t for t in CHECKPOINTS if all(t in reached(r) for r in everyone)]
    star=common[-1] if common else None
    g,u=records['graded'],records['uniform']
    both=lambda a,b:[t for t in CHECKPOINTS if t in reached(a) and t in reached(b)]
    comparisons=dict(graded_48_64=pair(g[48],g[64],out,both(g[48],g[64])),graded_32_48=pair(g[32],g[48],out,both(g[32],g[48])),
        uniform_32_48=pair(u[32],u[48],out,both(u[32],u[48])),
        graded48_versus_uniform48=cross_family(out,g[48],u[48],both(g[48],u[48])))
    gates=None if star is None else dict(graded_48_64={k:v<.05 for k,v in comparisons['graded_48_64'][label(star)].items()},
        uniform_32_48={k:v<.05 for k,v in comparisons['uniform_32_48'][label(star)].items()})
    final_gates={k:{f:v<.05 for f,v in c['final'].items()} if 'final' in c else 'unevaluated: a trajectory did not reach t=0.4'
        for k,c in comparisons.items() if k!='graded48_versus_uniform48'}
    per_mesh={}
    for family,runs in records.items():
        for n,r in runs.items():
            if family=='graded':hx,hy=(np.diff(e) for e in edges(n));fine,coarse=[.08/hx.min(),.08/hy.min()],[.08/hx.max(),.08/hy.max()]
            else:fine=coarse=[.08*n,.08*n]
            item=dict(cells=n*n,fine_ell_over_h=fine,coarse_ell_over_h=coarse,events=events(r),complete=r['failure'] is None,
                failure=r['failure'],reached=reached(r),solver_seconds=r['solver_seconds'],checks=r['checks'])
            if star is not None:item.update(slip_at_t_star=slip_measures(out/f'{r["name"]}-{label(star)}.npz'),
                geometry_at_t_star=geometry(r,out,label(star)))
            if r['failure'] is None:item.update(slip_final=slip_measures(out/f'{r["name"]}-final.npz'),geometry_final=geometry(r,out))
            if family=='uniform':item['reproduces_saved_final']=reproduction(r,out)
            per_mesh[f'{family}-{n}']=item
    report=dict(scope='Section 46.6 declared checkpoint protocol: static graded Q2 notch family and uniform nodal rerun, same Section 45 physics. Not dynamic adaptivity or a speedup claim.',
        bands=BANDS,checkpoints=CHECKPOINTS,t_star=star,comparisons=comparisons,gates_at_t_star=gates,final_gates=final_gates,
        per_mesh=per_mesh,all_checks=all(all(r['checks'].values()) for r in everyone),study_seconds=perf_counter()-start,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            concurrency='Five single-thread trajectory processes ran concurrently; solver seconds are not a cost comparison.'))
    reference=out/'graded-notch-96-0.00025.json'
    if reference.exists():report['qualification']=qualify(out,g,json.loads(reference.read_text()),comparisons)
    report['tip_timing_estimate']=tip_estimate([g[n] for n in sorted(g)])
    write_json(out/'report.json',report)
    print('t*',star,'gates',gates,'qualification',report.get('qualification',{}).get('by_checkpoint'),flush=True)


def qualify(out,g,r96,comparisons):
    """Section 46.7: graded n64 against the finer graded n96 reference, outside the 46.6 t* rule."""
    times=[t for t in CHECKPOINTS if t in reached(g[64]) and t in reached(r96)]
    rows=pair(g[64],r96,out,times);fine48=comparisons['graded_48_64'];by={}
    for key,errors in rows.items():
        if key=='final' or key not in fine48:continue
        by[key]=dict(below_5_percent=all(v<.05 for v in errors.values()),
            decreasing_slip_damage=all(errors[f]<fine48[key][f] for f in ('slip','damage')))
    hx,hy=(np.diff(e) for e in edges(96))
    return dict(comparison_64_96=rows,by_checkpoint=by,reference=dict(fine_ell_over_h=[.08/hx.min(),.08/hy.min()],events=events(r96),
        complete=r96['failure'] is None,failure=r96['failure'],reached=reached(r96),solver_seconds=r96['solver_seconds'],checks=r96['checks'],
        slip_by_checkpoint={label(t):slip_measures(out/f'{r96["name"]}-{label(t)}.npz') for t in reached(r96)}),
        tip_timing_estimate_48_64_96=tip_estimate([g[48],g[64],r96]),
        concurrency='n96 ran alone as a single-thread process after the five-run batch.')


def tip_estimate(runs):
    """Three-point extrapolation of the tip event in local tip spacing (an estimate, not an asymptotic rate)."""
    from scipy.optimize import brentq
    h=np.array([np.diff(edges(r['n'])[0]).min() for r in runs]);t=np.array([events(r)['tip_max_damage_09'] for r in runs])
    out=dict(tip_spacing=h.tolist(),tip_times=t.tolist())
    if len(runs)>=3 and np.all(np.diff(t[-3:])<0):
        (h1,h2,h3),(t1,t2,t3)=h[-3:],t[-3:]
        f=lambda p:(t1-t2)/(t2-t3)-(h1**p-h2**p)/(h2**p-h3**p)
        try:
            order=brentq(f,.2,6.);C=(t2-t3)/(h2**order-h3**order);limit=t3-C*h3**order
            out.update(observed_order=order,extrapolated_tip_time=limit,finest_relative_timing_error=(t3-limit)/limit)
        except ValueError:out.update(observed_order=None)
    out['caution']='Three-point estimate on the finest three graded meshes; not an established asymptotic rate.'
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',nargs=2,metavar=('FAMILY','N'));args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);start=perf_counter()
    if args.run:trajectory(args.run[0],int(args.run[1]),out)
    else:assemble(out,start)


if __name__=='__main__':main()
