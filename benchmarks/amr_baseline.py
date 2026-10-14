"""Section 61.1: independent quadtree AMR baseline against uniform PF and the focused tensor grid (case 1)."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform
from pathlib import Path
from time import perf_counter
import numpy as np
from bmvr.amr import QuadtreeAMR
from bmvr.grid import Grid
from bmvr.energy import DoubleWell
from bmvr.diagnostics import radius,field_errors
from bmvr.adaptivity import transfer
from bmvr.experiments import ac_run

END,DT,EVERY=12.,.02,50


def amr_run():
    amr=QuadtreeAMR();X,Y=amr.finest_centers();ug=Grid.uniform(amr.length,amr.n)
    start=perf_counter();data=amr.regrid(fine=np.tanh((14.-np.hypot(X,Y))/np.sqrt(2)));init=perf_counter()-start
    solver=regrid=0.;energies=[DoubleWell().value(ug,amr.prolong(data))];cells=[amr.cells()];levels=[]
    for k in range(1,round(END/DT)+1):
        s=perf_counter();data=amr.step(data,DT);solver+=perf_counter()-s
        if k%EVERY==0:s=perf_counter();data=amr.regrid(data);regrid+=perf_counter()-s
        if k%25==0:energies.append(DoubleWell().value(ug,amr.prolong(data)));cells.append(amr.cells())
    levels=np.bincount(amr.levels,minlength=amr.L+1).tolist()
    return dict(solver_s=solver,adaptation_s=regrid,initialization_s=init,final_cells=cells[-1],mean_cells=float(np.mean(cells)),leaf_levels=levels,
        max_energy_increase=float(max(0,np.max(np.diff(energies)))),blocks=len(amr.leaves)),amr.prolong(data),ug


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);runs={'uniform':[],'focused':[],'amr':[]};fields={}
    for rep in range(3):
        for name in ('uniform','focused','amr'):
            if name=='amr':info,field,ug=amr_run()
            else:
                o,arr=ac_run(n=256,adaptive=name=='focused');t=o['timings'];info=dict(solver_s=t['solver_s'],adaptation_s=t['adaptation_s'],initialization_s=t['initialization_s'],
                    final_cells=o['metrics']['final_dofs'],mean_cells=o['metrics']['mean_dofs'],max_energy_increase=o['metrics']['max_recorded_energy_increase'])
                g=Grid(arr['xe'],arr['ye']);ug=Grid.uniform(128.,256);field=transfer(g,ug,arr['field'])
            runs[name].append(info);fields[name]=field
    ug=Grid.uniform(128.,256);ref=fields['uniform']
    accuracy={k:dict(relative_l2=field_errors(ug,(ref+1)/2,ug,(f+1)/2)['relative_l2'],radius_difference=abs(radius(ug,f)-radius(ug,ref))) for k,f in fields.items() if k!='uniform'}
    med=lambda name,key:float(np.median([r[key] for r in runs[name]]))
    cost={k:dict(solver_plus_adaptation_s=float(np.median([r['solver_s']+r['adaptation_s'] for r in runs[k]])),solver_s=med(k,'solver_s'),adaptation_s=med(k,'adaptation_s'),
        mean_cells=runs[k][0]['mean_cells'],final_cells=runs[k][0]['final_cells']) for k in runs}
    gates=dict(A1_accuracy=accuracy['amr']['relative_l2']<=.01 and accuracy['amr']['radius_difference']<=.05 and runs['amr'][0]['max_energy_increase']<=1e-9*abs(DoubleWell().value(ug,ref)),
        A2_prediction_fewer_cells=cost['amr']['mean_cells']<cost['focused']['mean_cells'],A2_prediction_slower=cost['amr']['solver_plus_adaptation_s']>cost['focused']['solver_plus_adaptation_s'])
    speed={f'{a}_over_{b}':cost[a]['solver_plus_adaptation_s']/cost[b]['solver_plus_adaptation_s'] for a,b in (('uniform','amr'),('uniform','focused'),('focused','amr'))}
    report=dict(scope='Section 61.1 quadtree AMR baseline, case 1 physics (R0=14, side 128, finest h=0.5, dt=0.02, t=12); median of three single-thread runs.',
        runs=runs,cost=cost,speedups=speed,accuracy_vs_uniform=accuracy,amr_structure=dict(leaf_levels=runs['amr'][0]['leaf_levels'],blocks=runs['amr'][0]['blocks']),gates=gates,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),threads=1))
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item())+'\n');np.savez_compressed(out/'fields.npz',**fields)
    print(gates,speed,accuracy,{k:(round(v['solver_plus_adaptation_s'],3),v['mean_cells']) for k,v in cost.items()},flush=True)


if __name__=='__main__':main()
