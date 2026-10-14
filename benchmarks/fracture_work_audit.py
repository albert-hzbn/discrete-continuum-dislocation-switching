"""Independent grip-work balance on the saved loading/relaxation split path."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,hashlib,json,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from bmvr.fracture2d import Fracture2D,FractureParameters


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'results/spatial-fracture-complete-2026-09-22');args=p.parse_args()
    start=perf_counter();reports={}
    for path in sorted(args.input.glob('*/report.json')):
        r=json.loads(path.read_text());rows=[]
        with np.load(path.parent/'fields.npz') as data:
            model=Fracture2D(r['n'],parameters=FractureParameters(residual=r['residual_stiffness']),toughness=data['toughness'])
            u=np.zeros(2*model.node_count);q=model.seed();initial=model.evaluate(u,q)['energy']
            old_energy=initial;old_reaction=0.;old_load=0.;work=release=0.
            top=2*np.flatnonzero(model.coordinates[:,1]==1)+1
            for index,load in enumerate(data['load']):
                fixed,values=model.boundary(float(load));held,held_state,_=model.equilibrate(u,q,fixed,values)
                held_reaction=float(np.sum(held_state['force'][top]));step_work=(old_reaction+held_reaction)*(load-old_load)/2
                u=data['displacement'][index];q=data['damage'][index];accepted=model.evaluate(u,q)
                step_release=held_state['energy']-accepted['energy'];work+=step_work;release+=step_release
                rows.append({'load':float(load),'held_reaction':held_reaction,'step_work':float(step_work),'step_relaxation_release':float(step_release),
                    'step_balance_error':float(step_work-(accepted['energy']-old_energy)-step_release),
                    'work':float(work),'relaxation_release':float(release),'cumulative_balance_error':float(work-(accepted['energy']-initial)-release)})
                old_energy=accepted['energy'];old_reaction=float(np.sum(accepted['force'][top]));old_load=load
        maximum=max(abs(row['cumulative_balance_error']) for row in rows)
        reports[path.parent.name]={'history':rows,'max_cumulative_balance_error':maximum,
            'checks':{'split_grip_work_balance':maximum<1e-10,'nonnegative_relaxation_release':min(row['step_relaxation_release'] for row in rows)>-1e-11}}
    source=[Path(__file__),ROOT/'src/bmvr/fracture2d.py']
    record={'reports':reports,'verification_s':perf_counter()-start,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        'scope':'Reaction-based work on the implemented hold-damage loading path; endpoint-trapezoid diagnostics remain separate.'}
    (args.input/'work-audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({name:{'balance':r['max_cumulative_balance_error'],'checks':r['checks']} for name,r in reports.items()},indent=2))


if __name__=='__main__':main()
