"""Sequential fresh-process parity, reference qualification and repeated costs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from thermochemical import comparison,write_json

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'benchmarks/thermochemical_cost_worker.py'


def worker(output,source,*arguments):
    output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONPATH=str(source.resolve()))
    start=perf_counter()
    with (output/'stdout.log').open('w') as log:
        process=subprocess.run([sys.executable,str(WORKER),'--output',str(output.resolve()),*arguments],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    elapsed=perf_counter()-start
    if process.returncode:
        raise RuntimeError(f'Worker failed ({process.returncode}); inspect {output}/stdout.log')
    report=json.loads((output/'worker.json').read_text());report['launch_to_exit_seconds']=elapsed
    report['startup_exit_residual_seconds']=elapsed-report['worker_elapsed_seconds']
    write_json(output/'worker.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True);start=perf_counter();source=ROOT/'src'
    state=ROOT/'results/quadratic-mechanics-2026-09-22/q2-coupled-24-0.00025-final.npz'
    old=worker(output/'micro-baseline',output/'baseline-src','--mode','micro','--state',str(state))
    new=worker(output/'micro-optimized',source,'--mode','micro','--state',str(state))
    with np.load(output/'micro-baseline/parity.npz') as a,np.load(output/'micro-optimized/parity.npz') as b:
        discrepancies={key:float(np.max(abs(a[key]-b[key]))) for key in a.files}
    parity=dict(errors=discrepancies,passed=all(v<(1e-12 if k.startswith('state_') else 1e-10) for k,v in discrepancies.items()),
        evaluator_speed_ratio=old['batch_median_seconds']/new['batch_median_seconds'],baseline=old,optimized=new)
    write_json(output/'evaluator.json',parity)
    print('Evaluator parity:',parity['passed'],'median ratio:',parity['evaluator_speed_ratio'],flush=True)
    if not parity['passed']:raise RuntimeError('Evaluator parity failed; full study was not started.')
    reference=worker(output/'reference-q2-48',source,'--mode','trajectory','--n','48','--degree','2')
    print('Reference Q2 n=48 complete:',reference['result']['solver_seconds'],'solver seconds',flush=True)
    runs=[];counts={1:0,2:0}
    for degree in (1,2,2,1,1,2):
        counts[degree]+=1;label=f'q{degree}-repeat-{counts[degree]}';directory=output/label
        report=worker(directory,source,'--mode','trajectory','--n',str(48 if degree==1 else 24),'--degree',str(degree))
        report['label']=label;report['directory']=str(directory)
        errors=comparison(report['result'],reference['result'],directory,output/'reference-q2-48')
        ref=reference['result'];errors['energy']=abs(report['result']['rows'][-1]['total_energy']-ref['rows'][-1]['total_energy'])/ref['energy_scale']
        report['accuracy_errors']=errors;report['accuracy_gates']={key:value<(.01 if key=='energy' else .05) for key,value in errors.items()}
        runs.append(report);write_json(output/f'{label}-qualified.json',report)
        print(label,'solver',report['result']['solver_seconds'],'accuracy',report['accuracy_gates'],flush=True)
    reproducibility={}
    for degree in (1,2):
        group=[x for x in runs if x['result']['degree']==degree];difference=0.
        first=Path(group[0]['directory'])/f'{group[0]["result"]["name"]}-final.npz'
        with np.load(first) as a:
            for item in group[1:]:
                with np.load(Path(item['directory'])/f'{item["result"]["name"]}-final.npz') as b:
                    difference=max(difference,max(float(np.max(abs(a[key]-b[key]))) for key in ('u','v','c','eta','T','Fp','gamma')))
        reproducibility[f'q{degree}']=dict(max_absolute_difference=difference,passed=difference<=1e-12)
    summaries={}
    for degree in (1,2):
        group=[x for x in runs if x['result']['degree']==degree];stats={}
        for key in ('solver_seconds','initialization_seconds','diagnostics_output_seconds'):
            values=[x['result'][key] for x in group];stats[key]=dict(median=float(np.median(values)),min=min(values),max=max(values))
        for key in ('launch_to_exit_seconds','peak_rss_bytes'):
            values=[x[key] for x in group];stats[key]=dict(median=float(np.median(values)),min=min(values),max=max(values))
        stats['evolving_scalars']=group[0]['result']['evolving_stored_scalars'];stats['evolving_bytes']=group[0]['result']['evolving_array_bytes']
        summaries[f'q{degree}']=stats
    qualified=all(all(x['accuracy_gates'].values()) and all(x['result']['checks'].values()) for x in runs) and all(reference['result']['checks'].values()) and all(x['passed'] for x in reproducibility.values())
    speed={key:summaries['q1'][key]['median']/summaries['q2'][key]['median'] for key in ('solver_seconds','launch_to_exit_seconds','peak_rss_bytes')}
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/bmvr').glob('*.py')}
    write_json(output/'report.json',dict(scope='Conventional full-grid Q1/Q2 thermochemistry with identical optimized evaluator; no BMVR adaptive speedup claim.',
        evaluator=parity,reference=reference,runs=runs,reproducibility=reproducibility,summary=summaries,
        accuracy_qualified=qualified,raw_median_ratios=speed,qualified_median_ratios=speed if qualified else None,
        study_seconds=perf_counter()-start,source_sha256=sources))
    print('Complete; accuracy-qualified:',qualified,'ratios:',speed,flush=True)


if __name__=='__main__':main()
