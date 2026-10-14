"""Paired-seed grain-statistics mesh sensitivity with preserved baseline data."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
ROOT=Path(__file__).resolve().parents[1]


def collect(folder,baseline):
    rows=[];checks={};distributions={96:[],128:[]};histories={}
    for seed in (11,29,47):
        coarse_path=baseline/f'grains-long-{seed}';fine_path=folder/f'n128-seed{seed}'
        coarse=json.loads((coarse_path/'report.json').read_text());fine=json.loads((fine_path/'report.json').read_text())
        with np.load(coarse_path/'fields.npz') as a,np.load(fine_path/'fields.npz') as b:
            same_seeds=np.array_equal(a['seeds'],b['seeds'])
            same_times=a['time'].shape==b['time'].shape and np.allclose(a['time'],b['time'],atol=1e-9,rtol=0)
            if not same_times:raise ValueError('Physical observation times differ; do not compare rows as matching observations.')
            difference=b['areas']-a['areas'];late=a['time']>=40
            final_mean_coarse=4096/a['grain_count'][-1];final_mean_fine=4096/b['grain_count'][-1]
            row={'seed':seed,'coarse_report':str(coarse_path/'report.json'),'fine_report':str(fine_path/'report.json'),
                 'initial_area_l1_over_domain':float(np.abs(difference[0]).sum()/4096),
                 'max_late_area_l1_over_domain':float(np.max(np.abs(difference[late]).sum(axis=1)/4096)),
                 'final_area_relative_l2':float(np.linalg.norm(difference[-1])/np.linalg.norm(b['areas'][-1])),
                 'final_grain_counts':[int(a['grain_count'][-1]),int(b['grain_count'][-1])],
                 'max_count_difference':int(np.max(np.abs(a['grain_count']-b['grain_count']))),
                 'final_mean_areas':[float(final_mean_coarse),float(final_mean_fine)],
                 'final_mean_area_relative_change':float(abs(final_mean_fine-final_mean_coarse)/final_mean_fine),
                 'growth_slopes':[coarse['metrics']['mean_area_slope'],fine['metrics']['mean_area_slope']],
                 'von_neumann_errors':[coarse['metrics']['von_neumann_normalized_rms_error'],fine['metrics']['von_neumann_normalized_rms_error']],
                 'fine_checks':fine['checks'],'same_physical_seed_coordinates':same_seeds,'same_observation_times':same_times}
            histories[f'time_{seed}']=a['time'];histories[f'area_error_{seed}']=np.abs(difference).sum(axis=1)/4096
            for n,data in ((96,a),(128,b)):
                ids=np.unique(data['final_labels']);areas=data['areas'][-1,ids];areas=areas[areas>1]
                distributions[n].extend((areas/areas.mean()).tolist())
                histories[f'mean_area_{n}_{seed}']=4096/data['grain_count']
            checks[f'seed_coordinates_{seed}']=same_seeds
            checks[f'observation_times_{seed}']=same_times
            checks[f'fixed_steps_{seed}']=coarse['metrics']['steps']==fine['metrics']['steps']==3200
            checks[f'per_seed_final_area_{seed}']=row['final_mean_area_relative_change']<=.15
            rows.append(row)
    ensemble={}
    for key,label in (('final_mean_areas','final_mean_area'),('growth_slopes','growth_slope')):
        values=np.array([r[key] for r in rows]);means=values.mean(axis=0)
        ensemble[label]={'means':means.tolist(),'seed_sd':values.std(axis=0,ddof=1).tolist(),
                         'relative_change':float(abs(means[1]-means[0])/abs(means[1]))}
        checks['ensemble_'+label]=ensemble[label]['relative_change']<=.1
    a=np.sort(distributions[96]);b=np.sort(distributions[128]);points=np.sort(np.r_[a,b])
    distance=float(np.max(np.abs(np.searchsorted(a,points,side='right')/len(a)-np.searchsorted(b,points,side='right')/len(b))))
    distribution={'counts':[len(a),len(b)],'ecdf_distance':distance,'wasserstein_distance':float(wasserstein_distance(a,b)),
                  'interpretation':'Descriptive distances between normalized areas; correlated grains, no independent-sample significance claim.'}
    report={'date':'2026-09-13','rows':rows,'ensemble':ensemble,'distribution':distribution,
            'comparison_checks':{k:bool(v) for k,v in checks.items()},
            'fine_checks_passed':sum(sum(r['fine_checks'].values()) for r in rows),
            'fine_checks_total':sum(len(r['fine_checks']) for r in rows),
            'comparison_checks_passed':sum(bool(v) for v in checks.values()),'comparison_checks_total':len(checks),
            'all_comparison_checks_pass':all(checks.values()),
            'limits':['Two grids and three paired seeds; no asymptotic mesh-convergence order or universal growth-law claim.',
                      'The same finite-width model, dt=0.05, t=160 and physical observation interval are used.']}
    histories['normalized_areas_96']=a;histories['normalized_areas_128']=b
    np.savez_compressed(folder/'comparison.npz',**histories)
    (folder/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:report[key] for key in ('ensemble','distribution','fine_checks_passed','fine_checks_total','comparison_checks_passed','comparison_checks_total')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/grain-mesh-2026-09-13')
    parser.add_argument('--baseline',type=Path,default=ROOT/'results/microstructure-2026-09-13');parser.add_argument('--collect',action='store_true')
    args=parser.parse_args();folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=True)
    if not args.collect:
        for seed in (11,29,47):
            subprocess.run([sys.executable,str(ROOT/'benchmarks/microstructure.py'),'--worker','grains','--n','128','--seed',str(seed),
                            '--end','160','--output',str(folder/f'n128-seed{seed}')],check=True,cwd=ROOT)
    collect(folder,args.baseline.resolve())
