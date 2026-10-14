"""Generate tables and figures from saved, equally optimized transport comparisons."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import argparse,csv,json
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'results/response-cache-optimized-2026-09-13');args=p.parse_args()
    started=perf_counter();s=json.loads((args.input/'summary.json').read_text());rows=s['comparisons']
    out=args.input/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');x=np.arange(len(rows))
    axes[0].bar(x-.17,[r['online_ratio'] for r in rows],.32,label='Online evolution + audits',color='#275d8c')
    axes[0].bar(x+.17,[r['launch_ratio'] for r in rows],.32,label='Launch to exit',color='#b06432')
    axes[0].axhline(1,color='black',ls=':',lw=1)
    axes[0].set(xticks=x,xticklabels=[str(r['n']) for r in rows],xlabel='Grid n per axis',ylabel='Full / cached elapsed time',title='Both paths use optimized scalar mechanics');axes[0].legend(fontsize=8)
    axes[0].set_ylim(0,2.2)
    axes[1].plot([r['affine_online_s'] for r in rows],[100*r['errors']['alpha']['relative_l2'] for r in rows],'o-',color='#337650')
    for r in rows:axes[1].annotate(f"n={r['n']}",(r['affine_online_s'],100*r['errors']['alpha']['relative_l2']),xytext=(5,7),textcoords='offset points')
    axes[1].axhline(1,color='black',ls=':',lw=1,label='1% comparison threshold')
    axes[1].set(xlabel='Cached online time (s)',ylabel='Nye-field discrepancy (%)',title='Error against matched full-field solution');axes[1].margins(x=.2,y=.15);axes[1].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(out/f'cache-cost-accuracy.{ext}',dpi=160)
    plt.close(fig)
    selected=[]
    for n in (13,25,33):
        for kind in ('field','affine'):
            group=sorted([r for r in s['reports'] if r['n']==n and r['kind']==kind],key=lambda r:r['launch_to_exit_s'])
            selected.append(group[1])
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');left=np.zeros(6)
    categories=[('initialization_s','Initialization'),('online_s','Online + audits'),('other_evolution_s','Other evolution'),
        ('full_tensor_validation_s','Tensor validation'),('array_io_s','Array output'),('report_io_s','Report output'),
        ('process_measurement_overhead_s','Process/measurement overhead')]
    for key,label in categories:
        values=np.array([r['timings'][key] for r in selected]);ax.barh(np.arange(6),values,left=left,label=label);left+=values
    ax.set(yticks=np.arange(6),yticklabels=[f"n={r['n']} {r['kind']}" for r in selected],xlabel='Elapsed seconds',title='Disjoint timing categories: actual median-total runs')
    ax.invert_yaxis();ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3)
    for ext in ('png','svg'):fig.savefig(out/f'cache-timing-categories.{ext}',dpi=160)
    plt.close(fig)
    fields=['n','repetitions','field_online_s','affine_online_s','online_ratio','field_launch_s','affine_launch_s','launch_ratio','accuracy_met','field_peak_rss','affine_peak_rss','field_persistent_arrays','affine_persistent_arrays','final_affine_coordinates']
    with (args.input/'performance.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    (out/'plotting.json').write_text(json.dumps({'plotting_tables_and_io_s':perf_counter()-started},indent=2)+'\n')
    print('Generated CSV and two PNG/SVG figure pairs from saved runs.')


if __name__=='__main__':main()
