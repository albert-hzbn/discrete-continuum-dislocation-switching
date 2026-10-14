"""Saved accumulated-error accuracy, correction history and measured costs."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import argparse,csv,json
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'results/trajectory-control-2026-09-22');args=p.parse_args()
    start=perf_counter();folder=args.input;s=json.loads((folder/'summary.json').read_text());out=folder/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for kind,label in [('monitor','Monitor only'),('corrected','Audited correction')]:
        rows=[next(r for r in s['matched'] if r['run']==f'{kind}-{n}') for n in (13,21,29)]
        axes[0,0].semilogy([13,21,29],[100*r['errors']['population']['relative_l2'] for r in rows],'o-',label=label)
        axes[0,1].semilogy([13,21,29],[100*r['energy_relative'] for r in rows],'o-',label=label)
    axes[0,0].axhline(1,color='black',ls=':',label='1% gate');axes[0,1].axhline(.1,color='black',ls=':',label='0.1% gate')
    axes[0,0].set(xlabel='Grid n per axis',ylabel='Population-excess L2 error (%)',title='Same full physical law and final tolerance')
    axes[0,1].set(xlabel='Grid n per axis',ylabel='Relative energy error (%)',title='Accumulated-error correction')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    with np.load(folder/'corrected-21/fields.npz') as data:
        axes[1,0].plot(data['time'][1:],np.max(data['indicators']/np.array([.005,.005,.005,.0005]),axis=1),label='Before any correction')
        axes[1,0].axhline(1,color='black',ls=':',label='Correction trigger')
        for c in s['reports']['corrected-21']['corrections']:axes[1,0].axvline(c['time'],color='#b86c32',alpha=.3,lw=1)
    axes[1,0].set(xlabel='Time',ylabel='Largest indicator / its trigger',title='n=21: correction times shown vertically');axes[1,0].legend(fontsize=8)
    x=np.arange(3)
    for i,(kind,label) in enumerate([('field','Full'),('monitor','Monitor only'),('corrected','Corrected')]):
        axes[1,1].bar(x+(i-1)*.25,[s['reports'][f'{kind}-{n}']['timings']['online_s'] for n in (13,21,29)],.23,label=label)
    axes[1,1].set(xticks=x,xticklabels=['13','21','29'],xlabel='Grid n per axis',ylabel='Online seconds (single run)',title='All tangent, audit and rebuild costs included');axes[1,1].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(out/f'trajectory-accuracy-cost.{ext}',dpi=160)
    plt.close(fig)
    with np.load(folder/'field-21/fields.npz') as full,np.load(folder/'monitor-21/fields.npz') as monitor,np.load(folder/'corrected-21/fields.npz') as corrected:
        actual=full['population']-monitor['population'];predicted=monitor['error'][1];remaining=full['population']-corrected['population']
        z=int(np.unravel_index(np.argmax(abs(actual)),actual.shape)[2]);fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        common=max(float(np.max(abs(actual[:,:,z]))),float(np.max(abs(predicted[:,:,z]))))
        for i,(values,title) in enumerate([(actual,'Actual missing population'),(predicted,'First-order estimate'),(remaining,'Remaining error after corrections')]):
            limit=common if i<2 else float(np.max(abs(values[:,:,z])))
            im=axes[i].imshow(values[:,:,z].T,origin='lower',extent=(-6,6,-6,6),cmap='coolwarm',vmin=-limit,vmax=limit)
            axes[i].set(title=title,xlabel='x',ylabel='y');fig.colorbar(im,ax=axes[i],shrink=.8)
        fig.suptitle(f'n=21, t=1, z={((z+.5)*12/21-6):.2f}; last panel has its own smaller scale')
        for ext in ('png','svg'):fig.savefig(out/f'trajectory-error-fields.{ext}',dpi=160)
        plt.close(fig)
    keys=['run','phase_error_pct','population_error_pct','alpha_error_pct','energy_error_pct','all_gates_pass','corrections','maximum_physical_coordinates','auxiliary_error_coordinates','maximum_total_coordinates','online_s','persistent_mib']
    with (folder/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in s['matched']:
            r=s['reports'][row['run']]
            writer.writerow({'run':row['run'],'phase_error_pct':100*row['errors']['phase']['relative_l2'],
                'population_error_pct':100*row['errors']['population']['relative_l2'],'alpha_error_pct':100*row['errors']['alpha']['relative_l2'],
                'energy_error_pct':100*row['energy_relative'],'all_gates_pass':all(row['gates'].values()),'corrections':r['correction_count'],
                'maximum_physical_coordinates':r['maximum_physical_coordinates'],'auxiliary_error_coordinates':r['auxiliary_error_coordinates'],
                'maximum_total_coordinates':r['maximum_total_evolving_coordinates'],'online_s':r['timings']['online_s'],'persistent_mib':r['persistent_array_bytes']/2**20})
    (out/'plotting.json').write_text(json.dumps({'plotting_tables_io_s':perf_counter()-start},indent=2)+'\n')
    print('Generated comparison CSV and two PNG/SVG figure pairs.')


if __name__=='__main__':main()
