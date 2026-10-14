"""Expose sustained coordinate reduction alongside retained accuracy failures."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import argparse,csv,json
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'results/localized-reset-2026-09-22')
    p.add_argument('--initial',type=Path,default=ROOT/'results/localized-transport-2026-09-22');args=p.parse_args()
    start=perf_counter();s=json.loads((args.input/'summary.json').read_text());old=json.loads((args.initial/'summary.json').read_text())
    out=args.input/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for key,label in [('phase','Phase excess'),('population','Population excess'),('alpha','Nye density')]:
        axes[0,0].plot([13,21,29],[100*r['errors'][key]['relative_l2'] for r in s['matched'][:3]],'o-',label=label)
    axes[0,0].axhline(1,color='black',ls=':',label='1% gate')
    axes[0,0].set(xlabel='Grid n per axis',ylabel='Relative L2 discrepancy (%)',title='Reset policy: population accuracy fails')
    axes[0,0].legend(fontsize=8)
    for study,label in [(old,'Promote at cap'),(s,'Reset at cap')]:
        axes[0,1].plot([13,21,29],[100*r['energy_relative'] for r in study['matched'][:3]],'o-',label=label)
    axes[0,1].axhline(.1,color='black',ls=':',label='0.1% gate')
    axes[0,1].set(xlabel='Grid n per axis',ylabel='Relative energy discrepancy (%)',title='Energy descent does not imply accuracy')
    axes[0,1].legend(fontsize=8)
    with np.load(args.input/'localized-21/fields.npz') as data:
        for i,label in enumerate(('Phase','Population','Gamma')):
            axes[1,0].step(data['time'],data['coordinates'][:,i],where='post',label=label)
        axes[1,0].axhline(32,color='black',ls=':',label='Per-block mode cap')
        axes[1,0].set(xlabel='Time',ylabel='Evolving coordinates per block',title='n=21: resets keep all fields reduced')
        axes[1,0].legend(fontsize=8)
    x=np.arange(3)
    for i,(study,kind,label) in enumerate([(s,'field','Full'),(old,'localized','Promote at cap'),(s,'localized','Reset at cap')]):
        axes[1,1].bar(x+(i-1)*.25,[study['reports'][f'{kind}-{n}']['timings']['online_s'] for n in (13,21,29)],.23,label=label)
    axes[1,1].set(xticks=x,xticklabels=['13','21','29'],xlabel='Grid n per axis',ylabel='Online seconds (single run)',title='Dense audits and basis work remain expensive')
    axes[1,1].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(out/f'localized-accuracy-cost.{ext}',dpi=160)
    plt.close(fig)
    with np.load(args.input/'localized-21/fields.npz') as data,np.load(args.input/'localized-21/restart.npz') as checkpoint:
        fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
        values=[(data['phase']+1)/2,data['population']-.2,np.sqrt(np.sum(data['alpha']**2,axis=(0,1)))]
        for i,label in enumerate(('Structural phase coordinate','Population excess','Nye density norm')):
            im=axes[0,i].imshow(values[i][:,:,10].T,origin='lower',extent=(-6,6,-6,6),cmap='viridis')
            axes[0,i].set(title=label,xlabel='x',ylabel='y');fig.colorbar(im,ax=axes[0,i],shrink=.75)
            q=checkpoint[f'coefficients_{i}'];modes=checkpoint[f'modes_{i}'];contribution=np.abs(q)*np.linalg.norm(modes.reshape(len(q),-1),axis=1)
            index=int(np.argmax(contribution));correction=q[index]*modes[index]
            # Select the slice through the largest magnitude of this localized correction.
            z=int(np.unravel_index(np.argmax(abs(correction)),correction.shape)[2]);limit=float(np.max(abs(correction[:,:,z])))
            im=axes[1,i].imshow(correction[:,:,z].T,origin='lower',extent=(-6,6,-6,6),cmap='coolwarm',vmin=-limit,vmax=limit)
            axes[1,i].set(title=f'{("v","c","gamma")[i]}: largest current mode contribution\nz={((z+.5)*12/21-6):.2f}',xlabel='x',ylabel='y')
            fig.colorbar(im,ax=axes[1,i],shrink=.75)
        fig.suptitle('n=21, t=1: fields and localized correction supports')
        for ext in ('png','svg'):fig.savefig(out/f'localized-fields-modes.{ext}',dpi=160)
        plt.close(fig)
    keys=['policy','run','phase_error_pct','population_error_pct','alpha_error_pct','energy_error_pct','all_gates_pass','coordinates','full_blocks','online_s','persistent_mib']
    with (args.input/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for study,policy in ((old,'promote'),(s,'reset')):
            for row in study['matched']:
                r=study['reports'][row['run']]
                writer.writerow({'policy':policy,'run':row['run'],'phase_error_pct':100*row['errors']['phase']['relative_l2'],
                    'population_error_pct':100*row['errors']['population']['relative_l2'],'alpha_error_pct':100*row['errors']['alpha']['relative_l2'],
                    'energy_error_pct':100*row['energy_relative'],'all_gates_pass':all(row['gates'].values()),'coordinates':sum(r['final_coordinates']),
                    'full_blocks':str(r['final_full_blocks']),'online_s':r['timings']['online_s'],'persistent_mib':r['persistent_array_bytes']/2**20})
    (out/'plotting.json').write_text(json.dumps({'plotting_tables_io_s':perf_counter()-start},indent=2)+'\n')
    print('Generated CSV and two PNG/SVG figure pairs, including failed accuracy gates.')


if __name__=='__main__':main()
