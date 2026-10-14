"""Render saved crack propagation, force curves and fixed-width refinement."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import argparse,csv,json
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'results/spatial-fracture-complete-2026-09-22');args=p.parse_args()
    start=perf_counter();folder=args.input;s=json.loads((folder/'summary.json').read_text());out=folder/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    curves=[('mesh-24','n=24'),('mesh-32','n=32'),('mesh-48','n=48')]+([('mesh-64','n=64')] if 'mesh-64' in s['reports'] else [])
    curves += [('load-halfstep','n=32, half load step'),('heterogeneous','n=32, toughness interface')]
    for name,label in curves:
        r=s['reports'][name];h=r['history'][:r['ramp_steps']+1]
        axes[0,0].plot([v['load'] for v in h],[v['reaction'] for v in h],label=label,lw=1.5)
    axes[0,0].set(xlabel='Imposed grip displacement',ylabel='Reaction per thickness',title='Crack loading and post-peak softening');axes[0,0].legend(fontsize=8)
    for n in (16,32,64):
        with np.load(folder/f'profile-{n}.npz') as d:
            x=d['coordinates'][:,0].reshape(n+1,n+1)[:,0];q=d['q'].reshape(n+1,n+1)[:,n//2]
            axes[0,1].plot(x,q,'o-',ms=2,label=f'n={n}')
            if n==64:axes[0,1].plot(x,d['exact'].reshape(n+1,n+1)[:,n//2],color='black',ls=':',label='Finite-domain exact')
    axes[0,1].set(xlabel='x',ylabel='Damage q',title='Planar profile at fixed ell=0.1');axes[0,1].legend(fontsize=8)
    h=s['reports']['mesh-32']['history'];indices=np.arange(len(h))
    for key,label in [('elastic_energy','Elastic'),('fracture_energy','Fracture'),('energy','Total')]:axes[1,0].plot(indices,[r[key] for r in h],label=label)
    axes[1,0].plot(indices,[r['work_trapezoid']+s['reports']['mesh-32']['initial_energy'] for r in h],ls=':',label='Initial energy + trapezoid work')
    if (folder/'work-audit.json').exists():
        audit=json.loads((folder/'work-audit.json').read_text())['reports']['mesh-32']['history']
        axes[1,0].plot(indices,[r['work']+s['reports']['mesh-32']['initial_energy'] for r in audit],ls='--',label='Initial energy + split-path work')
    axes[1,0].axvline(20,color='gray',ls='--',lw=1)
    axes[1,0].set(xlabel='Accepted load index (unloading after 20)',ylabel='Energy per thickness',title='Loading work is distinct from fixed-load descent');axes[1,0].legend(fontsize=8)
    rows=s['comparisons'][:3]+[r for r in s['comparisons'] if r['reference']=='mesh-64'];x=np.arange(len(rows))
    axes[1,1].bar(x-.17,[100*r['peak_reaction_relative'] for r in rows],.32,label='Peak reaction')
    axes[1,1].bar(x+.17,[100*r['fracture_energy_relative'] for r in rows],.32,label='Final fracture energy')
    axes[1,1].axhline(5,color='black',ls=':',label='5% target')
    axes[1,1].set(xticks=x,xticklabels=['24 / 48','32 / 48','Step / 2']+(['48 / 64'] if len(rows)>3 else []),ylabel='Relative difference (%)',title='Mesh and loading-increment sensitivity');axes[1,1].legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(out/f'fracture-verification.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for row,name in enumerate(('mesh-32','heterogeneous')):
        report=s['reports'][name];peak=int(np.argmax([r['reaction'] for r in report['history']]))
        with np.load(folder/name/'fields.npz') as d:
            n=report['n'];x=np.linspace(0,1,n+1)
            for col,(index,label) in enumerate([(0,'Initial equilibrium'),(peak,'Peak reaction'),(report['ramp_steps'],'Maximum imposed displacement')]):
                im=axes[row,col].pcolormesh(x,x,d['damage'][index].reshape(n+1,n+1).T,shading='gouraud',cmap='magma',vmin=0,vmax=1)
                axes[row,col].set(title=f'{label}\ndelta={d["load"][index]:.5f}',xlabel='x',ylabel='y',aspect='equal')
                if row:axes[row,col].axvline(.55,color='cyan',ls=':',lw=1)
        axes[row,0].set_ylabel(('Uniform toughness' if row==0 else 'Diffuse toughness interface')+'\ny')
    fig.colorbar(im,ax=axes,label='Damage q',shrink=.85)
    for ext in ('png','svg'):fig.savefig(out/f'fracture-propagation.{ext}',dpi=160)
    plt.close(fig)
    keys=['run','reference','damage_relative_l2','peak_reaction_relative','fracture_energy_relative','front_absolute','peak_reaction_5pct','fracture_energy_5pct']
    with (folder/'comparisons.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in s['comparisons']:writer.writerow({**{k:v for k,v in r.items() if k!='gates'},**r['gates']})
    (out/'plotting.json').write_text(json.dumps({'plotting_tables_io_s':perf_counter()-start},indent=2)+'\n')
    print('Generated CSV and two PNG/SVG figure pairs.')


if __name__=='__main__':main()
