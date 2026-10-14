"""Render spatial junctions, finite-width feedback and frozen encounter kinetics."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def render(folder):
    start=perf_counter();output=folder/'figures';output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

    def save(fig,name):
        for suffix in ('png','svg'):fig.savefig(output/f'{name}.{suffix}',dpi=170,bbox_inches='tight')
        plt.close(fig)

    fig=plt.figure(figsize=(11,8),layout='constrained')
    for index,branch in enumerate(('incident','transmitted','absorbed','reflected'),1):
        ax=fig.add_subplot(2,2,index,projection='3d')
        report=json.loads((folder/branch/'report.json').read_text())
        graph=report['final_graph'];nodes=np.array(graph['nodes']);edges=np.array(graph['edges'])
        yy,zz=np.meshgrid([-1.3,1.3],[-.55,.55])
        ax.plot_surface(np.zeros_like(yy),yy,zz,color='#91b2ad',alpha=.18,shade=False)
        for (i,j),kind,b in zip(edges,graph['kinds'],graph['burgers']):
            color='#b43b3b' if kind else ('#bd8428' if abs(b[2])>1e-10 else '#3564a3')
            ax.plot(nodes[[i,j],0],nodes[[i,j],1],nodes[[i,j],2],color=color,lw=3 if kind else 2)
        ax.scatter(*nodes.T,color='black',s=15)
        for i,node in enumerate(nodes):ax.text(*node,f'  {i}',fontsize=8)
        ax.set(title=branch,xlabel='x',ylabel='y',zlabel='z',xlim=(-3.2,.65),ylim=(-1.3,1.3),zlim=(-.55,.55))
        ax.view_init(elev=24,azim=-60)
        ax.set_box_aspect((3.8,2.6,1.1))
    fig.suptitle('Spatial reaction graphs: red = residual, gold = outgoing, plane = x = 0')
    save(fig,'spatial-branches')

    with np.load(folder/'kinetics/fields.npz') as data:
        report=json.loads((folder/'kinetics/report.json').read_text())
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        for j,name in enumerate(report['state_names']):
            axes[0,0].plot(data['time'],data['probability'][:,j],label=name)
            if j:
                axes[1,0].plot(data['angles'],data['state_energies'][:,j]-data['state_energies'][:,0],label=name)
        axes[0,0].set(xlabel='Frozen-family exposure time',ylabel='Probability',title='Reversible candidate-family kinetics')
        axes[0,0].legend(fontsize=8)
        axes[0,1].plot(data['time'],data['ensemble_free_energy']-data['ensemble_free_energy'][0])
        axes[0,1].set(xlabel='Frozen-family exposure time',ylabel='Change in ensemble free energy',title='Entropy term included')
        axes[1,0].set(xlabel='Misorientation (degrees)',ylabel='Candidate − incident energy',title='Actual spatial energy differences')
        axes[1,0].legend(fontsize=8)
        axes[1,1].plot(data['angles'],data['residual_burgers_norm'],'o-',label='Residual graph edge')
        axes[1,1].plot(data['angles'],.4*np.sin(np.deg2rad(data['angles'])/2),'--',label='2 |b| sin(θ/2)')
        axes[1,1].set(xlabel='Misorientation (degrees)',ylabel='Residual Burgers magnitude',title='Spatial charge conservation')
        axes[1,1].legend(fontsize=8)
        fig.suptitle('One-encounter closure with synthetic embryos and barriers; no material prediction')
        save(fig,'reaction-kinetics')

    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    colors=['#3564a3','#b43b3b','#9c7328']
    with np.load(folder/'incident/fields.npz') as control:
        for branch,color in zip(('transmitted','absorbed','reflected'),colors):
            with np.load(folder/branch/'fields.npz') as data:
                axes[0].plot(data['time'],data['energy']-data['energy'][0],label=branch,color=color)
        with np.load(folder/'transmitted/fields.npz') as data:
            middle=data['phase'].shape[2]//2
            for ax,key,title in zip(axes[1:],('phase','population'),('Phase feedback, z = 0','Population feedback, z = 0')):
                difference=(data[key]-control[key])[:,:,middle]
                maximum=np.max(np.abs(difference))
                im=ax.imshow(difference.T,origin='lower',extent=(-6,6,-6,6),cmap='coolwarm',vmin=-maximum,vmax=maximum)
                ax.set(xlabel='x',ylabel='y',title=title)
                fig.colorbar(im,ax=ax,shrink=.8)
    axes[0].set(xlabel='Time after event',ylabel='Continuous energy change',title='Thermal jump excluded')
    axes[0].legend(fontsize=8)
    fig.suptitle('Energy-driven trajectories; field maps compare transmission with the incident control')
    save(fig,'interface-feedback')
    (output/'plotting.json').write_text(json.dumps({'plotting_and_figure_io_s':perf_counter()-start},indent=2))
    print('Saved three PNG/SVG pairs in',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=ROOT/'results/interface-graphs-2026-09-12')
    render(parser.parse_args().input.resolve())
