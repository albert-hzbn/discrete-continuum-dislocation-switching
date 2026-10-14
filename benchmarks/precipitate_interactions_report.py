"""Render saved tensor-precipitate interactions and refinement evidence."""
import argparse,json
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def render(folder):
    start=perf_counter();output=folder/'figures';output.mkdir(exist_ok=True)
    report=json.loads((folder/'report.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#396cad','#b67c29','#28836b']
    def save(fig,name):
        for extension in ('png','svg'):fig.savefig(output/f'{name}.{extension}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    with np.load(folder/'fields.npz') as data:
        layer_drive=np.where((data['phi2']>0)&(data['phi2']<1),data['cross_phase_drive'],np.nan)
        for ax,field,title in zip(axes,(data['phi1']+data['phi2'],data['stress'][0,0],layer_drive),
                                 ('Disjoint finite-width inclusions','Combined elastic stress σxx','Cross drive in second interface')):
            limit=float(np.nanmax(np.abs(field)))
            im=ax.imshow(field,origin='lower',extent=(-20,20,-20,20),cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.contour(data['x'],data['y'],data['phi1']+data['phi2'],levels=[.5],colors='black',linewidths=.7)
            ax.set(xlim=(-12,12),ylim=(-8,8),xlabel='x',ylabel='y',title=title)
            fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('Static tensor-eigenstrain mechanics; periodic cell of side 40, zero mean strain')
    save(fig,'precipitate-pair-fields')
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for d,color in zip((9.,12.,15.),colors):
        rows=[r for r in report['scan'] if r['distance']==d and r['second_tensor_orientation']==0.]
        axes[0,0].plot([np.degrees(r['direction']) for r in rows],[r['interaction_energy'] for r in rows],'o-',color=color,label=f'd = {d}')
    for orientation,color in zip((0.,np.pi/2),colors):
        rows=[r for r in report['scan'] if r['direction']==0. and r['second_tensor_orientation']==orientation]
        axes[0,1].plot([r['distance'] for r in rows],[r['force2'][0] for r in rows],'o-',color=color,label=f'Tensor 2 rotated {np.degrees(orientation):.0f}°')
        rows=[r for r in report['refinement'] if r['second_tensor_orientation']==orientation and r['n']<257]
        label=f'{np.degrees(orientation):.0f}°'
        axes[1,0].loglog([r['h'] for r in rows],[r['relative_energy_error'] for r in rows],'o-',color=color,label=label+' energy')
        axes[1,0].loglog([r['h'] for r in rows],[r['relative_force_error'] for r in rows],'s--',color=color,label=label+' force')
    axes[0,0].set(xlabel='Separation direction (degrees)',ylabel='Interaction energy per unit length',title='Same transformation orientation; direction scan')
    axes[0,1].axhline(0,color='black',lw=.7)
    axes[0,1].set(xlabel='Center separation along x',ylabel='Interaction force on right inclusion',title='Positive: repulsion; negative: attraction')
    axes[1,0].set(xlabel='Grid spacing; physical width = 2',ylabel='Relative error versus N = 257',title='Fixed geometry and tensor orientations')
    rows=report['negative_control']
    residual=[r['isotropic_identity_error'] for r in rows]
    axes[1,1].plot(np.arange(len(rows))+1,residual,'o-',color=colors[2])
    axes[1,1].set(xlabel='Disjoint isotropic control configuration',ylabel='Absolute error in exact inventory identity',title='Dilatational control: no extra directional energy')
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.legend(fontsize=8)
    fig.suptitle('Pair energy and forces: periodic-image effects retained; no infinite-domain claim')
    save(fig,'precipitate-pair-convergence')
    (output/'plotting.json').write_text(json.dumps({'plotting_and_figure_io_s':perf_counter()-start},indent=2)+'\n')
    print('Saved two PNG/SVG figure pairs in',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=ROOT/'results/precipitate-interactions-2026-09-13')
    render(parser.parse_args().input.resolve())
