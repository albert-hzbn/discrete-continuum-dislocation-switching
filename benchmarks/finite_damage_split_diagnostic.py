"""Locate retained Section 43 slip error and distributed damage activation."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from bmvr.thermofracture import ThermalFracture2D


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args();out=args.input
    (out/'figures').mkdir(exist_ok=True)
    models=[];states=[]
    for n in (24,32):
        m,s,_=ThermalFracture2D.load(out/f'tensile-split-{n}-0.00025-final.npz');models.append(m);states.append(s)
    def grid(m,n):return m.mechanics.mesh.quadrature_points.reshape(n,n,3,3,2).transpose(0,2,1,3,4).reshape(3*n,3*n,2)
    xa,xb=grid(models[0],24),grid(models[1],32)
    gb=states[1].gamma.reshape(32,32,3,3,2).transpose(0,2,1,3,4).reshape(96,96,2)
    target=RegularGridInterpolator((xb[:,0,0],xb[0,:,1]),gb)(xa).reshape(24,3,24,3,2).transpose(0,2,1,3,4).reshape(576,9,2)
    mesh=models[0].mechanics.mesh;weights=mesh.weights[None,:,None]
    error2=np.sum(weights*(states[0].gamma-target)**2,axis=(1,2));norm2=np.sum(weights*target**2)
    x,y=mesh.coordinates[mesh.cells].mean(axis=1).T
    regions={'notch':(abs(y-.5)<.16)&(x<.75),'bottom_pin':((x-.5)**2+y*y)<.125**2,
             'top_strip':y>.85,'left_strip':x<.15}
    peak=int(error2.argmax());fractions={key:float(error2[mask].sum()/error2.sum()) for key,mask in regions.items()}
    m,s,_=ThermalFracture2D.load(out/'tensile-split-32-0.00025-initial.npz');rate=m.rate(s);low=s.damage<1e-4
    activation={'low_initial_damage_cutoff':1e-4,'low_damage_area_fraction':float(low.mean()),
        'positive_rate_fraction_within_low_damage':float(np.mean(rate['damage'][low]>1e-12)),
        'mean_damage_rate_within_low_damage':float(np.mean(rate['damage'][low])),
        'max_damage_rate_within_low_damage':float(np.max(rate['damage'][low]))}
    norms={}
    for n,m,s in zip((24,32),models,states):
        mesh=m.mechanics.mesh;force=m.mechanics.evaluate(s.u)['force'];pin=m.fixed[-1]
        norms[str(n)]={'weighted_slip_norm':float(np.sqrt(np.sum(mesh.weights[None,:,None]*s.gamma**2))),
                      'pin_horizontal_reaction':float(force[pin]),'net_horizontal_internal_force':float(force[0::2].sum())}
    result={'relative_slip_error':float(np.sqrt(error2.sum()/norm2)),'absolute_weighted_slip_error':float(np.sqrt(error2.sum())),
        'squared_error_fractions':fractions,'peak_error_cell_center':[float(x[peak]),float(y[peak])],
        'initial_damage_activation':activation,'slip_norms_and_pin_reactions':norms,
        'interpretation':'AT2 has no finite damage-initiation threshold at d=0 when positive elastic driving is present. Regional error fractions localize numerical discrepancy; they do not establish its cause. Region masks can overlap.'}
    (out/'failure-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,values,n,title in [(axes[0],np.sqrt(np.sum(models[0].mechanics.mesh.weights[None,:,None]*states[0].gamma**2,axis=(1,2))/models[0].mechanics.mesh.h**2),24,'n24 RMS slip'),
        (axes[1],np.sqrt(np.sum(models[1].mechanics.mesh.weights[None,:,None]*states[1].gamma**2,axis=(1,2))/models[1].mechanics.mesh.h**2),32,'n32 RMS slip'),
        (axes[2],error2/error2.sum(),24,'Fraction of total squared slip error')]:
        im=ax.imshow(values.reshape(n,n).T,origin='lower',extent=[0,1,0,1],cmap='magma')
        ax.set(title=title,xlabel='x',ylabel='y');fig.colorbar(im,ax=ax,shrink=.8)
    for ext in ('png','svg'):fig.savefig(out/f'figures/finite-damage-split-slip-error.{ext}',dpi=170)
    plt.close(fig);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
