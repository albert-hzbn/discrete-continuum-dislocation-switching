"""Tensor-eigenstrain pair interactions and the dilatational negative control."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bmvr import _resource as resource
from bmvr.grid import Grid
from bmvr.precipitate_interaction import pair_evaluation,transformation_tensor


def run(folder):
    folder.mkdir(parents=True,exist_ok=True);wall=perf_counter();solver=diagnostics=0.
    report={'parameters':{'domain':40.,'inner_radius':2.,'outer_radius':4.,'physical_width':2.,'lam':2.,'shear':1.,
                         'mean_strain':0.,'tensor_eigenvalues':[.12,-.04,-.04]},
            'refinement':[],'scan':[],'negative_control':[],'checks':{},
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                           'platform':platform.platform(),'threads':1}}
    arrays={}
    def evaluate(n,distance,direction,orientation=0.,isotropic=False,fields=False):
        nonlocal solver
        vector=distance*np.array([np.cos(direction),np.sin(direction)])
        grid=Grid.uniform(40,n,periodic=True);start=perf_counter()
        row=pair_evaluation(grid,-vector/2,vector/2,transformation_tensor(dilatational=isotropic),
                            transformation_tensor(orientation,dilatational=isotropic),save_fields=fields)
        solver+=perf_counter()-start
        row.update({'n':n,'h':40/n,'distance':distance,'direction':direction,'second_tensor_orientation':orientation})
        return row
    for orientation in (0.,np.pi/2):
        rows=[evaluate(n,10.,0.,orientation) for n in (65,129,193,257)]
        reference=rows[-1]
        for row in rows[:-1]:
            row['relative_energy_error']=abs(row['interaction_energy']-reference['interaction_energy'])/abs(reference['interaction_energy'])
            row['relative_force_error']=float(np.linalg.norm(np.array(row['force2'])-reference['force2'])/np.linalg.norm(reference['force2']))
        report['refinement'].extend(rows)
        report['checks'][f'energy_refinement_{orientation}']=rows[-2]['relative_energy_error']<.01
        report['checks'][f'force_refinement_{orientation}']=rows[-2]['relative_force_error']<.01
    for distance in (9.,12.,15.):
        for direction in np.linspace(0,np.pi/2,5):
            for orientation in (0.,np.pi/2):
                report['scan'].append(evaluate(193,distance,float(direction),orientation))
            row=evaluate(193,distance,float(direction),isotropic=True)
            expected=2*.08**2*(4*row['overlap']+8*np.prod(row['inventories'])/1600)
            row['isotropic_identity_error']=abs(row['interaction_energy']-expected)
            report['negative_control'].append(row)
    selected=evaluate(193,10.,0.,fields=True);arrays.update(selected.pop('fields'))
    grid=Grid.uniform(40,193,periodic=True);left=np.array([-5.,0.]);right=-left;step=1e-4;forces=[]
    start=perf_counter()
    for axis in np.eye(2):
        plus=pair_evaluation(grid,left,right+step*axis)['interaction_energy']
        minus=pair_evaluation(grid,left,right-step*axis)['interaction_energy']
        forces.append(-(plus-minus)/(2*step))
    solver+=perf_counter()-start
    start=perf_counter()
    selected['finite_difference_force']=forces
    selected['force_difference_norm']=float(np.linalg.norm(np.array(forces)-selected['force2']))
    report['force_validation']=selected
    all_rows=report['refinement']+report['scan']+report['negative_control']
    report['checks']['reciprocity']=max(r['reciprocity_error'] for r in all_rows)<1e-10
    report['checks']['mechanical_equilibrium']=max(r['equilibrium_rms'] for r in all_rows)<1e-10
    report['checks']['isotropic_identity']=max(r['isotropic_identity_error'] for r in report['negative_control'])<1e-10
    report['checks']['negative_control_disjoint']=max(r['overlap'] for r in report['negative_control'])==0
    report['checks']['force_derivative']=selected['force_difference_norm']<1e-8+1e-5*np.linalg.norm(forces)
    signal=[r['interaction_energy'] for r in report['scan'] if r['distance']==12. and r['second_tensor_orientation']==0.]
    uncertainty=max(abs(a['interaction_energy']-b['interaction_energy']) for a,b in zip(report['refinement'][2::4],report['refinement'][3::4]))
    report['orientation_signal']={'energy_range':float(np.ptp(signal)),'refinement_difference_at_distance_10':uncertainty,
        'qualification':'Uncertainty measured at separation 10, not a uniform bound for the entire angular scan.'}
    report['checks']['orientation_signal_resolved']=np.ptp(signal)>10*uncertainty
    report['checks']={k:bool(v) for k,v in report['checks'].items()};report['all_checks_pass']=all(report['checks'].values())
    diagnostics+=perf_counter()-start
    report['limits']=['Prescribed compact profiles and static interactions; no morphology trajectory.',
                      'Homogeneous small-strain plane strain; no out-of-plane shear or stiffness contrast.',
                      'Periodic-image and mean-strain effects retained; not an infinite-domain pair potential.',
                      'Synthetic tensor eigenstrains, no material calibration or novelty claim.']
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_before_io_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    report['timings']={'mechanical_evaluation_and_local_diagnostics_s':solver,'summary_diagnostics_s':diagnostics,
                       'unclassified_pre_io_s':max(0.,perf_counter()-wall-solver-diagnostics),'compilation_s':0.}
    start=perf_counter();np.savez_compressed(folder/'fields.npz',**arrays);report['timings']['array_io_s']=perf_counter()-start
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'checks':report['checks'],'refinement':report['refinement'],'orientation_signal':report['orientation_signal']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/precipitate-interactions-2026-09-13')
    run(parser.parse_args().output.resolve())
