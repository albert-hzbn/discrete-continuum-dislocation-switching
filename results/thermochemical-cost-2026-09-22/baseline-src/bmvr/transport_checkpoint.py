"""Persist the complete planar chart/detail or full-field controller state."""
import json
import numpy as np
from .continuum3d import Cube
from .loops3d import Loop
from .transport_metric import PlanarTransport


def save_transport(path,model,loops,details,amplitudes,gamma,mode,time,step,
                   history=0.,budget=.05,max_details=2):
    if mode not in ('chart','field'):raise ValueError('Unknown controller mode.')
    base=model.base
    configuration={'core_coefficient':base.core_coefficient,'field_mobility':base.field_mobility,
                   'mean_strain':base.mean_strain.tolist(),'lam':base.mechanics.elastic.lam,
                   'shear':base.mechanics.elastic.shear,'quadrature_order':base.mechanics.quadrature_order}
    metadata={'length':model.grid.length,'n':model.grid.n,'burgers':model.burgers.tolist(),
              'configuration':configuration,'loops':[loop.to_dict() for loop in loops],
              'mode':mode,'time':time,'step':step,'history':history,'budget':budget,'max_details':max_details}
    detail_array=np.asarray(details) if len(details) else np.zeros((0,)+model.grid.shape)
    np.savez_compressed(path,gamma=gamma,details=detail_array,amplitudes=amplitudes,
                        background=base.background,metadata=np.array(json.dumps(metadata)))


def load_transport(path):
    with np.load(path,allow_pickle=False) as data:
        m=json.loads(str(data['metadata']))
        model=PlanarTransport(Cube(m['length'],m['n']),burgers=m['burgers'],background=data['background'].copy(),**m['configuration'])
        loops=[Loop(**loop) for loop in m['loops']]
        details=[row.copy() for row in data['details']]
        amplitudes=data['amplitudes'].copy();gamma=data['gamma'].copy()
        clock={key:m[key] for key in ('mode','time','step','history','budget','max_details')}
        if clock['mode']=='chart':
            reconstructed=model.chart_state(loops,details,amplitudes,clock['history'])['gamma']
            if not np.array_equal(gamma,reconstructed):raise ValueError('Stored chart and field disagree.')
        return model,loops,details,amplitudes,gamma,clock
