"""First-order accumulated representation error and audited corrections.

The transported estimate omits nonlinear remainders; it is not an error bound.
"""
import io,json
import numpy as np
from .localized_transport import LocalizedTransport


def tangent_rate(model,fields,direction,current=None):
    """Analytic action of the full physical-rate Jacobian, including mobility."""
    v,c,gamma=fields;wv,wc,wgamma=direction;p=model.p;grid=model.grid
    current=model.evaluate(*fields) if current is None else current
    dw=p.phase_misfit*wv/2+p.solute_expansion*wc
    rd=np.fft.ifftn(model.kernel[:,0]*np.fft.fftn(dw),axes=(-3,-2,-1)).real
    rg=model.response_gamma(wgamma);hdw=-rd[0]-rg[0];hgw=-rd[1]-rg[1]
    gvw=p.W*(3*v*v-1)*wv-grid.lap(wv,p.gradient)+2*p.binding*((c-p.reference_population)*wv+v*wc)-p.phase_misfit*hdw/2
    muw=p.susceptibility*wc+2*p.binding*v*wv-p.solute_expansion*hdw
    Kw=2*model.field_mobility*np.sum(current['gamma_response'][2:]*rg[2:],axis=0)
    rates=(-p.phase_mobility*gvw,p.diffusion_mobility*grid.lap(muw),Kw*current['gamma_conjugate']+current['gamma_mobility']*hgw)
    return {'rates':rates,'force_variation':(gvw,muw,-hgw)}


class TrajectoryControl:
    def __init__(self,chart,correct=True,field_limits=(.005,.005,.005),energy_limit=.0005):
        self.chart=chart;self.correct=bool(correct);self.field_limits=tuple(field_limits);self.energy_limit=energy_limit
        if len(field_limits)!=3 or min(field_limits)<=0 or energy_limit<=0:
            raise ValueError('Positive observable trigger levels are required.')
        self.error=tuple(np.zeros_like(f) for f in chart.bases)
        self.disabled=False;self.corrections=0;self.rejections=0;self.tangent_calls=0;self.indicator_calls=0

    def indicators(self,state):
        model=self.chart.model;fields=state['fields'];shadow=tuple(y+e for y,e in zip(fields,self.error))
        reference=model.evaluate(*shadow);self.indicator_calls+=1
        norm=lambda a,b:float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-30))
        alpha_error=model.response_gamma(self.error[2])[2:]
        indicators={'phase':norm(self.error[0],shadow[0]+1),
            'population':norm(self.error[1],shadow[1]-model.p.reference_population),
            'alpha':norm(alpha_error,reference['gamma_response'][2:]),
            'energy':float(abs(reference['energy']-state['energy'])/max(abs(reference['energy']),1e-30))}
        triggered=any(indicators[key]>limit for key,limit in zip(('phase','population','alpha'),self.field_limits)) or indicators['energy']>self.energy_limit
        return indicators,triggered,shadow,reference

    def transfer(self,state,shadow,reference):
        model=self.chart.model;g=model.grid;before=state['fields']
        mass_jump=g.integrate(shadow[1]-before[1]);jump=reference['energy']-state['energy']
        accepted=bool(all(np.isfinite(f).all() for f in shadow) and 0<shadow[1].min() and shadow[1].max()<1
            and abs(mass_jump)<=1e-10 and np.isfinite(reference['energy'])
            and jump<=1e-12*max(1.,abs(state['energy'])))
        old=self.chart
        fields=shadow if accepted else before
        self.chart=LocalizedTransport(model,fields,old.budgets,old.caps,old.reset_at_cap)
        self.chart.time=old.time;self.chart.steps=old.steps
        if accepted:
            self.error=tuple(np.zeros_like(e) for e in self.error);self.corrections+=1
        else:
            self.chart.full=[True]*3;self.disabled=True;self.rejections+=1
        return self.chart.state(project=False),{'accepted':accepted,'time':old.time,'proposed_energy_jump':jump,
            'inventory_jump':mass_jump,'proposed_field_jump_max':[float(np.max(abs(a-b))) for a,b in zip(shadow,before)]}

    def step(self,dt):
        state,events=self.chart.activate();before_error=self.error
        used_coordinates=[self.chart.model.grid.n**3 if self.chart.full[b] else len(self.chart.modes[b]) for b in range(3)]
        tangent=tangent_rate(self.chart.model,state['fields'],before_error,current=state);self.tangent_calls+=1
        defect=tuple(rate-p['rate'] for rate,p in zip(state['rates'],state['projections']))
        following,used,trials=self.chart.step(dt,current=state)
        self.error=tuple(e+used*(j+d) for e,j,d in zip(before_error,tangent['rates'],defect))
        indicators,triggered,shadow,reference=self.indicators(following)
        energy_before_correction=following['energy']
        correction=None
        if triggered and self.correct and not self.disabled:following,correction=self.transfer(following,shadow,reference)
        return following,used,trials,{'basis_events':events,'indicators':indicators,'triggered':triggered,
            'correction':correction,'used_residuals':[p['relative_residual'] for p in state['projections']],
            'used_coordinates':used_coordinates,'energy_before_correction':energy_before_correction}

    def save(self,path):
        buffer=io.BytesIO();self.chart.save(buffer)
        metadata={'correct':self.correct,'field_limits':self.field_limits,'energy_limit':self.energy_limit,
            'disabled':self.disabled,'corrections':self.corrections,'rejections':self.rejections,
            'tangent_calls':self.tangent_calls,'indicator_calls':self.indicator_calls}
        np.savez_compressed(path,chart=np.frombuffer(buffer.getvalue(),dtype=np.uint8),
            **{f'error_{b}':e for b,e in enumerate(self.error)},metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            p=json.loads(str(data['metadata']));chart=LocalizedTransport.load(io.BytesIO(data['chart'].tobytes()))
            obj=cls(chart,p.pop('correct'),p.pop('field_limits'),p.pop('energy_limit'))
            for key,value in p.items():setattr(obj,key,value)
            obj.error=tuple(data[f'error_{b}'].copy() for b in range(3))
            return obj
