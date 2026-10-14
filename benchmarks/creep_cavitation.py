"""Section 97: creep cavitation and intergranular cracking; coupled (A), fracture-only (B), fixed-stress cavitation (C).

python benchmarks/creep_cavitation.py run A 0.016 64     # one job
python benchmarks/creep_cavitation.py analyze            # gates K1-K6, report.json
"""
import json, platform, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from bmvr.creep_cavitation import CreepCavitation  # noqa: E402

OUT = Path(__file__).resolve().parents[1]/'results'/'creep-cavitation-2026-09-30'
STRESSES = (0.008, 0.012, 0.016, 0.020)
MODES = {'A': 'coupled', 'B': 'fracture', 'C': 'uncoupled'}
T_MAX, DT_MECH = 20., 0.25


def run(model, S, n):
    OUT.mkdir(parents=True, exist_ok=True)
    m = CreepCavitation(n, MODES[model], S)
    frames = []

    def grab(sim):
        qc = sim.q[sim.mech.cells].mean(axis=1).reshape(n, n) if sim.mech is not None else 0*sim.p
        frames.append(dict(t=sim.t, p=sim.p.copy(), q=qc, c=sim.c.copy(), sigma=sim.sigma_n.copy()))

    t0 = time.process_time()
    result = m.run(T_MAX, DT_MECH, frames=np.arange(0, T_MAX + 1e-9, 0.5), callback=grab)
    grab(m)
    cpu = time.process_time() - t0
    tag = f'{model}_{S:.3f}_{n}'
    np.savez_compressed(OUT/f'{tag}.npz', t=[f['t'] for f in frames], **{k: np.array([f[k] for f in frames]) for k in ('p', 'q', 'c', 'sigma')})
    ledger = max(h['ledger'] for h in result['history'])
    summary = dict(model=model, S=S, n=n, failed=result['failed'], t_f=result['t_f'], steps=result['steps'], cpu_s=cpu,
                   ledger_max=ledger, max_damage=max(h['max_damage'] for h in result['history']), history=result['history'])
    (OUT/f'{tag}.json').write_text(json.dumps(summary, indent=1))
    print(tag, 'failed', result['failed'], result['t_f'], 'cpu', round(cpu), 'ledger', ledger)


def analyze():
    load = lambda m, S, n=64: json.loads((OUT/f'{m}_{S:.3f}_{n}.json').read_text())
    A = {S: load('A', S) for S in STRESSES}; C = {S: load('C', S) for S in STRESSES}; B = load('B', 0.020)
    tA = np.array([A[S]['t_f'] or np.inf for S in STRESSES]); tC = np.array([C[S]['t_f'] or np.inf for S in STRESSES])
    ledger = max(r['ledger_max'] for r in [*A.values(), *C.values()])
    slope = float(np.polyfit(np.log(STRESSES), np.log(tA), 1)[0]) if np.all(np.isfinite(tA)) else None
    slopeC = float(np.polyfit(np.log(STRESSES), np.log(tC), 1)[0]) if np.all(np.isfinite(tC)) else None
    fine = load('A', 0.016, 96)
    k6 = abs(fine['t_f'] - A[0.016]['t_f'])/A[0.016]['t_f'] if fine['t_f'] else None
    gates = {
        'K1 ledger <= 1e-10': dict(value=ledger, passed=ledger <= 1e-10),
        'K2 fracture only forms no crack at S=0.02 (max q < 0.5)': dict(value=B['max_damage'], passed=B['max_damage'] < .5 and not B['failed']),
        'K3 coupled fails at every S within t_max': dict(value=tA.tolist(), passed=bool(np.all(np.isfinite(tA)))),
        'K4 t_f(A) < t_c(C) at every S': dict(value=(tA/tC).tolist(), passed=bool(np.all(tA < tC))),
        'K5 monotone, log-log slope in [-3, -0.8]': dict(value=slope, passed=bool(np.all(np.diff(tA) < 0)) and slope is not None and -3 <= slope <= -.8),
        'K6 n=96 vs n=64 within 15% at S=0.016': dict(value=k6, passed=k6 is not None and k6 <= .15),
    }
    report = dict(section=97, stresses=STRESSES, t_f_A=tA.tolist(), t_c_C=tC.tolist(), slope_A=slope, slope_C=slopeC,
                  t_f_A_n96=fine['t_f'], cpu_s={f'{k}_{S}': r['cpu_s'] for k, d in (('A', A), ('C', C)) for S, r in d.items()},
                  cpu_B=B['cpu_s'], cpu_A96=fine['cpu_s'], gates=gates,
                  machine=dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version()),
                  parameters='theory Section 97 with the pre-run amendments', t_max=T_MAX, dt_mech=DT_MECH)
    (OUT/'report.json').write_text(json.dumps(report, indent=1))
    for k, v in gates.items():
        print('PASS' if v['passed'] else 'FAIL', k, v['value'])
    print('t_f A', tA, 't_c C', tC, 'slopes', slope, slopeC)


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
    else:
        analyze()
