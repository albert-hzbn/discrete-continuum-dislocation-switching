"""Section 99: gates M1-M4 for the 3D ExaDiS allocator test (results/allocator-3d-2026-09-30).

python benchmarks/allocator3d_report.py calibrate     # choose alpha on seed 1
python benchmarks/allocator3d_report.py               # gates and report.json
SECTION=100 python benchmarks/allocator3d_report.py [calibrate]   # Section 100 (sources on the wall faces)
"""
import json, os, sys
from pathlib import Path

import numpy as np

SECTION = int(os.environ.get('SECTION', 99))
OUT = Path(__file__).resolve().parents[1]/'results'/('allocator-3d-2026-09-30' if SECTION == 99 else 'allocator-3d-wall-2026-10-01')
ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5) if SECTION == 99 else (0.1, 0.2, 0.3, 0.5, 1.0)
G = 'M' if SECTION == 99 else 'N'                   # gate prefix
T_END = 1e-9


def load(tag):
    return json.loads((OUT/f'{tag}.json').read_text())


def curve(run, col=2):
    log = np.array(run['log'], dtype=float)
    t = np.linspace(T_END/30, T_END, 30)
    return np.interp(t, log[:, 0], log[:, col])


def err(run, ref, col=2):
    a, b = curve(run, col), curve(ref, col)
    return float(np.linalg.norm(a - b)/np.linalg.norm(b))


def calibrate():
    ref = load('D_s1')
    errors = {a: err(load(f'H_s1_a{a:g}'), ref) for a in ALPHAS}
    best = min(errors, key=errors.get)
    out = dict(errors=errors, alpha=best)
    (OUT/'calibration.json').write_text(json.dumps(out, indent=1))
    print(out)
    return best


def report():
    cal = json.loads((OUT/'calibration.json').read_text()); a = cal['alpha']
    rows = {}
    for s in (1, 2, 3):
        D, Z = load(f'D_s{s}'), load(f'0_s{s}')
        H = load(f'H_s{s}_a{a:g}')
        rate = lambda r: r['wall_s']/(r['time_s']*1e9)
        rows[s] = dict(e_H=err(H, D), e_0=err(Z, D), e_H_total=err(H, D, 1), cost_D=rate(D), cost_H=rate(H), cost_0=rate(Z),
                       speedup=rate(D)/rate(H), segs_D=D['segments_initial'], segs_H=H['segments_initial'],
                       Ech=dict(D=D['Ep_channel'], H=H['Ep_channel'], zero=Z['Ep_channel']),
                       Ewall_D=D['Ep_wall'], Etot=dict(D=D['Ep'][2], H=H['Ep'][2]),
                       ledger_burgers=H['ledger_burgers_max'], ledger_distortion=H['ledger_distortion_rel'],
                       longrange=H['longrange_rel'], tau_f_max=H['tau_f_max'], steps=dict(D=D['steps'], H=H['steps']))
    test = [rows[2], rows[3]]
    gates = {
        'M1 conversion exact (Burgers 0, distortion content <= 1e-12)': dict(
            value=[max(r['ledger_burgers'] for r in rows.values()), max(r['ledger_distortion'] for r in rows.values())],
            passed=all(r['ledger_burgers'] == 0 and r['ledger_distortion'] <= 1e-12 for r in rows.values())),
        'M1b long-range stress of removed loops <= 10% of applied': dict(
            value=max(r['longrange'] for r in rows.values()), passed=all(r['longrange'] <= .1 for r in rows.values())),
        'M2 channel strain error of H <= 15% on seeds 2, 3': dict(value=[r['e_H'] for r in test], passed=all(r['e_H'] <= .15 for r in test)),
        'M3 no-wall error >= 3x H error on seeds 2, 3': dict(value=[r['e_0']/max(r['e_H'], 1e-12) for r in test],
                                                             passed=all(r['e_0'] >= 3*r['e_H'] for r in test)),
        'M4 H cost <= D/5 per simulated ns': dict(value=[r['speedup'] for r in rows.values()], passed=all(r['speedup'] >= 5 for r in rows.values())),
    }
    gates = {G + k[1:]: v for k, v in gates.items()}
    report = dict(section=SECTION, alpha=a, calibration=cal, rows=rows, gates=gates,
                  code='ExaDiS commit 351607f (OpenMP, 16 threads, WSL Ubuntu)', T_end=T_END)
    (OUT/'report.json').write_text(json.dumps(report, indent=1, default=float))
    for k, v in gates.items():
        print('PASS' if v['passed'] else 'FAIL', k, v['value'])
    for s, r in rows.items():
        print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == '__main__':
    calibrate() if len(sys.argv) > 1 and sys.argv[1] == 'calibrate' else report()
