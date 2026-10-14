"""Section 98: 3D creep cavitation and intergranular cracking in copper at 773 K (GPU).

python benchmarks/creep_cavitation3d.py run A 30 96 [gamma_factor]   # one job
python benchmarks/creep_cavitation3d.py rajashby                      # gate L7
python benchmarks/creep_cavitation3d.py analyze                       # gates L1-L8, report.json
"""
from dataclasses import replace
import json, os, platform, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from bmvr.creep3d import Copper773, Creep3D, Params3D, to_host, GPU  # noqa: E402

OUT = Path(__file__).resolve().parents[1]/'results'/'creep-cavitation-3d-2026-09-30'
STRESSES = (20., 30., 45., 60.)
T_MAX = 25.


def tag(model, S, n, gf):
    return f'{model}_{S:g}_{n}' + ('' if gf == 1 else f'_g{gf:g}')


def run(model, S, n, gf=1.):
    OUT.mkdir(parents=True, exist_ok=True)
    mat = replace(Copper773(), gamma_s=1.7*gf)
    sim = Creep3D(mat, S, model, Params3D(n=n))
    frames = []

    def grab(s):
        f = dict(t=s.t, area=s.area_fraction(), sigma=s.sigma_n.astype(np.float32), c=s.c.astype(np.float32),
                 R=s.R.copy())
        if s.mech_on:
            z = s.n//2
            d = 1 - (1 - s.q[:, :, z - 1:z + 1])*(1 - s.p3[:, :, z - 1:z + 1])
            j = int(round(s.xc[0, 1]*s.n - .5)) if len(s.xc) else s.n//2
            f.update(dplane=to_host(d.max(axis=2)).astype(np.float32),
                     q_xz=to_host(s.q[:, j, :]).astype(np.float32), p_xz=to_host(s.p3[:, j, :]).astype(np.float32),
                     szz_xz=to_host(s.sig[2][:, j, :]).astype(np.float32), slice_y=j)
        frames.append(f)
        if os.environ.get('DUMP3D') and s.mech_on:        # visualization only: keep the last two 3D states
            vol.append(dict(t=s.t, q=to_host(s.q).astype(np.float16), p=to_host(s.p3).astype(np.float16),
                            szz=to_host(s.sig[2]).astype(np.float32)))
            del vol[:-2]

    vol = []
    ckpt = OUT/f'{tag(model, S, n, gf)}.ckpt'          # checkpoint after every frame; a crashed run resumes from it
    wall0, resume = 0., False
    if ckpt.exists():
        import pickle
        sim = Creep3D.restore(ckpt)
        extra = pickle.loads((OUT/f'{ckpt.name}.driver').read_bytes())
        frames, vol, wall0, resume = extra['frames'], extra['vol'], extra['wall'], True
        print('resumed at t =', sim.t, flush=True)

    def grab_ckpt(s):
        grab(s)
        import pickle
        s.checkpoint(ckpt)
        (OUT/f'{ckpt.name}.driver').write_bytes(pickle.dumps(dict(frames=frames, vol=vol, wall=wall0 + time.perf_counter() - t0)))

    t0 = time.perf_counter()
    pending = [f for f in np.arange(1., T_MAX + 1e-9, 1.) if f > sim.t + 1e-9]
    res = sim.run(T_MAX, frames=pending, callback=grab_ckpt, resume=resume)
    grab(sim)
    wall = wall0 + time.perf_counter() - t0
    name = tag(model, S, n, gf)
    np.savez_compressed(OUT/f'{name}.npz', **{k: np.array([f[k] for f in frames if k in f]) for k in frames[-1]},
                        xc=sim.xc, t_unit=mat.t_unit)
    if vol:
        np.savez_compressed(OUT/f'{name}_3d.npz', t=[v['t'] for v in vol], **{k: np.array([v[k] for v in vol]) for k in ('q', 'p', 'szz')})
    summary = dict(model=model, S_MPa=S, n=n, gamma_factor=gf, failed=res['failed'], t_f=res['t_f'],
                   t_f_hours=res['t_f']*mat.t_unit/3600 if res['t_f'] else None, wall_s=wall, gpu=GPU,
                   ledger_max=max(h['ledger'] for h in sim.history), history=sim.history)
    (OUT/f'{name}.json').write_text(json.dumps(summary, indent=1, default=float))
    print(name, 'failed', res['failed'], res['t_f'], 'hours', summary['t_f_hours'], 'wall', round(wall), 'ledger', summary['ledger_max'], flush=True)
    for f in (ckpt, OUT/f'{ckpt.name}.driver'):
        f.unlink(missing_ok=True)


def rajashby():
    """Gates L7a (calibration case within 5%) and L7b (transfer cases within 20%)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from creep_calibration3d import ratio
    k = Params3D().k_ex
    cal = ratio(Params3D().l_s, .04, 30., k_ex=k)[0]
    transfer = {f'{R0}_{S:g}': ratio(Params3D().l_s, R0, S, k_ex=k)[0] for R0, S in ((.03, 30.), (.08, 30.), (.04, 60.), (.04, 20.))}
    out = dict(k_ex=k, calibration_ratio=cal, transfer_ratios=transfer, passed_a=abs(cal - 1) <= .05,
               passed_b=all(abs(r - 1) <= .2 for r in transfer.values()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'rajashby.json').write_text(json.dumps(out, indent=1, default=float))
    print(out)


def analyze():
    N0, NF = 48, 64                                   # post hoc amendment (theory Section 98, v2 status)
    load = lambda m, S, n=N0, gf=1.: json.loads((OUT/f'{tag(m, S, n, gf)}.json').read_text())
    A = [load('A', S) for S in STRESSES]; C = [load('C', S) for S in STRESSES]
    A0 = [load('A0', S) if (OUT/f'{tag("A0", S, N0, 1.)}.json').exists() else dict(t_f=None, ledger_max=0., wall_s=0., model='A0', S_MPa=S, n=N0, gamma_factor=1.) for S in STRESSES]
    B = load('B', 60.); fine = load('A', 30., NF); gl, gh = load('A', 30., N0, .8), load('A', 30., N0, 1.2)
    ra = json.loads((OUT/'rajashby.json').read_text())
    tA = np.array([r['t_f'] or np.inf for r in A]); tC = np.array([r['t_f'] or np.inf for r in C])
    tA0 = np.array([r['t_f'] or np.inf for r in A0])
    ledger = max(r['ledger_max'] for r in A + C + A0 + [fine, gl, gh])
    slope = float(np.polyfit(np.log(STRESSES), np.log(tA), 1)[0]) if np.all(np.isfinite(tA)) else None
    k6 = abs(fine['t_f'] - A[1]['t_f'])/A[1]['t_f'] if fine['t_f'] and A[1]['t_f'] else None
    k8 = max(abs(r['t_f'] - A[1]['t_f'])/A[1]['t_f'] for r in (gl, gh)) if gl['t_f'] and gh['t_f'] and A[1]['t_f'] else None
    bq = max(h.get('max_q', 0) for h in B['history'])
    gates = {
        'L1 ledger <= 1e-10': dict(value=ledger, passed=ledger <= 1e-10),
        'L2 fracture only at 60 MPa forms no crack (max q < 0.5)': dict(value=bq, passed=bq < .5 and not B['failed']),
        'L3 coupled fails at every S within t_max': dict(value=tA.tolist(), passed=bool(np.all(np.isfinite(tA)))),
        'L4 t_f(A) < t_f(C) at every S': dict(value=(tA/tC).tolist(), passed=bool(np.all(tA < tC))),
        'L5 monotone, log-log slope in [-3, -0.8]': dict(value=slope, passed=bool(np.all(np.diff(tA) < 0)) and slope is not None and -3 <= slope <= -.8),
        'L6 n=64 vs n=48 within 15% at 30 MPa (post hoc)': dict(value=k6, passed=k6 is not None and k6 <= .15),
        'L7a calibration case within 5% of Raj-Ashby': dict(value=ra['calibration_ratio'], passed=ra['passed_a']),
        'L7b transfer cases within 20% of Raj-Ashby': dict(value=ra['transfer_ratios'], passed=ra['passed_b']),
        'L8 gamma_s +-20% changes t_f(A) at 30 MPa by <= 15%': dict(value=k8, passed=k8 is not None and k8 <= .15),
    }
    tu = Copper773().t_unit/3600
    report = dict(section=98, stresses_MPa=STRESSES, t_unit_hours=tu, t_f_A=tA.tolist(), t_f_C=tC.tolist(), t_f_A0=tA0.tolist(),
                  hours_A=(tA*tu).tolist(), hours_C=(tC*tu).tolist(), hours_A0=(tA0*tu).tolist(), slope_A=slope,
                  slope_C=float(np.polyfit(np.log(STRESSES), np.log(tC), 1)[0]) if np.all(np.isfinite(tC)) else None,
                  t_f_A_fine=fine['t_f'], n=N0, n_fine=NF, t_f_gamma=[gl['t_f'], gh['t_f']], rajashby=ra,
                  area_at_failure_A=[r['history'][-1]['area_fraction'] for r in A],
                  wall_s={tag(r['model'], r['S_MPa'], r['n'], r['gamma_factor']): r['wall_s'] for r in A + C + A0 + [B, fine, gl, gh]},
                  gates=gates, machine=dict(platform=platform.platform(), processor=platform.processor(),
                                            python=platform.python_version(), gpu='NVIDIA RTX 2000 Ada (CuPy)' if GPU else None))
    (OUT/'report.json').write_text(json.dumps(report, indent=1, default=float))
    for k, v in gates.items():
        print('PASS' if v['passed'] else 'FAIL', k, v['value'])
    print('hours A', report['hours_A'], 'C', report['hours_C'], 'A0', report['hours_A0'], 'slope', slope)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'run':
        run(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]) if len(sys.argv) > 5 else 1.)
    elif cmd == 'rajashby':
        rajashby()
    else:
        analyze()
