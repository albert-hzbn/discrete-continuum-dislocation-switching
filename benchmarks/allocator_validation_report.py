"""Section 93 report: writes ALLOCATOR_VALIDATION_RESULTS.md from results/allocator-validation-2026-09-30/report.json."""
import json
from pathlib import Path

from docpaths import write_doc

ROOT = Path(__file__).resolve().parents[1]
V = ROOT/'results'/'allocator-validation-2026-09-30'
mark = lambda b: 'PASS' if b else 'FAIL'
pc = lambda x: f'{100*x:.2f}%'


def main():
    r = json.loads((V/'report.json').read_text()); e = r['environment']
    L = ['# Allocator validation: converged reference, ensemble size, cost decomposition and seeds', '',
         f"September 30, 2026. Section 93 tests the Section 91 allocator against four open objections. The discrete reference was grid-defined, the cost was measured at one size, the timings ran on a loaded machine without metadata, and a compiled particle kernel might remove the advantage. Machine: {e['platform']}, {e['processor']}, {e['cpu_count']} logical cores; Python {e['python']}, NumPy {e['numpy']}, SciPy {e['scipy']}; one numerical thread; CPU time. Gates were declared in theory Section 93 before the runs.", '',
         '| Gate | Outcome |', '| --- | --- |']
    L += [f'| V1 converged reference (D512 vs D1024: all ≤ 1%, sparse ≤ 2%) | {mark(r["V1"]["gate"])} |',
          f'| V2 accuracy vs D* (H sparse ≤ 1.25 × D256; H < C256) | {mark(r["V2"]["gate"])} |',
          f'| V3 scaling in N (affine D, flat H, H ≤ 1.1 D, H ≤ D/3 for N ≥ 1e5, H error ≤ 1%) | {mark(r["V3"]["gate"])} |',
          f'| V5 seeds 91–95 (H < C sparse; H block ≤ 0.5%) | {mark(r["V5"]["gate"])} |',
          f'| V6 reproduction of Section 91.2 within 2% | {mark(r["V6"]["gate"])} |', '']
    v1 = r['V1']['D512_vs_Dstar']
    L += ['## V1–V2: accuracy against a grid- and time-converged discrete reference', '',
          f"D* is D on 1024² with dt/4 and the same physical core. D on 512² with dt/2 differs from it by {pc(v1['all'])} over all cells, {pc(v1['block'])} in blocks and {pc(v1['sparse'])} in the sparse region.", '',
          '| Method | All cells | Blocks | Sparse region |', '| --- | ---: | ---: | ---: |']
    for k, v in r['V2']['errors_vs_Dstar'].items():
        L.append(f"| {k} | {pc(v['all'])} | {pc(v['block'])} | {pc(v['sparse'])} |")
    L += ['', f"Sub-gates: (a) H ≤ 1.25 × D256 in the sparse region: {mark(r['V2']['gate_a'])}; (b) H < C256: {mark(r['V2']['gate_b'])}; H with n_c = 256 on (a): {mark(r['V2']['H256_gate_a'])}.", '']
    f = r['V3']['D_fit']
    L += ['## V3: scaling with the ensemble size', '',
          '| N dense | D (ms) | C (ms) | H (ms) | D/H | H particles | H sparse err | C sparse err |', '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for x in r['V3']['rows']:
        L.append(f"| {x['n_dense']:,} | {1e3*x['cost_D']:.1f} | {1e3*x['cost_C']:.1f} | {1e3*x['cost_H']:.1f} | {x['cost_D']/x['cost_H']:.1f} | {x['H_particles']} | {pc(x['H_err']['sparse'])} | {pc(x['C_err']['sparse'])} |")
    L += ['', f"D's cost fits {1e3*f['intercept']:.2f} ms + {1e9*f['per_dislocation']:.1f} ns × N (R² = {f['r2']:.4f}). First N at which H is cheaper than D: {r['V3']['first_N_H_cheaper']}. Sub-gates a–e: " + ', '.join(f"{k[-1]} {mark(r['V3'][k])}" for k in ('gate_a', 'gate_b', 'gate_c', 'gate_d', 'gate_e')) + '.', '']
    v4 = r['V4']
    L += ['## V4: cost decomposition and a compiled-particle projection', '',
          '| Component (ms per step) | D | H |', '| --- | ---: | ---: |']
    for k in ('stress', 'particle_grid', 'transport', 'allocation', 'rest', 'total'):
        L.append(f"| {k} | {1e3*v4['D'][k]:.2f} | {1e3*v4['H'][k]:.2f} |")
    L += ['', f"Measured D/H: {v4['measured_speedup']:.1f}. Projected D/H if particle–grid work were k times cheaper (model, not measurement): " + ', '.join(f"k = {k}: {s:.1f}" for k, s in v4['projected_speedup'].items()) + '. ' + v4['note'], '']
    v5 = r['V5']
    L += ['## V5: seed robustness', '', '| Seed | H sparse | C sparse | H block | C block |', '| ---: | ---: | ---: | ---: | ---: |']
    for x in v5['rows']:
        L.append(f"| {x['seed']} | {pc(x['H']['sparse'])} | {pc(x['C']['sparse'])} | {pc(x['H']['block'])} | {pc(x['C']['block'])} |")
    L += ['', f"Mean ± standard deviation: H {pc(v5['H_sparse_mean'])} ± {pc(v5['H_sparse_std'])}, C {pc(v5['C_sparse_mean'])} ± {pc(v5['C_sparse_std'])}.", '']
    v6 = r['V6']
    L += ['## V6: reproduction on a second machine', '', '| Quantity | Saved (Section 91.2, macOS) | Reproduced here | Relative difference |', '| --- | ---: | ---: | ---: |']
    for k in v6['saved']:
        L.append(f"| {k} | {pc(v6['saved'][k])} | {pc(v6['reproduced'][k])} | {v6['relative_difference'][k]:.2e} |")
    L += ['', '```sh', 'python benchmarks/allocator_validation.py run --method D --grid 1024 --tag ref-D1024   # and the other jobs listed in the report', 'python benchmarks/allocator_validation.py analyze', 'python benchmarks/allocator_validation_report.py', '```', '',
          '[Report](results/allocator-validation-2026-09-30/report.json), [theory Section 93](unified_multiscale_defect_framework.md).']
    write_doc('ALLOCATOR_VALIDATION_RESULTS.md', '\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
