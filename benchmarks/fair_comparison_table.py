"""Brief section 19 table: every measured quantity of the matched uniform/adaptive comparisons, from saved data."""
import json
from pathlib import Path
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/performance-complete-2026-09-13/summary.json'
SCALING=ROOT/'results/parallel-scaling-2026-09-26/report.json'
e=lambda x:'–' if x is None else f'{x:.1e}'


def main():
    s=json.loads(SOURCE.read_text());rows={r['name']:r for r in s['rows']}
    drift=lambda r:next((v for k,v in r['metrics'].items() if k in ('mass_drift','inventory_drift')),None)
    lines=['# Fair comparison against uniform phase field: every section 19 quantity','',
        'September 26, 2026. This table collects, from saved data, every quantity that brief section 19 asks for in the matched comparisons. Every pair uses the same domain, coefficients, physical interface width, initial and boundary conditions, time step, tolerances, hardware and a single numerical thread. Timings are medians of three fresh-process runs ([protocol](PERFORMANCE_PROTOCOL.md), [timing study](PERFORMANCE_RESULTS.md), [source data](results/performance-complete-2026-09-13/summary.json)). The adaptive and uniform variants of a pair solve the same discretized physics, so the first table measures what adaptivity changes. The second table measures accuracy against a finer reference solution.','',
        '## Adaptive variant against its uniform reference','',
        '| Variant | Field | L1 | L2 | L∞ | Relative L2 | Energy error (rel.) | Line position error | Solver speedup | Launch-to-exit speedup | DOF ratio | Steps | Conservation drift (uniform / adaptive) | Peak RSS MiB (uniform / adaptive) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for p in s['pairs']:
        a,u=rows[p['variant']],rows[p['reference']];first=True
        for field,err in p['field_errors'].items():
            common=(f"| {e(p['native_energy_relative'])} | {e(p['line_position_absolute'])} | {p['evolution_ratio']:.2f} | {p['launch_to_exit_ratio']:.2f} | {p['dof_ratio']:.2f} | {a['steps']} | {e(drift(u))} / {e(drift(a))} | {u['post_output_peak_rss_bytes']/2**20:.1f} / {a['post_output_peak_rss_bytes']/2**20:.1f} |") if first else '| | | | | | | | |'
            lines.append(f"| {p['variant'] if first else ''} | {field} | {e(err['l1'])} | {e(err['l2'])} | {e(err['linf'])} | {e(err['relative_l2'])} "+common)
            first=False
    lines+=['','Solver iteration counts are not reported by these explicit or semi-implicit solvers, which take one solve per accepted step; the step count is the iteration measure. The dense Cahn–Hilliard pair is identical because its allocator falls back to the full grid (no saving, speedup 0.98).','',
        '## Accuracy against a finer reference (Allen–Cahn, n=512, dt=0.0025)','',
        '| Configuration | L1 | L2 | L∞ | Relative L2 | Interface-position error | Free-energy error (rel.) | Declared gates (1% field, 0.05 radius, 1% energy) |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for name,r in rows.items():
        f=r.get('fine_reference_comparison')
        if not f:continue
        pf=f['phase_fraction'];lines.append(f"| {name} | {e(pf['l1'])} | {e(pf['l2'])} | {e(pf['linf'])} | {e(pf['relative_l2'])} | {f['radius_absolute']:.4f} | {e(f['native_energy_relative'])} | {'pass' if f['accuracy_met'] else 'FAIL'} |")
    lines+=['','Adaptive and uniform variants have the same error against the finer reference to three digits. Adaptivity does not add measurable error, and the coarsest pair (n=128) fails its declared field gate for both variants.','']
    if SCALING.exists():
        g=json.loads(SCALING.read_text());ratio=g['adaptive_over_uniform_speedup']
        lines+=['## Same comparison under thread parallelism','',
            f"With the case 1 step compiled and run on 1, 2, 4 and 8 threads (Section 55, [PARALLEL_SCALING_RESULTS.md](PARALLEL_SCALING_RESULTS.md)), the adaptive/uniform ratio at equal thread count is "+', '.join(f"{ratio[p]:.2f} ({p} thread{'s' if p!='1' else ''})" for p in ratio)+f". The adaptive and uniform fields differ by {g['accuracy_t4']['relative_l2']:.1e} relative L2. Adaptive savings shrink as threads are added, because the smaller adaptive problem saturates first.",'']
    lines+=['## Not measured','','- An independent adaptive-mesh PF code as a second baseline (the brief makes this conditional on one being implemented). The adaptive variant here is itself tensor-grid refinement, so it is not a separate baseline.',
        '- Distributed-memory scaling (Section 55 measures shared-memory threading only).','','```sh','python3 benchmarks/fair_comparison_table.py','```','']
    write_doc('FAIR_COMPARISON.md','\n'.join(lines))


if __name__=='__main__':main()
