"""Section 70 report: TCP strips, message-cost model and multi-node projection, from saved evidence."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docpaths import write_doc

ROOT=Path(__file__).resolve().parents[1]
mark=lambda b:'PASS' if b else 'FAIL'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input;r=json.loads((out/'report.json').read_text());g=r['gates']
    figures=out/'figures';figures.mkdir(exist_ok=True);fig,ax=plt.subplots(figsize=(6.4,4.2),layout='constrained')
    for i,(name,pr) in enumerate(r['projection'].items()):
        for ls,(net,v) in zip(('-','--'),pr.items()):
            ps=[int(p) for p in v];ax.plot(ps,[v[str(p)]['efficiency'] for p in ps],ls,color=f'C{i}',marker='o',ms=3,label=f'{name}, {net}')
    ax.set_xscale('log',base=2);ax.set(xlabel='Nodes (one rank each)',ylabel='Projected parallel efficiency',title='Multi-node projection from the validated message model',ylim=(0,1.05));ax.legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(figures/f'network-projection.{ext}',dpi=150)
    plt.close(fig);rel=out.relative_to(ROOT) if out.is_absolute() else out
    lines=['# Network transport for the strips and a multi-node projection','',
        'September 27, 2026. Section 70 replaces the Section 61 pipes with authenticated TCP sockets, so the same strip workers run across machines. The declared tests run on one machine over loopback, alongside the Section 68 notch runs. A message-cost model built from ping-pong microbenchmarks is validated against the measured TCP step times and then projects multi-node efficiency for stated interconnects. No second machine was used, so the projection is labelled as one.','',
        '| Gate | Outcome |','| --- | --- |']+[f'| {k} | {mark(v)} |' for k,v in g.items()]
    lines+=['','| Grid | p | Pipe s/step | TCP s/step | TCP / pipe | Bitwise (TCP) |','| --- | ---: | ---: | ---: | ---: | --- |']
    for name,rs in r['rows'].items():
        for p in (1,2,4,8):
            a=[x for x in rs if x['parts']==p];lines.append(f"| {name} | {p} | {a[0]['seconds_per_step']:.2e} | {a[1]['seconds_per_step']:.2e} | {a[1]['seconds_per_step']/a[0]['seconds_per_step']:.2f} | {a[1]['bitwise']} |")
    m=r['one_way_message_seconds']
    lines+=['',f"The ping-pong one-way time for an {r['halo_bytes']}-byte halo row is {m['pipe']*1e6:.1f} μs over a pipe and {m['tcp']*1e6:.1f} μs over loopback TCP. With these, the model predicts the uniform-grid TCP step time within "+', '.join(f"{100*e:.1f}% (p={p})" for p,e in r['model_relative_error'].items())+'.',
        '','N2 fails as declared, on the adaptive grid: TCP/pipe is 1.26 at p = 1, 1.34 at p = 4 and 0.79 at p = 8. At p = 1 there are no messages at all, so a 26% difference there is timing noise. The adaptive steps last 1–2 ms and share the machine with three notch runs. On the uniform grid, whose 2–8 ms steps are less noisy, every ratio is within 0.96–1.03, in line with the 16 μs extra per message that the model assigns to TCP.',
        '','## Projection (no gate)','','| Grid | Network | p = 2 | 4 | 8 | 16 | 32 | 64 |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,pr in r['projection'].items():
        for net,v in pr.items():lines.append(f"| {name} | {net} | "+' | '.join(f"{v[str(p)]['efficiency']:.2f}" for p in (2,4,8,16,32,64))+' |')
    lines+=['','The uniform 1024² case keeps 77% efficiency on 8 nodes of 1 GbE and 90% on 10 GbE. The adaptive grid has 9.5x fewer cells, so latency dominates sooner: 51% and 67% on 8 nodes. This is the Section 55 conclusion carried to networks. Adaptivity and distributed parallelism compete for the same work, and a small adaptive problem stops scaling first. The projection ignores memory-bandwidth effects, load imbalance and network jitter. It is a model prediction validated only on loopback, not a cluster measurement.',
        '',f'![Projected multi-node efficiency]({rel}/figures/network-projection.png)','','```sh',f'python3 benchmarks/network_strips.py --output {rel}',f'python3 benchmarks/network_strips_report.py --input {rel}','```','',
        f'[Numerical report]({rel}/report.json), [theory Section 70](unified_multiscale_defect_framework.md).','']
    two=ROOT/'results'/'two-machine-2026-09-27'/'report.json'
    if two.exists():
        x=json.loads(two.read_text());c=x['compute_seconds_per_step'];trel=two.parent.relative_to(ROOT)
        lines+=['## Two machines (Section 70.4)','',
            f"Rank 0 runs on this Mac (arm64) and rank 1 on remote-vm ({x['environment']['rank1']}), linked by an SSH port forward across the tailnet. The case is the uniform 1024² grid in two strips.",'',
            '| Gate | Outcome |','| --- | --- |',f"| M1: field equals serial to 1e-12 | {mark(x['gates']['M1_field'])} ({x['field_check']['relative_Linf']:.1e}; {x['field_check']['cells_differing']} cells differ in the last bit, so not bitwise) |",
            f"| M2: declared model within 25% | {mark(x['gates']['M2_declared_model'])} (predicted {1e3*x['declared_model']['prediction']:.1f} ms, measured {1e3*x['best']:.1f} ms) |",'',
            f"Compute per step is {1e3*c['rank0_mac']:.1f} ms on the Mac and {1e3*c['rank1_vm']:.1f} ms on the guest, and the one-way halo time is {1e3*x['one_way_message_seconds']:.1f} ms. The two-machine runs took "+', '.join(f'{1e3*v:.1f}' for v in x['two_machine_seconds_per_step'])+' ms per step.',
            '',f"The declared model adds the round trip to the slower rank's compute. The measurement shows the latency hidden behind the slow rank instead. The post hoc overlap form, {x['post_hoc_overlap_model']['formula']}, predicts {1e3*x['post_hoc_overlap_model']['prediction']:.1f} ms ({100*x['post_hoc_overlap_model']['relative_error']:.1f}% off). For balanced ranks both forms agree, so the loopback check and the projection above stand. The first attempt's handshake race (EOFError through the forward) is recorded in the report and fixed in connect_tcp.",
            '',f'[Two-machine report]({trel}/report.json). Commands: on the guest, `python benchmarks/strip_node.py run --rank 1 --steps 100 --warmup 10`; on the Mac, with `ssh -N -L 47400:127.0.0.1:47400 ai-vm` running, `python3 benchmarks/strip_node.py run --rank 0 --steps 100 --warmup 10 --out field.npy`.','']
    ns=ROOT/'results'/'node-scaling-2026-09-27'/'report.json'
    if ns.exists():
        x=json.loads(ns.read_text());tcp={r['parts']:r['seconds_per_step'] for r in x['guest']['rows'] if r['transport']=='tcp'}
        lines+=['## Beyond two machines (Section 70.6)','','| Gate | Outcome |','| --- | --- |']+[f'| {k} | {mark(v)} |' for k,v in x['gates'].items()]
        lines+=['','Guest alone (remote-vm, TCP strips; every run bitwise equal to serial):','','| p | Measured ms/step | Model T_1/p + 2 t_msg | Error |','| ---: | ---: | ---: | ---: |']
        lines+=[f"| {p} | {1e3*tcp[p]:.1f} | {1e3*x['P2'][str(p)]['model']:.1f} | {100*x['P2'][str(p)]['rel_error']:.0f}% |" for p in (2,4,8,16,32)]
        lines+=['','A single strip run alone computes in T_1/p, but concurrent strips on the virtualized guest run 1.5–1.6x slower at p = 4–8 and vary widely beyond that. The communication term (2 × 189 μs) is not the cause. P2 fails on the compute-scaling assumption under contention.','',
            'Mixed layouts (rank 0 on the Mac, the other ranks on the guest):','','| p | Best ms/step | Overlap model | Error |','| ---: | ---: | ---: | ---: |']
        lines+=[f"| {p} | {1e3*v['best']:.1f} | {1e3*v['model']:.1f} | {100*v['rel_error']:.1f}% |" for p,v in x['mixed'].items()]
        lines+=['','The mixed runs are latency-bound by the 65.8 ms tailnet round trip, and the overlap model predicts them within 10%. Their fields equal the Mac serial field bitwise.','','Scripts: [node_scaling.py](benchmarks/node_scaling.py), [strip_node.py](benchmarks/strip_node.py) (launch mode), [concurrent_compute.py](benchmarks/concurrent_compute.py).','']
    write_doc('NETWORK_STRIPS_RESULTS.md','\n'.join(lines))


if __name__=='__main__':main()
