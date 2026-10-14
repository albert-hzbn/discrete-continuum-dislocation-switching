"""Direct Q2 temporal sensitivity, separate from inherited Q1 evidence."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from thermochemical import run,comparison,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-existing',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);start=perf_counter()
    reference=json.loads((args.base/'report.json').read_text());fine=next(x for x in reference['runs'] if x['n']==16)
    if args.reuse_existing:
        coarse=json.loads((args.output/'q2-coupled-16-0.0005.json').read_text())
        if (coarse['n'],coarse['dt'],coarse['degree'],coarse['variant'])!=(16,.0005,2,'coupled'):
            raise ValueError('Unexpected saved trajectory metadata.')
    else:coarse=run(16,.0005,'coupled',args.output,degree=2)
    errors=comparison(coarse,fine,args.output,args.base);gates={key:value<.01 for key,value in errors.items()}
    write_json(args.output/'report.json',dict(run=coarse,comparison=errors,gates=gates,
        trajectory_reused=args.reuse_existing,invocation_seconds=perf_counter()-start))
    print({'errors':errors,'gates':gates},flush=True)


if __name__=='__main__':main()
