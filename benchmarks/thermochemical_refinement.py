"""Preserve the failed 16/24 slip gate and extend the fixed-width mesh study."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from thermochemical import run,comparison,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True);start=perf_counter()
    original=json.loads((args.base/'report.json').read_text())
    coarse=next(x for x in original['runs'] if x['variant']=='coupled' and x['n']==24)
    runs=[run(n,.00025,'coupled',output) for n in (32,48)]
    previous=comparison(coarse,runs[0],args.base,output);fine=comparison(runs[0],runs[1],output)
    gates={key:value<.05 for key,value in fine.items()}
    report=dict(runs=runs,comparison_24_32_different_dt=previous,comparison_32_48=fine,gates=gates,
        all_checks=all(all(x['checks'].values()) for x in runs),retained_original_slip_error=original['mesh_comparison']['slip'],
        study_seconds=perf_counter()-start)
    write_json(output/'report.json',report);print({'gates':gates,'errors':fine,'seconds':report['study_seconds']},flush=True)


if __name__=='__main__':main()
