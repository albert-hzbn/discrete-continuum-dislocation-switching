"""Stronger spatial relaxation supplement, keeping the first matrix intact."""
import argparse
from time import perf_counter
from pathlib import Path
from thermoplastic import run, comparison, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True);start=perf_counter()
    runs=[run(16,dt,'nonaffine',output) for dt in (.001,.0005)]
    errors=comparison(*runs,output)
    gates={'temperature_time_refinement':errors['temperature']<.01,'slip_time_refinement':errors['slip']<.01}
    report=dict(runs=runs,comparison=errors,gates=gates,all_checks=all(all(r['checks'].values()) for r in runs),
                study_seconds=perf_counter()-start)
    write_json(output/'report.json',report)
    print({'gates':gates,'errors':errors,'seconds':report['study_seconds']},flush=True)


if __name__=='__main__':main()
