"""Run one Section 98 job, resuming from its checkpoint after a crash (up to five attempts); every attempt is logged.

python benchmarks/creep_retry.py A 30 96
"""
import datetime, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'/'creep-cavitation-3d-2026-09-30'
args = sys.argv[1:]
log = OUT/f'retry_{"_".join(args)}.log'
for attempt in range(1, 6):
    with open(log, 'a') as f:
        f.write(f'{datetime.datetime.now().isoformat()} attempt {attempt} start\n')
        r = subprocess.run([sys.executable, str(ROOT/'benchmarks'/'creep_cavitation3d.py'), 'run', *args],
                           stdout=f, stderr=subprocess.STDOUT)
        f.write(f'{datetime.datetime.now().isoformat()} attempt {attempt} exit {r.returncode}\n')
    if r.returncode == 0:
        break
