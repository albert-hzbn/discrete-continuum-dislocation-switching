"""Section 70.6 diagnostic: strip compute time when p strips run at the same time, without messages (contention probe)."""
import json,subprocess,sys
from pathlib import Path

if __name__=='__main__':
    here=Path(__file__).resolve().parent/'strip_node.py';out={}
    for p in map(int,sys.argv[1:]):
        procs=[subprocess.Popen([sys.executable,str(here),'compute','--rank',str(r),'--parts',str(p),'--steps','30'],stdout=subprocess.PIPE,text=True) for r in range(p)]
        times=[json.loads(q.communicate()[0].strip().splitlines()[-1])['compute_seconds_per_step'] for q in procs]
        out[p]=dict(max=max(times),median=sorted(times)[len(times)//2]);print(p,out[p],flush=True)
    print(json.dumps(out))
