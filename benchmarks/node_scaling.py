"""Section 70.6: strong scaling of the case 1 strips on one many-core machine, with a halo ping-pong, for the message model."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
from bmvr.grid import Grid,circle
from bmvr.energy import DoubleWell
from bmvr.solvers import ac_step
from bmvr import decomposition as D

P_LIST=(1,2,4,8,16,32);DT=.02


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--steps',type=int,default=40);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);g=Grid.uniform(512.,1024);v0=circle(g,14.);rows=[];ref=None
    for p in P_LIST:
        for tr in ('pipe','tcp'):
            best=None
            for _ in range(3):
                t,field=D.run(g,v0,p,args.steps,10,DT,transport=tr,base_port=47500);best=t if best is None else min(best,t)
            if ref is None:
                ref=v0.copy()
                for _ in range(args.steps+10):ref=ac_step(g,ref,DoubleWell(),DT)
            rows.append(dict(parts=p,transport=tr,seconds_per_step=best,bitwise_vs_guest_serial=bool(np.array_equal(field,ref))));print(rows[-1],flush=True)
    np.save(out/'guest-serial-field.npy',ref)
    here=Path(__file__).resolve().parent/'strip_node.py';env=dict(os.environ)
    echo=subprocess.Popen([sys.executable,str(here),'echo','--steps','2020','--port','47600'],env=env);time.sleep(3)
    ping=subprocess.run([sys.executable,str(here),'ping','--steps','2020','--port','47600'],env=env,capture_output=True,text=True);echo.wait()
    t_msg=json.loads(ping.stdout.strip().splitlines()[-1])['one_way_seconds']
    (out/'report.json').write_text(json.dumps(dict(rows=rows,one_way_message_seconds=t_msg,steps=args.steps,platform=platform.platform(),numpy=np.__version__,cpu_count=os.cpu_count()),indent=1)+'\n')
    print('t_msg',t_msg,flush=True)


if __name__=='__main__':main()
