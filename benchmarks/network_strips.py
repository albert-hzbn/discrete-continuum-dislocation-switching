"""Section 70: TCP transport for the message-passing strips, a validated message-cost model and a multi-node projection."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform
import multiprocessing as mp
from multiprocessing.connection import Listener,Client
from pathlib import Path
from time import perf_counter
import numpy as np
from bmvr.grid import Grid,circle
from bmvr.energy import DoubleWell
from bmvr.solvers import ac_step
from bmvr.adaptivity import focused_grid
from bmvr import decomposition as D

P_LIST=(1,2,4,8);H=.5;DT=.02
NETWORKS={'1GbE':dict(alpha=50e-6,beta=117e6),'10GbE':dict(alpha=15e-6,beta=1.1e9)}


def serial(grid,v,steps):
    for _ in range(steps):v=ac_step(grid,v,DoubleWell(),DT)
    return v


def _echo(conn_kind,address,n,size):
    if conn_kind=='tcp':
        with Listener(address,authkey=D.AUTHKEY) as listener:
            with listener.accept() as c:
                for _ in range(n):c.send_bytes(c.recv_bytes())
    else:
        c=address
        for _ in range(n):c.send_bytes(c.recv_bytes())


def pingpong(kind,size,n=4000):
    """One-way message time for `size` bytes, from round trips between two processes."""
    ctx=mp.get_context('spawn');payload=b'x'*size
    if kind=='tcp':
        address=('127.0.0.1',47300);p=ctx.Process(target=_echo,args=('tcp',address,n,size));p.start()
        for _ in range(600):
            try:c=Client(address,authkey=D.AUTHKEY);break
            except OSError:__import__('time').sleep(.05)
    else:
        a,b=ctx.Pipe();p=ctx.Process(target=_echo,args=('pipe',b,n,size));p.start();c=a
    for _ in range(200):c.send_bytes(payload);c.recv_bytes()
    start=perf_counter()
    for _ in range(n-200):c.send_bytes(payload);c.recv_bytes()
    t=(perf_counter()-start)/(n-200)/2;c.close();p.join();return t


def measure(grid,parts,transport,per_cell):
    est=per_cell*grid.size/parts;warmup=max(20,int(.5/est));steps=max(30,int(.6/est))
    v0=circle(grid,14.);best=None
    for _ in range(3):
        t,field=D.run(grid,v0,parts,steps,warmup,DT,transport=transport);best=t if best is None else min(best,t)
    return dict(parts=parts,transport=transport,seconds_per_step=best,bitwise=bool(np.array_equal(field,serial(grid,v0,warmup+steps))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    g=Grid.uniform(256.,512);v=circle(g,14.);start=perf_counter();serial(g,v,40);per_cell=(perf_counter()-start)/40/g.size
    grids=dict(uniform=Grid.uniform(512.,1024),adaptive=focused_grid(512.,H,((-25,25),(-25,25))))
    rows={name:[] for name in grids}
    for name,grid in grids.items():
        for p in P_LIST:
            for tr in ('pipe','tcp'):rows[name].append(measure(grid,p,tr,per_cell));print(name,rows[name][-1],flush=True)
    halo=8*grids['uniform'].shape[1];msg={k:pingpong(k,halo) for k in ('pipe','tcp')};print('ping-pong',msg,flush=True)
    t=lambda name,p,tr:next(r['seconds_per_step'] for r in rows[name] if r['parts']==p and r['transport']==tr)
    predicted={p:t('uniform',p,'pipe')+2*(msg['tcp']-msg['pipe']) for p in P_LIST if p>1}
    model_error={str(p):abs(predicted[p]-t('uniform',p,'tcp'))/t('uniform',p,'tcp') for p in predicted}
    gates=dict(N1_bitwise=all(r['bitwise'] for rs in rows.values() for r in rs),
        N2_tcp_overhead=all(t(n,p,'tcp')<=1.25*t(n,p,'pipe') for n in grids for p in P_LIST if p>1),
        N3_message_model=all(e<=.25 for e in model_error.values()))
    projection={}
    for name,grid in grids.items():
        T1=t(name,1,'pipe');m=8*grid.shape[1]
        projection[name]={net:{str(p):dict(seconds_per_step=T1/p+(2*(c['alpha']+m/c['beta']+msg['tcp']) if p>1 else 0.),
            efficiency=(T1/p)/(T1/p+(2*(c['alpha']+m/c['beta']+msg['tcp']) if p>1 else 0.))) for p in (1,2,4,8,16,32,64)} for net,c in NETWORKS.items()}
    report=dict(scope='Section 70: TCP strips measured on one machine (loopback), message-cost model validated there, multi-node projection with stated interconnect assumptions; no second machine used.',
        serial_ns_per_cell=per_cell*1e9,rows=rows,halo_bytes=halo,one_way_message_seconds=msg,predicted_tcp=predicted,model_relative_error=model_error,gates=gates,
        networks=NETWORKS,projection=projection,environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),cpu_count=os.cpu_count()))
    (out/'report.json').write_text(json.dumps(report,indent=1,default=lambda o:o.item())+'\n');print(gates,model_error,flush=True)


if __name__=='__main__':main()
