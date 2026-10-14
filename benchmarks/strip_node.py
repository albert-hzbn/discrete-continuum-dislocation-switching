"""Section 70.4: one rank of the case 1 strip decomposition, for runs whose ranks live on different machines.

Every rank builds the same grid and initial field, owns one strip, exchanges halo rows over TCP (decomposition.connect_tcp)
and, at the end, ranks > 0 send their strip to rank 0 over the same link. Modes: `run` (a decomposed trajectory), `compute`
(the strip update alone, no messages) and `echo`/`ping` (halo-sized ping-pong).
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,platform
from time import perf_counter
import numpy as np
from bmvr.grid import Grid,circle
from bmvr.energy import DoubleWell
from bmvr import decomposition as D

LENGTH,N,DT,RADIUS=512.,1024,.02,14.


def setup(rank,parts):
    grid=Grid.uniform(LENGTH,N);v0=circle(grid,RADIUS);ny=grid.shape[0];r0,r1=D.strips(ny,parts)[rank]
    g,lo,hi=D._extended(grid.xe,grid.ye,r0,r1,ny);return g,v0[lo:hi].copy(),r0-lo,r1-lo


def update(g,v,a,b,model=DoubleWell()):
    force=model.W*(v**3-v)-g.lap(v,model.k);v[a:b]=v[a:b]-DT*force[a:b]


def compute_only(rank,parts,steps):
    g,v,a,b=setup(rank,parts)
    for _ in range(20):update(g,v,a,b)
    best=None
    for _ in range(3):
        start=perf_counter()
        for _ in range(steps):update(g,v,a,b)
        t=(perf_counter()-start)/steps;best=t if best is None else min(best,t)
    return best


def run(rank,parts,addresses,steps,warmup):
    g,v,a,b=setup(rank,parts);up,down,_=D.connect_tcp(rank,parts,addresses)
    def step():
        if up is not None:
            v[a-1]=np.frombuffer(up.recv_bytes(),dtype=float);up.send_bytes(v[a].tobytes())
        if down is not None:
            down.send_bytes(v[b-1].tobytes());v[b]=np.frombuffer(down.recv_bytes(),dtype=float)
        update(g,v,a,b)
    for _ in range(warmup):step()
    start=perf_counter()
    for _ in range(steps):step()
    elapsed=(perf_counter()-start)/steps
    # Chain gather (Section 70.6): each rank receives the strips of all ranks below it, then passes them up with its own.
    pieces=[]
    if down is not None:
        for _ in range(int(down.recv_bytes())):pieces.append((json.loads(down.recv_bytes()),down.recv_bytes()))
    if rank>0:
        own=(dict(rank=rank,seconds_per_step=elapsed,platform=platform.platform(),numpy=np.__version__),v[a:b].tobytes())
        up.send_bytes(str(len(pieces)+1).encode());up.send_bytes(json.dumps(own[0]).encode());up.send_bytes(own[1])
        for info,data in pieces:up.send_bytes(json.dumps(info).encode());up.send_bytes(data)
        return None
    pieces.sort(key=lambda x:x[0]['rank'])
    field=np.concatenate([v[a:b].ravel()]+[np.frombuffer(d,dtype=float) for _,d in pieces]).reshape(-1,v.shape[1])
    return dict(rank0_seconds_per_step=elapsed,ranks=[i for i,_ in pieces],field=field)


def pingpong(role,address,n):
    from multiprocessing.connection import Listener,Client
    payload=b'x'*(8*N)
    if role=='echo':
        with Listener(address,authkey=D.AUTHKEY) as listener, listener.accept() as c:
            for _ in range(n):c.send_bytes(c.recv_bytes())
        return None
    c=Client(address,authkey=D.AUTHKEY)
    for _ in range(20):c.send_bytes(payload);c.recv_bytes()
    start=perf_counter()
    for _ in range(n-20):c.send_bytes(payload);c.recv_bytes()
    t=(perf_counter()-start)/(n-20)/2;c.close();return t


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','compute','echo','ping','launch'));p.add_argument('--rank',type=int,default=0);p.add_argument('--parts',type=int,default=2)
    p.add_argument('--port',type=int,default=47400);p.add_argument('--steps',type=int,default=100);p.add_argument('--warmup',type=int,default=10);p.add_argument('--out');p.add_argument('--first',type=int,default=1);a=p.parse_args()
    address=('127.0.0.1',a.port);addresses=[('127.0.0.1',a.port+i) for i in range(a.parts-1)]
    if a.mode=='launch':  # start ranks first..parts-1 on this machine and wait (the guest side of a mixed run)
        import subprocess,sys
        procs=[subprocess.Popen([sys.executable,__file__,'run','--rank',str(r),'--parts',str(a.parts),'--port',str(a.port),'--steps',str(a.steps),'--warmup',str(a.warmup)]) for r in range(a.first,a.parts)]
        print(json.dumps(dict(exit=[q.wait() for q in procs])),flush=True);return
    if a.mode=='compute':print(json.dumps(dict(rank=a.rank,compute_seconds_per_step=compute_only(a.rank,a.parts,a.steps),platform=platform.platform())),flush=True)
    elif a.mode in ('echo','ping'):
        t=pingpong(a.mode,address,a.steps)
        if t is not None:print(json.dumps(dict(one_way_seconds=t)),flush=True)
    else:
        r=run(a.rank,a.parts,addresses,a.steps,a.warmup)
        if r is not None:
            np.save(a.out,r.pop('field'));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
