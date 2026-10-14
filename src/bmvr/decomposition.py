"""Section 61.2: message-passing strip decomposition of the case 1 Allen-Cahn step.

Each process owns a strip of rows in private memory and exchanges one halo row with each neighbour per step
through a pipe. The strip update applies the serial finite-volume operator to the halo-extended rows, so every
owned cell repeats the serial operation order and the assembled field equals the serial step bitwise.
"""

import os
from time import perf_counter, sleep
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Listener, Client
from .grid import Grid
from .energy import DoubleWell


def strips(ny, parts):
    bounds = np.linspace(0, ny, parts + 1).round().astype(int)
    return list(zip(bounds[:-1], bounds[1:]))


def _extended(xe, ye, r0, r1, ny):
    lo, hi = max(r0 - 1, 0), min(r1 + 1, ny)
    return Grid(xe, ye[lo:hi + 1]), lo, hi


AUTHKEY = b"bmvr-strips"


def connect_tcp(rank, parts, addresses):
    """Section 70: TCP links between neighbouring ranks, one per boundary. Boundary i (between ranks i and i+1) listens at
    addresses[i] on rank i+1 and is dialled by rank i. Every rank opens its listener before dialling, and a dial waits
    for the listener, so the chain forms without deadlock whether the ranks share a machine or not."""
    listener = Listener(addresses[rank - 1], authkey=AUTHKEY) if rank > 0 else None
    down = None
    if rank < parts - 1:
        for _ in range(600):
            try:
                down = Client(addresses[rank], authkey=AUTHKEY)
                break
            except (ConnectionRefusedError, OSError, EOFError):  # a port forward can accept before the far listener exists
                sleep(.05)
        else:
            raise ConnectionError(f'rank {rank} could not reach {addresses[rank]}')
    up = listener.accept() if listener is not None else None
    return up, down, listener


def _worker(rank, xe, ye, field, r0, r1, ny, up, down, steps, warmup, dt, barrier, result, addresses=None):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    if addresses is not None:
        up, down, _ = connect_tcp(rank, len(addresses) + 1, addresses)
    grid, lo, hi = _extended(xe, ye, r0, r1, ny)
    model = DoubleWell()
    v = field.copy()  # rows lo..hi-1: owned rows plus existing halos
    a, b = r0 - lo, r1 - lo  # owned slice within v

    def exchange():
        # Deadlock-free order: across each boundary the upper rank receives first and the lower rank sends first.
        # (Sending first on both sides deadlocks once a row exceeds the socket buffer, about 8 KB on macOS.)
        if up is not None:
            v[a - 1] = np.frombuffer(up.recv_bytes(), dtype=float)
            up.send_bytes(v[a].tobytes())
        if down is not None:
            down.send_bytes(v[b - 1].tobytes())
            v[b] = np.frombuffer(down.recv_bytes(), dtype=float)

    def step():
        exchange()
        force = model.W * (v ** 3 - v) - grid.lap(v, model.k)
        v[a:b] = v[a:b] - dt * force[a:b]

    for _ in range(warmup):
        step()
    barrier.wait()
    start = perf_counter()
    for _ in range(steps):
        step()
    elapsed = perf_counter() - start
    result.put((rank, elapsed, v[a:b].copy()))


def run(grid, v0, parts, steps, warmup, dt, transport="pipe", host="127.0.0.1", base_port=47100):
    """Run warmup+steps decomposed steps; returns (wall seconds per timed step, assembled field).

    transport="tcp" (Section 70) links neighbours through TCP sockets at host:base_port+i instead of pipes; the
    arithmetic is identical, so the assembled field is still bitwise equal to the serial step."""
    ctx = mp.get_context("spawn")
    ny = grid.shape[0]
    bands = strips(ny, parts)
    tcp = transport == "tcp"
    addresses = [(host, base_port + i) for i in range(parts - 1)] if tcp else None
    pipes = [] if tcp else [ctx.Pipe() for _ in range(parts - 1)]  # pipe i connects rank i (down end) and rank i+1 (up end)
    barrier, result, procs = ctx.Barrier(parts), ctx.Queue(), []
    for rank, (r0, r1) in enumerate(bands):
        lo, hi = max(r0 - 1, 0), min(r1 + 1, ny)
        up = pipes[rank - 1][1] if rank > 0 and not tcp else None
        down = pipes[rank][0] if rank < parts - 1 and not tcp else None
        p = ctx.Process(target=_worker, args=(rank, grid.xe, grid.ye, v0[lo:hi], r0, r1, ny, up, down, steps, warmup, dt, barrier, result, addresses if tcp and parts > 1 else None))
        p.start()
        procs.append(p)
    pieces = [result.get() for _ in range(parts)]
    for p in procs:
        p.join()
    pieces.sort()
    field = np.concatenate([piece[2] for piece in pieces])
    return max(piece[1] for piece in pieces) / steps, field
