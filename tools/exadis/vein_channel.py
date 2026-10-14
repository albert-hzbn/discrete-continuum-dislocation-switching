"""Section 99: 3D allocator test on copper with ExaDiS (loop-rich wall + loop-free channel).

Runs inside WSL with the ExaDiS python binding (tools/exadis/run.sh).
    python vein_channel.py <method D|H|0> <seed> [alpha] [outdir]
Writes <outdir>/<method>_s<seed>[_a<alpha>].json and a VTK file of the final network.
"""
import json, os, sys, time

import numpy as np
import pyexadis
from pyexadis_base import (ExaDisNet, DisNetManager, NodeConstraints, SimulateNetwork, CalForce, MobilityLaw,
                           TimeIntegration, Collision, Topology, Remesh)
from pyexadis_utils import insert_frank_read_src, insert_prismatic_loop, write_vtk, dislocation_density

L = 8000.0                  # box side in b (2.04 um)
Z0, HW = 4000.0, 800.0      # wall centre and half-thickness
R_LOOP, N_WALL, N_STRAY = 120.0, 3000, 100
MIN_SEP = 250.0
# Section 99 defaults; Section 100 overrides SIGMA, L_SRC and Z_SRC through the environment.
N_SRC, L_SRC = 6, float(os.environ.get("L_SRC", 1500.0))
SIGMA = float(os.environ.get("SIGMA", 200e6))
Z_SRC = float(os.environ.get("Z_SRC", 2000.0))
SRC_AT_WALL = os.environ.get("SRC_AT_WALL", "0") == "1"   # Section 100: sources centred on the wall faces
T_END = float(os.environ.get("T_END", 1e-9))
BLOCKS, N_C = 8, 8
NGRID_FRIC = 32
STATE = dict(crystal='fcc', burgmag=2.55e-10, mu=54.6e9, nu=0.324, a=6.0, maxseg=300.0, minseg=60.0,
             rtol=3.0, rann=5.0, nextdt=1e-12, maxdt=1e-10)
B110 = np.array([[0, 1, 1], [0, -1, 1], [1, 0, 1], [-1, 0, 1], [1, 1, 0], [-1, 1, 0]])/np.sqrt(2)
N111 = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]])/np.sqrt(3)


def slip_systems_schmid():
    out = []
    for n in N111:
        for b in B110:
            if abs(n @ b) < 1e-9:
                m = abs(n[2]*b[2])
                out.append((b, n, m))
    return [(b, n) for b, n, m in out if m > .4]           # the eight systems with Schmid factor 0.408 for [001]


def loop_centres(rng, count, zlo, zhi, taken, inside_wall):
    out = []
    while len(out) < count:
        c = rng.random(3)*L
        c[2] = zlo + rng.random()*(zhi - zlo) if inside_wall else c[2]
        if not inside_wall and abs((c[2] - Z0 + L/2) % L - L/2) <= HW + R_LOOP:
            continue
        pts = np.array(taken + out) if (taken or out) else np.zeros((0, 3))
        if len(pts):
            d = (pts - c + L/2) % L - L/2
            if np.min(np.linalg.norm(d, axis=1)) < MIN_SEP:
                continue
        out.append(c)
    return out


def build(seed):
    rng = np.random.default_rng(seed)
    wall = loop_centres(rng, N_WALL, Z0 - HW, Z0 + HW, [], True)
    stray = loop_centres(rng, N_STRAY, 0, L, wall, False)
    loops = [(c, B110[rng.integers(6)]*rng.choice([-1, 1])) for c in wall + stray]
    systems = slip_systems_schmid()
    sources = []
    for k in range(N_SRC):
        b, n = systems[rng.integers(len(systems))]
        c = rng.random(3)*L; c[2] = (rng.uniform(-Z_SRC, Z_SRC)) % L
        if SRC_AT_WALL:                                  # Section 100: centre on a wall face (half of each source in the wall)
            c[2] = Z0 + HW*rng.choice([-1, 1])
        sources.append((c, b, n))
    return loops, sources


def block_of(c):
    return tuple((np.floor(c/(L/BLOCKS)).astype(int)) % BLOCKS)


def allocate(loops):
    counts = {}
    for c, _ in loops:
        counts[block_of(c)] = counts.get(block_of(c), 0) + 1
    cont = {k for k, v in counts.items() if v >= N_C}
    return cont, counts


def loop_polygon(c, b):
    cell = pyexadis.Cell(h=L*np.eye(3), is_periodic=[1, 1, 1])
    nodes, segs = insert_prismatic_loop('fcc', cell, [], [], b, R_LOOP, c, maxseg=STATE['maxseg'])
    return np.array(nodes)[:, :3], np.array(segs)


def network(loops, sources, keep):
    cell = pyexadis.Cell(h=L*np.eye(3), is_periodic=[1, 1, 1])
    nodes, segs = [], []
    for (c, b), k in zip(loops, keep):
        if k:
            nodes, segs = insert_prismatic_loop('fcc', cell, nodes, segs, b, R_LOOP, c, maxseg=STATE['maxseg'])
    for c, b, n in sources:
        nodes, segs = insert_frank_read_src(cell, nodes, segs, b, n, L_SRC, c, theta=90.0, numnodes=10)
    return DisNetManager(ExaDisNet(cell, np.array(nodes), np.array(segs)))


def conversion_ledger(loops, removed):
    """Net Burgers vector of each removed loop and its plastic-distortion content b (x) A n (M1)."""
    worst, total, P = 0., np.zeros((3, 3)), np.zeros((3, 3))
    for (c, b), r in zip(loops, removed):
        if not r:
            continue
        x, segs = loop_polygon(c, b)
        nb = np.zeros(3)
        A = np.zeros(3)
        for s in segs:
            i, j = int(s[0]), int(s[1]); bs = s[2:5]
            nb += bs if True else 0
            A += .5*np.cross(x[i], x[j])
        # net Burgers of a closed loop: every node has one incoming and one outgoing segment with the same b
        inflow = np.zeros((len(x), 3))
        for s in segs:
            inflow[int(s[0])] -= s[2:5]; inflow[int(s[1])] += s[2:5]
        worst = max(worst, float(np.abs(inflow).max()))
        bl = segs[0, 2:5]
        P += np.outer(bl, A)
        total += np.outer(bl, A)
    return worst, P, total


def friction_field(loops, cont, alpha, path):
    """Dispersed-barrier friction tau_f = alpha mu b sqrt(N d) on continuum blocks, 32^3 grid, in Pa."""
    bm = STATE['burgmag']
    vol_block = (L/BLOCKS*bm)**3
    counts = {}
    for c, _ in loops:
        k = block_of(c)
        if k in cont:
            counts[k] = counts.get(k, 0) + 1
    g = np.zeros((NGRID_FRIC,)*3)
    x = (np.arange(NGRID_FRIC) + .5)*L/NGRID_FRIC
    for i in range(NGRID_FRIC):
        for j in range(NGRID_FRIC):
            for k in range(NGRID_FRIC):
                blk = block_of(np.array([x[i], x[j], x[k]]))
                if blk in counts:
                    N = counts[blk]/vol_block
                    g[i, j, k] = alpha*STATE['mu']*bm*np.sqrt(N*2*R_LOOP*bm)
    with open(path, 'w') as f:
        f.write(f'{NGRID_FRIC}\n{NGRID_FRIC}\n{NGRID_FRIC}\n')
        np.savetxt(f, g.ravel(), fmt='%.6e')
    return float(g.max())


def longrange_check(loops, removed, sources):
    """RMS resolved shear, in the channel, of the removed loops' eigenstrain stress (FFT, 32^3), relative to applied."""
    n = 32; h = L/n
    beta = np.zeros((3, 3, n, n, n))
    for (c, b), r in zip(loops, removed):
        if not r:
            continue
        x, segs = loop_polygon(c, b)
        A = sum(.5*np.cross(x[int(s[0])], x[int(s[1])]) for s in segs)
        idx = tuple((np.floor(c/h).astype(int)) % n)
        beta[(slice(None), slice(None)) + idx] += np.outer(segs[0, 2:5], A)/h**3
    mu, nu = STATE['mu'], STATE['nu']; lam = 2*mu*nu/(1 - 2*nu)
    k = 2*np.pi*np.fft.fftfreq(n, d=h)
    K = np.array(np.meshgrid(k, k, k, indexing='ij'))
    eps_p = .5*(beta + beta.transpose(1, 0, 2, 3, 4))
    tau_p = lam*np.einsum('kk...->...', eps_p)[None, None]*np.eye(3)[:, :, None, None, None] + 2*mu*eps_p
    th = np.fft.fftn(tau_p, axes=(2, 3, 4))
    k2 = np.sum(K**2, axis=0); k2[0, 0, 0] = 1
    # displacement from div(C:grad u) = div(tau_p): isotropic Green function
    f = 1j*np.einsum('j...,ij...->i...', K, th)
    kf = np.einsum('i...,i...->...', K, f)
    u = (f/(mu*k2) - (lam + mu)/(mu*(lam + 2*mu))*K*kf/k2**2)*(-1)
    u[:, 0, 0, 0] = 0
    grad = 1j*np.einsum('j...,i...->ij...', K, u)
    eps = np.real(np.fft.ifftn(.5*(grad + grad.transpose(1, 0, 2, 3, 4)), axes=(2, 3, 4)))
    tp = np.real(np.fft.ifftn(th, axes=(2, 3, 4)))
    sig = lam*np.einsum('kk...->...', eps)[None, None]*np.eye(3)[:, :, None, None, None] + 2*mu*eps - tp
    z = (np.arange(n) + .5)*h
    chan = np.abs((z - Z0 + L/2) % L - L/2) > HW + 2*h
    rs = []
    for _, b, nn in sources:
        t = np.einsum('i,ij...,j->...', nn, sig, b)
        rs.append(np.sqrt(np.mean(t[:, :, chan]**2)))
    applied = SIGMA*abs(sources[0][2][2]*sources[0][1][2])
    return float(np.max(rs)/applied)


class Driver(SimulateNetwork):
    """Adds the swept-area plastic strain split into channel and wall regions (same rule for every method)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log, self.Ep, self.Ech, self.Ewall, self.t_obs = [], np.zeros(6), 0., 0., 0.
        self.progress = k.get('progress')

    def step_integrate(self, N, state):
        super().step_integrate(N, state)
        t = time.perf_counter()
        data = N.export_data(); cell = N.cell
        r = data['nodes']['positions']; sid = data['segs']['nodeids']; bb = data['segs']['burgers']
        rold = state['oldnodes_dict']['positions']
        r1 = r[sid[:, 0]]
        r2 = np.array(cell.closest_image(Rref=r1, R=r[sid[:, 1]]))
        r3 = np.array(cell.closest_image(Rref=r1, R=rold[sid[:, 0]]))
        r4 = np.array(cell.closest_image(Rref=r3, R=rold[sid[:, 1]]))
        n = .5*np.cross(r2 - r3, r1 - r4)
        ezz = n[:, 2]*bb[:, 2]/cell.volume()                     # (n (x) b)_zz, symmetric part
        zm = .25*(r1[:, 2] + r2[:, 2] + r3[:, 2] + r4[:, 2])
        wall = np.abs((zm - Z0 + L/2) % L - L/2) <= HW + 2*R_LOOP
        self.Ech += float(ezz[~wall].sum()); self.Ewall += float(ezz[wall].sum())
        self.t_obs += time.perf_counter() - t

    def step_end(self, N, state):
        self.Ep += state['dEp']
        if self.progress and len(self.log) % 200 == 0:
            with open(self.progress, 'a') as fh:
                fh.write(f"{state['time']:.4e} {time.perf_counter() - self.t0:.1f} {N.num_segments()}" + chr(10))
        self.log.append((state['time'], float(self.Ep[2]), self.Ech, self.Ewall,
                         time.perf_counter() - self.t0 - self.t_obs, N.num_segments()))

    def iterate(self, N, state):
        go = state.get('time', 0.) < T_END
        if go:
            self.tstep += 1
        return go


def main():
    method, seed = sys.argv[1], int(sys.argv[2])
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.
    out = sys.argv[4] if len(sys.argv) > 4 else '.'
    os.makedirs(out, exist_ok=True)
    tag = f'{method}_s{seed}' + (f'_a{alpha:g}' if method == 'H' else '')
    pyexadis.initialize()
    loops, sources = build(seed)
    cont, counts = allocate(loops)
    if method == 'D':
        keep = [True]*len(loops)
    elif method == 'H':
        keep = [block_of(c) not in cont for c, _ in loops]
    else:
        keep = [not (abs(c[2] - Z0) <= HW + 1e-9) for c, _ in loops]
    removed = [not k for k in keep]
    info = dict(method=method, seed=seed, alpha=alpha, loops_total=len(loops), loops_removed=int(sum(removed)),
                continuum_blocks=len(cont), blocks=BLOCKS**3, threads=int(os.environ.get('OMP_NUM_THREADS', 0)))
    if method == 'H':
        worst, P, total = conversion_ledger(loops, removed)
        info.update(ledger_burgers_max=worst, ledger_distortion_rel=float(np.abs(P - total).max()/max(np.abs(total).max(), 1e-300)),
                    distortion_content=P.tolist(), longrange_rel=longrange_check(loops, removed, sources))
        fpath = os.path.join(out, f'friction_{tag}.dat')
        info['tau_f_max'] = friction_field(loops, cont, alpha, fpath)
        mobility = MobilityLaw(mobility_law='FCC_0_FRIC', state=STATE, Medge=64103.0, Mscrew=64103.0, vmax=4000.0,
                               Fedge=0.0, Fscrew=0.0, Fscale=1.0, friction_field=fpath)
    else:
        mobility = MobilityLaw(mobility_law='FCC_0', state=STATE, Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
    N = network(loops, sources, keep)
    info['segments_initial'] = N.num_segments()
    calforce = CalForce(force_mode='DDD_FFT_MODEL', state=STATE, Ngrid=32, cell=N.cell)
    timeint = TimeIntegration(integrator='Trapezoid', state=STATE, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=STATE)
    topology = Topology(topology_mode='TopologyParallel', state=STATE, force=calforce, mobility=mobility)
    remesh = Remesh(remesh_rule='LengthBased', state=STATE)
    sim = Driver(calforce=calforce, mobility=mobility, timeint=timeint, collision=collision, topology=topology,
                 remesh=remesh, state=STATE, loading_mode='stress', progress=os.path.join(out, f'progress_{tag}.txt'),
                 applied_stress=np.array([0., 0., SIGMA, 0., 0., 0.]), max_step=10**9, burgmag=STATE['burgmag'])
    t0 = time.perf_counter()
    sim.run(N, STATE)
    wall = time.perf_counter() - t0
    info.update(wall_s=wall - sim.t_obs, observable_s=sim.t_obs, steps=len(sim.log), time_s=sim.log[-1][0], Ep=sim.Ep.tolist(),
                Ep_channel=sim.Ech, Ep_wall=sim.Ewall, log_columns=['time_s', 'Ep_zz_total', 'Ep_zz_channel', 'Ep_zz_wall', 'cpu_s', 'segments'],
                log=sim.log[::max(1, len(sim.log)//400)] + [sim.log[-1]], segments_final=N.num_segments(),
                density_final=dislocation_density(N, STATE['burgmag']))
    write_vtk(N, os.path.join(out, f'{tag}.vtk'))
    with open(os.path.join(out, f'{tag}.json'), 'w') as f:
        json.dump(info, f, indent=1, default=float)
    print(tag, 'wall', round(wall - sim.t_obs, 1), 'steps', len(sim.log), 'Ep_zz', sim.Ep[2], 'channel', sim.Ech, 'wallreg', sim.Ewall, 'segs', info['segments_initial'], flush=True)
    pyexadis.finalize()


if __name__ == '__main__':
    main()
