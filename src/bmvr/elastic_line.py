"""Section 77: planar dislocation lines with non-singular isotropic elastic self-interaction (energy and exact gradient)."""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
_G4 = np.polynomial.legendre.leggauss(4)
_G8 = np.polynomial.legendre.leggauss(8)


def _gauss(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.tensor((x+1)/2), torch.tensor(w/2)


def straight_self(length, a):
    """Exact double integral of 1/R_a over a straight segment of length l: 2[l asinh(l/a) - sqrt(l^2+a^2) + a]."""
    return 2*(length*torch.asinh(length/a)-torch.sqrt(length**2+a**2)+a)


def pair_energy(p0, p1, q0, q1, b1, b2, mu, nu, a, order):
    """Interaction term between segment sets P and Q (batched), both with in-plane positions (..., 3)."""
    s, w = _gauss(order)
    tp, tq = p1-p0, q1-q0
    x = p0[:, None, :]+s[None, :, None]*tp[:, None, :]; y = q0[:, None, :]+s[None, :, None]*tq[:, None, :]
    r = x[:, :, None, :]-y[:, None, :, :]; Ra = torch.sqrt((r*r).sum(-1)+a*a)
    ww = (w[:, None]*w[None, :])[None]
    bdp, bdq = (b1*tp).sum(-1), (b2*tq).sum(-1)
    cp, cq = torch.cross(b1.expand_as(tp), tp, dim=-1), torch.cross(b2.expand_as(tq), tq, dim=-1)
    first = (bdp*bdq)[:, None, None]/Ra
    cr_p = (cp[:, None, None, :]*r).sum(-1); cr_q = (cq[:, None, None, :]*r).sum(-1)
    T = (cp*cq).sum(-1)[:, None, None]/Ra-cr_p*cr_q/Ra**3
    return (ww*(mu/(8*np.pi)*first+mu/(8*np.pi*(1-nu))*T)).sum()


def line_energy(nodes, b, mu=1., nu=.324, a=1.):
    """Non-singular self-energy of an open planar polyline (nodes (N,2) torch tensor), b in-plane (2,)."""
    P = torch.cat([nodes, torch.zeros(len(nodes), 1)], 1); bb = torch.cat([b, torch.zeros(1)])
    p0, p1 = P[:-1], P[1:]; n = len(p0); t = p1-p0; l = torch.linalg.norm(t, dim=1); u = t/l[:, None]
    screw = (u@bb)**2; edge = (torch.cross(bb.expand_as(u), u, dim=-1)**2).sum(-1)
    self_part = (mu/(8*np.pi)*screw+mu/(8*np.pi*(1-nu))*edge)*straight_self(l, a)
    i, j = np.triu_indices(n, 1); adj = (j-i) == 1
    E = self_part.sum()
    for mask, order in ((adj, 8), (~adj, 4)):
        if mask.any():
            ii, jj = torch.tensor(i[mask]), torch.tensor(j[mask])
            E = E+2*pair_energy(p0[ii], p1[ii], p0[jj], p1[jj], bb, bb, mu, nu, a, order)
    return E


def parallel_interaction(d, angle, length=400., a=1., mu=1., nu=.324, n=80):
    """Interaction energy of two parallel straight lines (length, n segments each) at in-plane separation d."""
    x = torch.linspace(-length/2, length/2, n+1)
    b = torch.tensor([np.cos(angle), np.sin(angle), 0.])
    A = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], 1); B = A+torch.tensor([0., d, 0.])
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij'); i, j = torch.tensor(i.ravel()), torch.tensor(j.ravel())
    # every segment of A against every segment of B; the cross term of two distinct lines carries factor 2 (W_12 = 2 x 1/2)
    return float(pair_energy(A[:-1][i], A[1:][i], B[:-1][j], B[1:][j], b, b, mu, nu, a, 8)*2)


def area(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    return -.5*(x*torch.roll(y, -1)-torch.roll(x, -1)*y).sum()


def redistribute(nodes):
    seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]; t = np.linspace(0, s[-1], len(nodes))
    return np.stack([np.interp(t, s, nodes[:, 0]), np.interp(t, s, nodes[:, 1])], 1)


def bow(L, tau, b=(1., 0.), n=41, a=1., mu=1., nu=.324, steps=20000):
    """Overdamped bowing of a pinned segment under resolved shear tau: nodal velocity = force per unit length.

    Returns (stable, height/L): stable when the height stops changing (relative change < 1e-7 over 200 steps),
    unstable when the line passes height L (well beyond the critical, near-semicircular configuration).
    """
    x = np.linspace(-L/2, L/2, n); nodes = np.stack([x, 1e-3*L*(1-(2*x/L)**2)], 1); bt = torch.tensor(b)
    h = L/(n-1); gamma = mu*np.dot(b, b)/(4*np.pi*(1-nu))*np.log(L/a); dt = .2*h*h/gamma; last = 0.
    for k in range(steps):
        X = torch.tensor(nodes, requires_grad=True)
        F = line_energy(X, bt, mu, nu, a)-tau*float(np.linalg.norm(b))*area(X)
        g, = torch.autograd.grad(F, X); g = g.numpy(); g[[0, -1]] = 0.
        seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1); w = np.r_[seg[0], (seg[:-1]+seg[1:]), seg[-1]]/2
        nodes = nodes-dt*g/w[:, None]
        if k % 20 == 19:nodes = redistribute(nodes)
        height = nodes[:, 1].max()/L
        if height > 1.:return False, float(height)
        if k % 200 == 199:
            if abs(height-last) < 1e-7*max(height, 1e-3):return True, float(height)
            last = height
    return True, float(nodes[:, 1].max()/L)


def critical_stress(L, b=(1., 0.), a=1., mu=1., nu=.324, iterations=14, n=41):
    gamma = mu*np.linalg.norm(b)**2/(4*np.pi)*np.log(L/a); lt = 2*gamma/(np.linalg.norm(b)*L)
    lo, hi = .2*lt, 3*lt
    for _ in range(iterations):
        mid = .5*(lo+hi); stable, h = bow(L, mid, b, n, a, mu, nu)
        if stable:lo = mid
        else:hi = mid
    return .5*(lo+hi), lt
