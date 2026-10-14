"""Section 74: the defect types specified in theory Sections 2, 7 and 28 but not implemented before.

- Stacking faults: a dissociated dislocation as two Shockley partials bounding a faulted ribbon. The partials are slip
  eigenstrains on a glide plane, their elastic energy is computed by FFT (plane strain for edge, antiplane for screw
  components), and the ribbon carries the areal fault energy.
- Disclinations: wedge disclinations as rotational incompatibility sources of the plane-strain Airy stress function.
- Twin boundaries: a low-energy special boundary in a pairwise multiphase-field energy.
- Antisites: the Bragg-Williams long-range order field of an equiatomic two-sublattice alloy, whose antisite fraction is
  (1 - eta)/2 per sublattice, with antiphase boundaries between domains of opposite order.
"""

import numpy as np
from .solvers import project_simplex


def wavevectors(n, length):
    k = 2*np.pi*np.fft.fftfreq(n, d=length/n)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    return KX, KY


def eigenstrain_energy(eps_xy, gamma_zy, length, mu, nu):
    """Elastic energy per unit length of a periodic body with eigen shear eps*_xy (plane strain) and eigen shear
    gamma*_zy (antiplane), both on an n x n grid of cell area (length/n)^2; mean strain relaxes freely."""
    n = eps_xy.shape[0]; area = (length/n)**2; KX, KY = wavevectors(n, length); k2 = KX**2+KY**2; k2[0, 0] = 1.
    # Antiplane: E = mu/2 sum |g|^2 kx^2/k^2 (the k=0 mode is removed by the free mean strain).
    g = np.fft.fftn(gamma_zy)*area
    screw = .5*mu*np.sum(np.abs(g)**2*KX**2/k2)/length**2
    # Plane strain isotropic: for eigen shear e (eps*_xy = eps*_yx = e) the relaxed energy density in Fourier space is
    # 2 mu e^2 * [1 - (kx^2 + ky^2)/k^2 + 4 kx^2 ky^2/((1-nu) k^4)] ... derived from the isotropic Green function below.
    e = np.fft.fftn(eps_xy)*area
    lam = 2*mu*nu/(1-2*nu)
    # Stress of the eigenstrain: s = C:eps* = 2 mu eps*; displacement u = -i A^-1 (s k); strain eps = sym(i k u).
    # Energy = 1/2 sum (eps - eps*):C:(eps - eps*); evaluate per mode with 2x2 algebra.
    sxy = 2*mu*e
    fx, fy = sxy*KY, sxy*KX                       # traction-like source (s k)
    a11 = (lam+2*mu)*KX**2+mu*KY**2; a22 = (lam+2*mu)*KY**2+mu*KX**2; a12 = (lam+mu)*KX*KY
    det = a11*a22-a12**2; det[0, 0] = 1.
    ux = (a22*fx-a12*fy)/det; uy = (a11*fy-a12*fx)/det   # u = A^-1 (s k) (times -i, absorbed below)
    exx = KX*ux; eyy = KY*uy; exy = .5*(KX*uy+KY*ux)  # eps = sym(k u), consistent with the dropped factors of i
    dxx, dyy, dxy = exx, eyy, exy-e
    energy_density = .5*(lam*np.abs(dxx+dyy)**2+2*mu*(np.abs(dxx)**2+np.abs(dyy)**2+2*np.abs(dxy)**2))
    energy_density[0, 0] = 0.
    edge = np.sum(energy_density)/length**2
    return float(screw+edge)


def partial_burgers(beta, b=1.):
    """Shockley partials of a total vector at character angle beta (radians) to the line; components (x = edge, z = screw)."""
    bp = b/np.sqrt(3)
    b1 = bp*np.array([np.sin(beta+np.pi/6), np.cos(beta+np.pi/6)])
    b2 = bp*np.array([np.sin(beta-np.pi/6), np.cos(beta-np.pi/6)])
    return b1, b2


def ribbon_slip(n, d_cells, beta, b=1.):
    """Slip (x, z) along the glide row of a periodic dipole of dissociated dislocations: 0 | b1 | b | b1 | 0."""
    b1, b2 = partial_burgers(beta, b); s = np.zeros((n, 2))
    x1 = n//4; X = x1+n//2
    s[x1:x1+d_cells] = b1; s[x1+d_cells:X] = b1+b2; s[X:X+d_cells] = b1
    return s


def ribbon_energy(n, length, d_cells, beta, fault_energy, mu, nu, b=1.):
    """Elastic plus fault energy per unit length of the dipole cell, as a function of the ribbon width in cells."""
    s = ribbon_slip(n, d_cells, beta, b); h = length/n; j0 = n//2
    eps = np.zeros((n, n)); gam = np.zeros((n, n))
    eps[:, j0] = s[:, 0]/(2*h); gam[:, j0] = s[:, 1]/h
    return eigenstrain_energy(eps, gam, length, mu, nu)+2*fault_energy*d_cells*h


def analytic_separation(beta, fault_energy, mu, nu, b=1.):
    """Isotropic equilibrium separation of two parallel partials: F = mu/(2 pi d)[(b1.x)(b2.x) + (b1 x xi).(b2 x xi)/(1-nu)]."""
    b1, b2 = partial_burgers(beta, b)
    return mu/(2*np.pi*fault_energy)*(b1[1]*b2[1]+b1[0]*b2[0]/(1-nu))


def airy_stress(eta, length, mu, nu):
    """Plane-strain stresses from the Airy function with incompatibility source eta: lap^2 chi = E' eta, E' = 2 mu/(1-nu)."""
    n = eta.shape[0]; area = (length/n)**2; KX, KY = wavevectors(n, length); k2 = KX**2+KY**2; k2[0, 0] = 1.
    chi = 2*mu/(1-nu)*np.fft.fftn(eta)*area/k2**2; chi[0, 0] = 0.
    inv = lambda f: np.real(np.fft.ifftn(f))/area
    return dict(xx=inv(-KY**2*chi), yy=inv(-KX**2*chi), xy=inv(KX*KY*chi))


def gaussian(n, length, x0, y0, width):
    x = (np.arange(n)+.5)*length/n; X, Y = np.meshgrid(x, x, indexing='ij')
    g = np.exp(-((X-x0)**2+(Y-y0)**2)/(2*width**2)); return g/(g.sum()*(length/n)**2)


def pairwise_energy(grid, phases, a, W=1., k=1.):
    """sum_{i<j} a_ij [W phi_i phi_j - k grad phi_i . grad phi_j]; equals the Section 21 grain energy when every a_ij = 1."""
    total = 0.; n = len(phases)
    grads = [np.gradient(p, grid.dx[0], grid.dy[0]) for p in phases]
    for i in range(n):
        for j in range(i+1, n):
            total += a[i][j]*grid.integrate(W*phases[i]*phases[j]-k*(grads[i][0]*grads[j][0]+grads[i][1]*grads[j][1]))
    return float(total)


def pairwise_force(grid, phases, a, W=1., k=1., triple=0.):
    n = len(phases); force = np.zeros_like(phases)
    laps = [grid.lap(p) for p in phases]
    for i in range(n):
        for j in range(n):
            if i != j:force[i] += a[i][j]*(W*phases[j]+k*laps[j])
    if triple and n == 3:  # Section 74.5 post hoc: lambda phi_0 phi_1 phi_2 suppresses a third phase inside a binary boundary
        for i in range(3):force[i] += triple*np.prod([phases[j] for j in range(3) if j != i], axis=0)
    return force


def sector_phases(grid, sectors, width=1.):
    """Three phases occupying given angular sectors (degrees, counterclockwise from phase 0 at angle 0), smoothed."""
    from scipy.ndimage import gaussian_filter
    phi = np.mod(np.arctan2(grid.Y, grid.X), 2*np.pi); edges = np.r_[0, np.cumsum(np.radians(sectors))]
    labels = np.searchsorted(edges, phi, side='right')-1; labels = np.clip(labels, 0, 2)
    phases = np.array([(labels == i).astype(float) for i in range(3)])
    phases = np.array([gaussian_filter(p, width/grid.dx[0], mode='nearest') for p in phases])
    return phases/phases.sum(axis=0)


def pairwise_junction(grid, a, W=1., k=1., tolerance=2e-5, max_iterations=40000, sectors=None, triple=0.):
    """Stationary three-phase junction for pairwise coefficients a (Section 74.3). The boundary is held at 120 degrees,
    or (post hoc, 74.5) at given sectors; triple > 0 adds the ghost-phase suppression term."""
    if sectors is None:
        angles = np.arange(3)*2*np.pi/3
        scores = np.array([np.cos(t)*grid.X+np.sin(t)*grid.Y for t in angles])
        phases = np.exp(1.5*(scores-scores.max(axis=0))); phases /= phases.sum(axis=0)
    else:phases = sector_phases(grid, sectors)
    fixed = np.zeros(grid.shape, bool); fixed[[0, -1], :] = True; fixed[:, [0, -1]] = True; boundary = phases.copy()
    def project(v):
        p = project_simplex(v); p[:, fixed] = boundary[:, fixed]; return p
    step = .1*min(grid.dx[0], grid.dy[0])**2/max(max(r) for r in a)
    residual = np.inf
    for it in range(1, max_iterations+1):
        f = pairwise_force(grid, phases, a, W, k, triple); new = project(phases-step*f)
        if it % 200 == 0:
            residual = float(np.sqrt(np.mean(((phases-new)/step)**2)))
            if residual < tolerance:phases = new;break
        phases = new
    return phases, dict(iterations=it, residual=residual, converged=residual < tolerance)


def phase_sectors(grid, phases, radius, samples=7200):
    """Angular sector (degrees) occupied by each phase on a ring around the equal-fraction point."""
    from scipy.ndimage import map_coordinates
    from scipy.optimize import root
    score = np.sum((phases-1/3)**2, axis=0); m = 4; score[:m] = score[-m:] = np.inf; score[:, :m] = score[:, -m:] = np.inf
    guess = np.array(np.unravel_index(np.argmin(score), score.shape), float)
    diff = lambda q: np.array([map_coordinates(p, q[:, None], order=1)[0] for p in phases])[1:]-map_coordinates(phases[0], q[:, None], order=1)[0]
    y, x = root(diff, guess).x
    theta = np.arange(samples)*2*np.pi/samples
    coords = np.array([y+radius/grid.dy[0]*np.sin(theta), x+radius/grid.dx[0]*np.cos(theta)])
    labels = np.argmax([map_coordinates(p, coords, order=1) for p in phases], axis=0)
    return [float(np.mean(labels == i)*360) for i in range(3)]


def young_sectors(s01, s02, s12):
    """Sector of each phase from force balance: the sector of phase 0 lies between boundaries 01 and 02."""
    ang = lambda a, b, c: np.degrees(np.arccos((c**2-a**2-b**2)/(2*a*b)))
    return [ang(s01, s02, s12), ang(s01, s12, s02), ang(s02, s12, s01)]


def bw_free_energy(eta, T, Tc=1.):
    """Bragg-Williams free energy per site in units of k Tc: -eta^2/2 + (T/2)[(1+eta)ln(1+eta) + (1-eta)ln(1-eta)]."""
    e = np.clip(eta, -1+1e-15, 1-1e-15)
    return -e**2/2+T/(2*Tc)*((1+e)*np.log(1+e)+(1-e)*np.log(1-e))


def bw_force(eta, T, Tc=1.):
    e = np.clip(eta, -1+1e-15, 1-1e-15)
    return -e+T/(2*Tc)*np.log((1+e)/(1-e))


def bw_order(T, Tc=1.):
    """Self-consistent Bragg-Williams order eta = tanh(Tc eta / T) (zero above Tc), by bisection."""
    if T >= Tc:return 0.
    lo, hi = 1e-12, 1.
    for _ in range(200):
        mid = .5*(lo+hi)
        if mid-np.tanh(Tc*mid/T) < 0:lo = mid
        else:hi = mid
    return .5*(lo+hi)


def relax_order(T, eta0=.5, mobility=1., dt=.05, steps=200000, tol=1e-14):
    eta = eta0
    for _ in range(steps):
        new = eta-dt*mobility*bw_force(eta, T)
        if abs(new-eta) < tol:return new
        eta = new
    return eta


def apb_profile(T, kappa=1., n=801, width=40., dt=None, steps=400000, tol=1e-13):
    """Relax a 1D antiphase boundary between eta = -eta0 and +eta0 (pinned ends); returns x, eta and its excess energy."""
    eta0 = bw_order(T); x = np.linspace(-width/2, width/2, n); h = x[1]-x[0]
    eta = eta0*np.tanh(x/2); f0 = bw_free_energy(eta0, T); dt = dt or .2*h*h/kappa
    for _ in range(steps):
        lap = np.zeros_like(eta); lap[1:-1] = (eta[2:]-2*eta[1:-1]+eta[:-2])/h**2
        new = eta.copy(); new[1:-1] = eta[1:-1]-dt*(bw_force(eta[1:-1], T)-kappa*lap[1:-1])
        if np.max(abs(new-eta)) < tol:eta = new;break
        eta = new
    grad = np.diff(eta)/h; mid = .5*(eta[1:]+eta[:-1])
    energy = float(np.sum((bw_free_energy(mid, T)-f0+kappa*grad**2/2)*h))
    return x, eta, energy


def apb_analytic(T, kappa=1., samples=200001):
    """Exact 1D interface energy: integral over eta of sqrt(2 kappa (f(eta) - f(eta0)))."""
    eta0 = bw_order(T); e = np.linspace(-eta0, eta0, samples)
    g = np.sqrt(np.maximum(2*kappa*(bw_free_energy(e, T)-bw_free_energy(eta0, T)), 0))
    return float(np.trapezoid(g, e))


def cubic_stiffness(c11, c12, c44):
    """Fourth-order stiffness tensor of a cubic crystal in its cube axes."""
    d = np.eye(3); C = c12*np.einsum('ij,kl->ijkl', d, d) + c44*(np.einsum('ik,jl->ijkl', d, d) + np.einsum('il,jk->ijkl', d, d))
    return C + (c11 - c12 - 2*c44)*np.einsum('ij,ik,il->ijkl', d, d, d)


def energy_factor(C, line, samples=720):
    """Barnett-Lothe prelogarithmic energy factor K of a straight line along `line`: the circle average of
    (mm) - (mn)(nn)^-1(nm) with (ab)_jk = a_i C_ijkl b_l. The energy per length is b.K.b ln(R/r0)/(4 pi), and two
    parallel lines on one glide plane at distance d repel with b1.K.b2/(2 pi d). Isotropic: K = mu (screw), mu/(1-nu) (edge)."""
    t = np.asarray(line, float)/np.linalg.norm(line); a = np.array([1., 0, 0]) if abs(t[0]) < .9 else np.array([0, 1., 0])
    m0 = np.cross(t, a); m0 /= np.linalg.norm(m0); n0 = np.cross(t, m0); K = np.zeros((3, 3))
    ab = lambda a, b: np.einsum('i,ijkl,l->jk', a, C, b)
    for w in np.linspace(0, 2*np.pi, samples, endpoint=False):
        m = np.cos(w)*m0 + np.sin(w)*n0; n = np.cross(t, m); K += ab(m, m) - ab(m, n)@np.linalg.solve(ab(n, n), ab(n, m))
    return K/samples


def anisotropic_separation(C, line, b1, b2, fault_energy):
    """Equilibrium separation of two parallel partials on one glide plane: b1.K.b2/(2 pi gamma)."""
    return float(np.asarray(b1)@energy_factor(C, line)@np.asarray(b2)/(2*np.pi*fault_energy))
