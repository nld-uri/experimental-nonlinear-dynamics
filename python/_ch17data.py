"""Shared simulation and feature code for the Chapter 17 figures.

Every figure of Sections 17.4 and 17.6 that is computed from simulated data
uses this module, so the R\"ossler and Duffing records, the reference-point
sampling region, the characteristic distance and the SOD tracking are the
same everywhere.

Systems
-------
rossler_windows : one steady-state segment per value of a, integrated in
                  parallel across the windows with a fixed-step RK4.
                  b = 0.6, c = 6.0 (the non-standard set the chapter uses),
                  t_s = 0.06.
duffing_windows : the same for the two-well Duffing oscillator
                  xdd + 0.25 xd - x + x^3 = f cos(t), t_s = pi/36.

Features
--------
refpoints   : L reference points drawn uniformly in the bounding box of the
              pooled trajectory, enlarged by ENLARGE about its centre.  This
              is the sampling region used for every simulated example of the
              chapter; see the caption of Figure 17.5.
chardist    : characteristic distance from each window to each reference point
charpos     : characteristic position (m columns per reference point)
track       : centre the tracking matrix, take its SOD, return SOVs and SOCs

Copyright by David Chelidze.
"""
import numpy as np
from gsvd import sod

ENLARGE = 1.5           # bounding box of the pooled trajectory, times this


# ------------------------------------------------------------------ systems
def _rk4(u, f, dt, n, keep=False):
    out = np.empty((n,) + u.shape) if keep else None
    for i in range(n):
        k1 = f(u)
        k2 = f(u + 0.5*dt*k1)
        k3 = f(u + 0.5*dt*k2)
        k4 = f(u + dt*k3)
        u = u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        if keep:
            out[i] = u
    return (u, out) if keep else u


def rossler_windows(a, n=5000, dt=0.06, b=0.6, c=6.0, ntrans=20000, seed=0):
    """Return (N, n, 3) steady-state segments, one per entry of a."""
    a = np.asarray(a, float).ravel()
    N = a.size
    aa = a[:, None]

    def f(u):
        x, y, z = u[:, 0:1], u[:, 1:2], u[:, 2:3]
        return np.hstack([-y - z, x + aa*y, b + z*(x - c)])

    rng = np.random.default_rng(seed)
    u = np.tile(np.array([1.0, 1.0, 1.0]), (N, 1)) + 0.01*rng.standard_normal((N, 3))
    u = _rk4(u, f, dt, ntrans)
    _, traj = _rk4(u, f, dt, n, keep=True)      # (n, N, 3)
    return np.transpose(traj, (1, 0, 2))


def duffing_windows(fa, n=4000, dt=np.pi/36, gamma=0.25, ntrans=3600, seed=0):
    """Return (N, n) scalar x(t) segments of the two-well Duffing oscillator,
    one per forcing amplitude in fa.  ntrans steps = 50 forcing periods."""
    fa = np.asarray(fa, float).ravel()
    N = fa.size
    ff = fa[:, None]
    t = np.array([0.0])

    def make_f(t0):
        def f(u, tt):
            x, v = u[:, 0:1], u[:, 1:2]
            return np.hstack([v, -gamma*v + x - x**3 + ff*np.cos(tt)])
        return f

    rng = np.random.default_rng(seed)
    u = np.tile(np.array([1.0, 0.0]), (N, 1)) + 0.01*rng.standard_normal((N, 2))
    g = make_f(0.0)
    tt = 0.0
    for _ in range(ntrans):
        k1 = g(u, tt)
        k2 = g(u + 0.5*dt*k1, tt + 0.5*dt)
        k3 = g(u + 0.5*dt*k2, tt + 0.5*dt)
        k4 = g(u + dt*k3, tt + dt)
        u = u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        tt += dt
    out = np.empty((n, N))
    for i in range(n):
        k1 = g(u, tt)
        k2 = g(u + 0.5*dt*k1, tt + 0.5*dt)
        k3 = g(u + 0.5*dt*k2, tt + 0.5*dt)
        k4 = g(u + dt*k3, tt + dt)
        u = u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        tt += dt
        out[i] = u[:, 0]
    return out.T


# ------------------------------------------------------------------ features
def refpoints(traj, L, rng, enlarge=ENLARGE):
    """L reference points, uniform in the bounding box of the pooled
    trajectory enlarged by `enlarge` about its centre.  traj is (M, m)."""
    lo, hi = traj.min(axis=0), traj.max(axis=0)
    mid, half = 0.5*(lo + hi), 0.5*(hi - lo)*enlarge
    return mid + (2*rng.random((L, traj.shape[1])) - 1)*half


def embed_windows(w, d, tau):
    """(N, n) scalar windows -> (N, M, d) delay vectors, forward delays."""
    M = w.shape[1] - (d - 1)*tau
    return np.stack([w[:, j*tau:j*tau + M] for j in range(d)], axis=2)


def chardist(win, ref):
    """Characteristic distance matrix Y (N, L).

    win : (N, n, m) embedded windows;  ref : (L, m) reference points.
    """
    N, n, m = win.shape
    L = ref.shape[0]
    Y = np.empty((N, L))
    for l in range(N):
        Y[l] = np.sqrt(((win[l][:, None, :] - ref[None, :, :])**2).sum(-1)).mean(0)
    return Y


def charpos(win, ref):
    """Characteristic position matrix Y (N, L*m): the mean unit vector from
    each reference point to the trajectory, m columns per reference point."""
    N, n, m = win.shape
    L = ref.shape[0]
    Y = np.empty((N, L*m))
    for l in range(N):
        dv = win[l][:, None, :] - ref[None, :, :]           # (n, L, m)
        r = np.sqrt((dv**2).sum(-1))[..., None]
        Y[l] = (dv/r).mean(0).ravel()
    return Y


def track(Y, tol=1e-8):
    """Centre the columns of the tracking matrix and take its SOD.

    Step 5 of the procedure box: the centred matrix is projected onto its
    numerically significant principal components before the GSVD, since a
    tracking matrix built partly from far reference points is close to rank
    deficient.  Returns (lam, Q) with the SOVs lam in decreasing order and
    the SOCs Q as columns, smoothest first.
    """
    Yc = Y - Y.mean(axis=0, keepdims=True)
    _, sv, Vt = np.linalg.svd(Yc, full_matrices=False)
    k = int(np.sum(sv > tol*sv[0]))
    Z = Yc @ Vt[:k].T
    Phi, Psi, lam, Q = sod(Z.T)           # sod wants (d, N): features down rows
    return lam, Q


def corr(u, v):
    u = np.asarray(u, float).ravel(); v = np.asarray(v, float).ravel()
    return float(np.corrcoef(u, v)[0, 1])


# ------------------------------------------- the two simulated examples, fixed
# Rossler: N windows, a swept as a quarter sine from 0.1 to 0.4.
R_N, R_NPTS, R_DT, R_L, R_SEED = 786, 5000, 0.06, 50, 0
# Duffing: N windows, f swept linearly from 0.35 to 0.4.
D_N, D_NPTS, D_DT, D_L, D_TAU, D_DIM, D_SEED = 1001, 12000, np.pi/36, 400, 53, 4, 0


def rossler_drift(N=R_N):
    """a_l, a quarter sine from 0.1 to 0.4 across the N windows."""
    return 0.1 + 0.3*np.sin(np.pi*np.arange(N)/(2*(N - 1)))


def rossler_setup(L=R_L, seed=R_SEED, N=R_N, n=R_NPTS):
    """Return (a, traj, ref) for the Rossler example."""
    a = rossler_drift(N)
    traj = rossler_windows(a, n=n, dt=R_DT)
    pool = traj.reshape(-1, 3)[::max(1, traj.shape[0]*n//20000)]
    ref = refpoints(pool, L, np.random.default_rng(seed))
    return a, traj, ref


def duffing_setup(L=D_L, seed=D_SEED, N=D_N, n=D_NPTS):
    """Return (f, W, ref) for the Duffing example; W is (N, M, D_DIM)."""
    f = np.linspace(0.35, 0.4, N)
    w = duffing_windows(f, n=n, dt=D_DT)
    W = embed_windows(w, D_DIM, D_TAU)
    pool = W.reshape(-1, D_DIM)[::max(1, W.shape[0]*W.shape[1]//20000)]
    ref = refpoints(pool, L, np.random.default_rng(seed))
    return f, W, ref
