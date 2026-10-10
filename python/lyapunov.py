"""Largest short-time Lyapunov exponent metrics.

Python port of lyapunov.m (Appendix A.6). Returns the mean logarithmic
divergence against step number; the exponent is the slope of the scaling
region, divided by the sampling interval.

With r > 1 the neighbor trajectories are averaged with the weight
w(delta) = a^4/(delta^4 + a^4), eq. (loclinw) of Chapter 15, where delta
is the distance from the fiducial point to the neighbor and a is the
distance to its r-th (farthest) neighbor. The weight is the same in
lyapunov.m. For r = 1 -- the value the header recommends -- the single
weight cancels and the result is independent of it.

Copyright by David Chelidze. Python port for the second edition.
"""
import numpy as np
from scipy.spatial import cKDTree


def lyapunov(x, d, m, N, r, n, w=None, rng=None):
    """Return (ns, lnd): the step numbers and the mean log divergence.

    x : scalar time series
    d : embedding dimension
    m : delay
    N : number of fiducial points (should be large)
    r : number of nearest neighbors per fiducial point (1 works)
    n : maximum number of steps ahead
    w : Theiler window; 0 keeps temporal correlations. Default d*m, and
        needed for flow data.
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, float).ravel()
    x = (x - x.mean())/x.std()
    nm = x.size - (d - 1)*m
    if w is None:
        w = d*m                     # open for debate
    N = min(2**12, nm) if not N else min(N, nm)
    y = np.array([x[np.arange(nm) + i*m] for i in range(d)])

    indx = rng.permutation(nm - n)[:N]          # fiducial points
    tree = cKDTree(y[:, :-n].T)
    npt = r + 2*w + 1
    dis, idx = tree.query(y[:, indx].T, k=npt)
    if w == 0:
        dis, idx = dis[:, 1:], idx[:, 1:]
    else:                                       # remove temporal correlations
        keep = np.abs(idx - idx[:, :1]) > w
        dis = np.array([dis[i][keep[i]][:r] for i in range(N)])
        idx = np.array([idx[i][keep[i]][:r] for i in range(N)])

    lnd = np.zeros(n)
    steps = np.arange(1, n + 1)
    for k in range(N):
        a = dis[k, -1]                          # r-th neighbor distance
        W = a**4/(dis[k]**4 + a**4) if a > 0 else np.ones(r)
        dist = np.zeros(n)
        for j in range(r):
            tmp = y[:, idx[k, j] + steps] - y[:, indx[k] + steps]
            dist = dist + W[j]*np.sqrt(np.sum(tmp*tmp, axis=0))
        lnd = lnd + np.log(dist/W.sum())
    return steps, lnd/N
