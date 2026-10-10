"""Average pointwise dimension, fixed distance and fixed mass.

Python port of pwdim.m (Appendix A.3).

Copyright by David Chelidze. Python port for the second edition.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def pwdim(x, m, d, k=500, N=None, w=0, rng=None):
    """Return (lnD, lnN), each (k, 2): column 0 fixed distance, 1 fixed mass.

    x : scalar time series
    m : delay
    d : embedding dimension
    k : number of distance partitions, or points inside a ball
    N : maximum number of averages (default min(2**12, n))
    w : Theiler window; 0 keeps temporal correlations

    lnD[:, 0] = ln(eps),                    lnN[:, 0] = <ln mu(eps)>
    lnD[:, 1] = <ln(ball holding n points)>, lnN[:, 1] = ln(n/N)
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, float).ravel()
    n = x.size - m*(d - 1)              # maximum size possible
    N = min(2**12, n) if not N else min(N, n)
    y = np.array([x[np.arange(n) + i*m] for i in range(d)])     # (d, n)

    tree = cKDTree(y.T)
    npt = k + 1 + 2*w                   # neighbors to look up
    sel = rng.permutation(n)[:N]
    dis, idx = tree.query(y[:, sel].T, k=npt)
    if w == 0:
        dis = dis[:, 1:]                # drop the zero self-distance
    else:                               # remove temporal correlations
        keep = np.abs(idx - idx[:, :1]) > w
        dis = np.array([dis[i][keep[i]][:k] for i in range(N)])

    # fixed mass
    lnDm = np.log(dis).mean(axis=0)
    lnNm = digamma(np.arange(1, k + 1)) - np.log(N)   # Grassberger (1985)

    # fixed distance
    dm = dis.max()
    edgs = np.concatenate([[0.0], np.exp(np.linspace(
        lnDm[0], np.mean([lnDm[-1], np.log(dm)]), k))])
    acc = np.zeros((N, k))
    for i in range(N):
        h, _ = np.histogram(dis[i], bins=edgs)
        ln = np.log(np.cumsum(h), where=np.cumsum(h) > 0,
                    out=np.zeros(k))
        acc[i] = ln                     # zero count counts as zero
    nz = (acc != 0).sum(axis=0)
    lnNd = np.divide(acc.sum(axis=0), nz, out=np.zeros(k),
                     where=nz > 0) - np.log(N)
    lnN = np.column_stack([lnNd, lnNm])
    lnD = np.column_stack([np.log(edgs[1:]), lnDm])
    return lnD, lnN
