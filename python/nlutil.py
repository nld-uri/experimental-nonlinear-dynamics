"""nlutil -- shared pieces of the projective noise reduction routines.

Delay coordinate embedding, the edge padding both routines use, the
strand weighting, and the final shift-and-average. Kept in one place so
podfilt and sodfilt cannot drift apart, and so the padding convention is
stated once.

Copyright by David Chelidze. Python port for the second edition.
"""
import numpy as np
from scipy.spatial import cKDTree


def embed(x, d, m):
    """Delay coordinate embedding: y[i] = x[i : i+n] shifted by i*m."""
    x = np.asarray(x, float).ravel()
    n = x.size - m*(d - 1)
    return np.array([x[np.arange(n) + i*m] for i in range(d)]), n


def pad(y, n, npd):
    """Extend both ends by following the trajectory of a nearest neighbor
    of the first and last point, as podfilt.m and sodfilt.m do."""
    tree = cKDTree(y.T)
    _, i0 = tree.query(y[:, 0], k=npd)
    cand = np.atleast_1d(i0)
    cand = cand[cand > npd]
    if cand.size == 0:
        raise ValueError("no suitable left-hand neighbor for padding; "
                         "the record is too short for these parameters")
    ya = y[:, cand[0] - npd:cand[0]]
    _, i1 = tree.query(y[:, -1], k=npd)
    cand = np.atleast_1d(i1)
    cand = cand[cand < n - npd]
    if cand.size == 0:
        raise ValueError("no suitable right-hand neighbor for padding; "
                         "the record is too short for these parameters")
    yb = y[:, cand[0] + 1:cand[0] + 1 + npd]
    return np.hstack([ya, y, yb]), tree


def strand_weights(l, w):
    """The three weightings of sodfilt.m, as a row vector.
    0 comb (no averaging along the strand), 1 tent, 2 exponential.
    They need not be normalized: the caller divides by the weight each
    point accumulates."""
    if w == 1:
        wd = np.concatenate([np.arange(0, l + 1), np.arange(l - 1, -1, -1)])
    elif w == 2:
        wd = np.exp(-np.concatenate([np.arange(l, -1, -1),
                                     np.arange(1, l + 1)]))
    else:
        wd = np.concatenate([np.zeros(l), [1.0], np.zeros(l)])
    return np.asarray(wd, float)


def unshift(Yh, d, m, n, nx):
    """Average the d estimates of each sample. The offset is (d-k)*tau:
    each point sits one coordinate further back for each step down in k."""
    xf = np.array([Yh[k, m*(d - 1 - k):Yh.shape[1] - m*k]
                   for k in range(d)])
    return xf.mean(axis=0)
