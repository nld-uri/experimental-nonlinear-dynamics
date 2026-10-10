"""SOD-based smooth projective noise reduction.

Python port of sodfilt.m (Appendix A.5). Flow generated data only: the
smooth decomposition needs a data-based derivative, which map generated
data does not have.

The optional nearest-neighbour strand bundle of Section 13.2, step 2, is
included here as the argument r. With r = 1 (the default) each point is
filtered from its own strand alone. With r > 1 the r-1 nearest
temporally uncorrelated strands join it, each is decomposed separately,
and the smoothed strands are averaged with weights that fall off with
distance from the base strand -- the reading the chapter's step 2(e)
states. The weight is w(delta) = a^4/(delta^4 + a^4), eq. (loclinw) of
Chapter 15, with a the distance from the base point to the farthest
(r-th) strand centre of its bundle, so the scale follows the local
density and the farthest strand gets half the weight of the base one.
It costs roughly r times as much and is worth it only on noisy records;
see the table in Section 13.2.

Copyright by David Chelidze. Python port for the second edition.
"""
import numpy as np
from scipy.spatial import cKDTree
from nlutil import embed, pad, strand_weights, unshift
from gsvd import smooth_project


def sodfilt(x, l, m, d, q, w=0, r=1, tw=None):
    """Return the filtered time series xf.

    x  : scalar time series
    l  : points per strand (even values are increased by one)
    m  : delay time for the reconstruction
    d  : embedding dimension (d < l)
    q  : projection dimension (q < d)
    w  : weighting along the strand -- 0 comb (default), 1 tent,
         2 exponential
    r  : strands per bundle (default 1, no bundle). r = 4 is a good
         choice on noisy records.
    tw : Theiler window excluding temporally correlated strands from the
         bundle. Defaults to (d-1)*m + 2l + 1, one strand span, so no
         neighbour overlaps the base strand.
    """
    x = np.asarray(x, float).ravel()
    nx = m*(d - 1)                       # shrinkage in reconstruction
    if l <= d:
        raise ValueError("make strand length l larger than the embedding "
                         "dimension d")
    if d <= q:
        raise ValueError("make subspace dimension q smaller than embedding "
                         "dimension d")
    y, n = embed(x, d, m)
    l = l//2                             # distance to the mid-point
    sl = 2*l + 1                         # strand length
    npd = l + nx                         # padding for the edge effects
    y, _ = pad(y, n, npd)
    npr = n + 2*npd
    wd = strand_weights(l, w)

    if r > 1:
        if tw is None:
            tw = nx + sl                 # one strand span
        cen = np.arange(l, npr - l)      # strand centres available
        tree = cKDTree(y[:, cen].T)

    Yh = np.zeros((d, npr))              # adjusted trajectory points
    wa = np.zeros(npr)                   # weight accumulated at each point
    for k in range(npr - sl + 1):
        if r == 1:
            Y = y[:, k:k + sl]
            b = Y.mean(axis=1, keepdims=True)      # strand bias vector
            Ya = smooth_project(Y - b, q) + b      # smooth projection
        else:
            base = k + l
            nq = min(cen.size, r + 4*tw)
            dis, idx = tree.query(y[:, base], k=nq)
            c = cen[np.atleast_1d(idx)]
            dis = np.atleast_1d(dis)
            keep = np.abs(c - base) > tw           # temporally uncorrelated
            c, dis = c[keep][:r - 1], dis[keep][:r - 1]
            c = np.concatenate([[base], c])
            dis = np.concatenate([[0.0], dis])
            ok = (c - l >= 0) & (c + l < npr)
            c, dis = c[ok], dis[ok]
            a = dis[-1]                            # farthest strand of the bundle
            Wn = a**4/(dis**4 + a**4) if a > 0 else np.ones(dis.size)
            acc = np.zeros((d, sl))
            for ci, wi in zip(c, Wn):
                S = y[:, ci - l:ci + l + 1]
                B = S.mean(axis=1, keepdims=True)
                acc += wi*(smooth_project(S - B, q) + B)
            Ya = acc/Wn.sum()
        Yh[:, k:k + sl] += Ya*wd
        wa[k:k + sl] += wd
    # trim the padding and normalize by the weight each point received,
    # so the first and last l points are weighted averages too
    Yh = Yh[:, l:npr - l]/wa[l:npr - l]
    return unshift(Yh, d, m, n, nx)
