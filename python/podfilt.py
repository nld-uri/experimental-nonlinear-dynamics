"""POD-based projective noise reduction.

Python port of podfilt.m (Appendix A.4). Works on both flow and map
generated data.

Copyright by David Chelidze. Python port.
"""
import numpy as np
from nlutil import embed, pad, unshift


def podfilt(x, l, m, d, q, f=1, u=1):
    """Return the filtered time series xf.

    x : scalar time series
    l : points in the neighbor cloud, or the strand length
    m : delay time
    d : embedding dimension (d < l)
    q : projection dimension (q < d)
    f : 1 (default) for the neighbor-cloud scheme, 0 for strands
    u : curvature handling.
        1 (default) leave the curvature alone;
        0 the TEMPORAL correction, c'_k = (b_{k-1} + b_{k+1})/2;
        2 the SPATIAL correction of eq. (curvcorr), c'_k the mean of the
          mass centres of the l spatial neighbours of point k.
        Both apply 2 c_k - c'_k, and differ only in which points supply
        c'_k. The chapter specifies the spatial one. See
        chk_curvcorr.py for the measured comparison.
        NOTE: the cloud scheme with u = 0 diverges under repeated
        application -- a temporal second difference of the neighbourhood
        mass centres amplifies the jitter that comes from the neighbour
        set changing discontinuously from point to point. u = 2 does not
        have that failure mode, since it averages over the same
        neighbour set the projection already uses. The strand branch is
        fine with u = 0, and that is what the Chapter 14 figures used.
    """
    x = np.asarray(x, float).ravel()
    nx = m*(d - 1)
    if d <= q:
        raise ValueError("make subspace dimension q smaller than embedding "
                         "dimension d")
    n = x.size
    y, nrp = embed(x, d, m)
    R = np.eye(d)
    R[0, 0] = R[-1, -1] = 1000.0         # weighting toward the interior

    if f == 1:
        npd = 1 + nx
    else:
        if l <= d:
            raise ValueError("make strand length l larger than d")
        l = l//2 + 1                     # strand middle point
        sl = 2*l - 1
        npd = l + nx
    yp, tree = pad(y, nrp, npd)
    npr = nrp + 2*npd

    Yh = np.zeros((d, npr))
    if f == 1:                           # neighbor-cloud projection
        b = np.zeros((d, npr))
        _, pqi = tree.query(yp.T, k=l)
        for k in range(npr):
            Y = y[:, pqi[k]]
            b[:, k] = Y.mean(axis=1)     # crude mean filter
            U, S, Vt = np.linalg.svd(((Y - b[:, k:k + 1]).T @ R),
                                     full_matrices=False)
            C = (Vt.T[:, :q]*S[:q]) @ U[:, :q].T
            Yh[:, k] = np.linalg.solve(R, C[:, 0]) + b[:, k]
        if u == 2:                       # spatial correction, eq. (curvcorr)
            # pqi indexes the UNPADDED trajectory y; b is indexed over the
            # padded one, where y[:, j] sits at column j + npd.
            cp = b[:, pqi + npd].mean(axis=2)
            Yh = Yh + b - cp
        Yh = Yh[:, 1:npr - 1]
        if u == 0:                       # temporal correction
            Yh = Yh + b[:, 1:-1] - (b[:, :-2] + b[:, 2:])/2
    else:                                # strand projection
        b = np.zeros((d, npr - sl + 1))
        wd = np.concatenate([np.arange(0, l), np.arange(l - 2, -1, -1)])
        wa = np.zeros(npr)
        for k in range(npr - sl + 1):
            Y = yp[:, k:k + sl]
            b[:, k] = Y.mean(axis=1)
            U, S, Vt = np.linalg.svd(((Y - b[:, k:k + 1]).T @ R),
                                     full_matrices=False)
            C = (Vt.T[:, :q]*S[:q]) @ U[:, :q].T
            Yh[:, k:k + sl] += (np.linalg.solve(R, C) + b[:, k:k + 1])*wd
            wa[k:k + sl] += wd
        # trim and normalize by the accumulated weight, then curvature
        Yh = Yh[:, l:npr - l]/wa[l:npr - l]
        if u == 0:
            Yh = Yh + b[:, 1:-1] - (b[:, :-2] + b[:, 2:])/2
    return unshift(Yh, d, m, n, nx)
