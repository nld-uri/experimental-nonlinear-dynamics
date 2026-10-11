"""Local linear model prediction.

Python port of loclin.m (Appendix A.7).

Copyright by David Chelidze. Python port.
"""
import numpy as np
from scipy.spatial import cKDTree


def loclin(ym, yq, n, N, w=0, tw=0, iq=None):
    """Predict N steps ahead from yq using local linear models.

    ym : (d, M) reconstructed trajectory used for modeling
    yq : (d,) point to start the prediction from
    n  : number of nearest neighbors per model
    N  : number of steps to predict
    w  : 1 to weight neighbors by distance, anything else for no weights.
         The weight is w(delta) = a^4/(delta^4 + a^4), eq. (loclinw) of
         Chapter 15, with a the distance from the query point to its n-th
         (last requested) neighbor, so the scale follows the local density
         and the n-th neighbor gets half the weight of a coincident point.
    tw : Theiler window (default 0: no exclusion). Used only with iq.
    iq : column of ym that yq is (in-sample query), or None. At step k the
         columns j with |j - (iq + k)| <= tw are not used as neighbors.

    Returns yp, shape (d, N+1), with yp[:, 0] = yq.
    """
    ym = np.atleast_2d(np.asarray(ym, float))
    d, M = ym.shape
    yq = np.asarray(yq, float).ravel().copy()
    tree = cKDTree(ym[:, :-1].T)
    excl = iq is not None and tw > 0
    kq = min(n + 2*tw + 1, M - 1) if excl else n   # extra neighbors to discard
    yp = np.zeros((d, N + 1))
    yp[:, 0] = yq
    for k in range(N):
        dis, idx = tree.query(yq, k=kq)
        dis, idx = np.atleast_1d(dis), np.atleast_1d(idx)
        if excl:                                    # Theiler exclusion
            keep = np.abs(idx - (iq + k)) > tw
            dis, idx = dis[keep][:n], idx[keep][:n]
        Yn = np.vstack([ym[:, idx], np.ones(n)])    # (d+1, n) design X~
        Ynp1 = ym[:, idx + 1]                       # (d, n) images
        a = dis[-1]                                 # n-th neighbor distance
        if w == 1 and a > 0:
            sw = np.sqrt(a**4/(dis**4 + a**4))      # square roots of weights
        else:
            sw = np.ones(n)
        # weighted least squares on sqrt(W) X~^T directly (SVD), no Gram matrix
        B = np.linalg.lstsq(sw[:, None]*Yn.T, sw[:, None]*Ynp1.T,
                            rcond=None)[0].T        # (d, d+1) = [A, b]
        yq = B @ np.append(yq, 1.0)
        yp[:, k + 1] = yq
    return yp
