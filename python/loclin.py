"""Local linear model prediction.

Python port of loclin.m (Appendix A.7).

Copyright by David Chelidze. Python port for the second edition.
"""
import numpy as np
from scipy.spatial import cKDTree


def loclin(ym, yq, n, N, w=0):
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

    Returns yp, shape (d, N+1), with yp[:, 0] = yq.
    """
    ym = np.atleast_2d(np.asarray(ym, float))
    d = ym.shape[0]
    yq = np.asarray(yq, float).ravel().copy()
    tree = cKDTree(ym[:, :-1].T)
    yp = np.zeros((d, N + 1))
    yp[:, 0] = yq
    for k in range(N):
        dis, idx = tree.query(yq, k=n)
        dis = np.atleast_1d(dis)
        idx = np.atleast_1d(idx)
        Yn = np.vstack([ym[:, idx], np.ones(n)])        # (d+1, n)
        Ynp1 = ym[:, idx + 1]                           # (d, n)
        z = np.append(yq, 1.0)
        if w == 1:
            a = dis[-1]                                 # n-th neighbor distance
            Wd = np.diag(a**4/(dis**4 + a**4)) if a > 0 else np.eye(n)
            A = np.linalg.lstsq((Yn @ Wd @ Yn.T).T,
                                (Ynp1 @ Wd @ Yn.T).T, rcond=None)[0].T
        else:
            A = np.linalg.lstsq((Yn @ Yn.T).T,
                                (Ynp1 @ Yn.T).T, rcond=None)[0].T
        yq = A @ z
        yp[:, k + 1] = yq
    return yp
