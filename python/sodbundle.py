"""sodbundle -- sodfilt with the nearest-neighbour strand bundle (r > 1).

The chapter's algorithm has an optional step the shipped sodfilt does
not implement: for each base point, collect r-1 temporally uncorrelated
nearest neighbours, take their strands too, and combine. Two readings of
what "combine" means, both tested here because the text does not settle
it:

  bundle="points"    The chapter's wording -- "the filtered point is
                     determined by the weighted average of points in the
                     smoothed strands of the bundle". SOD each of the r
                     strands separately, then average the smoothed
                     POINTS, weighted by distance from the base point.
                     This is a spatial average across nearby trajectory
                     segments as well as a projection.

  bundle="subspace"  The reading usual in projective methods: the
                     bundle only estimates a better smooth subspace.
                     Debias each strand, stack them, take ONE SOD of the
                     stack, and project the base strand alone. The base
                     point is never pulled toward its neighbours.

Weighting is the book's own, eq. (loclinw) of Chapter 15,
w(delta) = a^4/(delta^4 + a^4) with a the distance from the base point to
the farthest (r-th) strand centre of its bundle, as in sodfilt, lyapunov
and loclin.

r = 1 reduces exactly to sodfilt in both modes.
"""
import numpy as np
from scipy.spatial import cKDTree
from nlutil import embed, pad, strand_weights, unshift
from gsvd import smooth_project, sod


def sodfilt_bundle(x, l, m, d, q, w=0, r=1, bundle="points", theiler=None):
    x = np.asarray(x, float).ravel()
    nx = m*(d - 1)
    if l <= d:
        raise ValueError("make strand length l larger than d")
    if d <= q:
        raise ValueError("make q smaller than d")
    y, n = embed(x, d, m)
    l = l//2
    sl = 2*l + 1
    npd = l + nx
    yp, _ = pad(y, n, npd)
    npr = n + 2*npd
    wd = strand_weights(l, w)

    if theiler is None:
        theiler = nx + sl          # exclude the base strand's own span

    centers = np.arange(l, npr - l)          # strand centres available
    tree = cKDTree(yp[:, centers].T) if r > 1 else None

    Yh = np.zeros((d, npr))
    wa = np.zeros(npr)
    for k in range(npr - sl + 1):
        base = k + l                          # centre index in yp
        if r == 1:
            Y = yp[:, k:k + sl]
            b = Y.mean(axis=1, keepdims=True)
            Ya = smooth_project(Y - b, q) + b
        else:
            # r-1 temporally uncorrelated neighbour centres
            nq = min(len(centers), r + 4*theiler)
            dist, idx = tree.query(yp[:, base], k=nq)
            cand = centers[np.atleast_1d(idx)]
            dist = np.atleast_1d(dist)
            keep = np.abs(cand - base) > theiler
            cand, dist = cand[keep][:r - 1], dist[keep][:r - 1]
            cen = np.concatenate([[base], cand])
            dis = np.concatenate([[0.0], dist])
            ok = (cen - l >= 0) & (cen + l < npr)
            cen, dis = cen[ok], dis[ok]
            a = dis[-1]                       # farthest strand of the bundle
            Wn = a**4/(dis**4 + a**4) if a > 0 else np.ones(dis.size)
            strands = [yp[:, c - l:c + l + 1] for c in cen]
            biases = [S.mean(axis=1, keepdims=True) for S in strands]
            if bundle == "subspace":
                stack = np.hstack([S - B for S, B in zip(strands, biases)])
                Phi, Psi, _, _ = sod(stack)
                Y0 = strands[0] - biases[0]
                Ya = Phi[:, :q] @ (Psi[:, :q].T @ Y0) + biases[0]
            else:                              # "points"
                acc = np.zeros((d, sl))
                for S, B, ww in zip(strands, biases, Wn):
                    acc += ww*(smooth_project(S - B, q) + B)
                Ya = acc/Wn.sum()
        Yh[:, k:k + sl] += Ya*wd
        wa[k:k + sl] += wd
    Yh = Yh[:, l:npr - l]/wa[l:npr - l]
    return unshift(Yh, d, m, n, nx)
