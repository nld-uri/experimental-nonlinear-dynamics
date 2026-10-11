"""Cross average mutual information.

Python port of ami_x.m (Appendix A.1.2). The binning helpers come from
ami.py, so the auto and cross routines cannot drift apart.

Copyright by David Chelidze. Python port.
"""
import numpy as np
from ami import bin_edges, binned_p, joint_p


def ami_x(x, y, k, nbx=32, nby=None):
    """Cross average mutual information of x and y, delays -k to k.

    x, y : the two scalar time series
    k    : maximum delay to consider
    nbx  : number of bins for x (default 32)
    nby  : number of bins for y (default nbx)

    Returns Ixy, length 2k+1, in bits, with Ixy[k] the zero-delay value.
    Index k+i holds the positive delay i (y lagged behind x) and k-i the
    negative one, matching ami_x.m.
    """
    if nby is None:
        nby = nbx
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    n, m = x.size, y.size
    xx, yy = bin_edges(x, nbx), bin_edges(y, nby)
    Px, Py = binned_p(x, xx), binned_p(y, yy)
    hx = np.sum(Px*np.log2(Px))
    hy = np.sum(Py*np.log2(Py))
    Ixy = np.zeros(2*k + 1)
    j = min(m, n)
    Pxy = joint_p(x[:j], y[:j], xx, yy)
    Ixy[k] = np.sum(Pxy*np.log2(Pxy)) - hx - hy
    for i in range(1, k + 1):
        j = min(m, n - i)                       # negative delays
        Pxy = joint_p(x[i:i + j], y[:j], xx, yy)
        Ixy[k - i] = np.sum(Pxy*np.log2(Pxy)) - hx - hy
        j = min(n, m - i)                       # positive delays
        Pxy = joint_p(x[:j], y[i:i + j], xx, yy)
        Ixy[k + i] = np.sum(Pxy*np.log2(Pxy)) - hx - hy
    return Ixy
