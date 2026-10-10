"""Average mutual information.

Python port of ami.m (Appendix A.1.1). Same arguments, same return, same
binning. NumPy only. The three binning helpers are public because
ami_x.py uses them too, so the auto and cross routines cannot drift
apart.

One deliberate difference: MATLAB extends the outer bin edges to -Inf
and +Inf, which numpy.histogram will not accept. The outer edges are
widened by one bin instead, which includes the same points.

Copyright by David Chelidze. Python port for the second edition.
"""
import numpy as np


def bin_edges(v, nb):
    """Uniform bin edges over the range of v, outer edges widened."""
    lo, hi = float(np.min(v)), float(np.max(v))
    b = (hi - lo)/nb
    e = lo + b*np.arange(nb + 1)
    e[0] -= b                   # MATLAB uses -Inf here
    e[-1] += b                  # and +Inf here
    return e


def binned_p(v, e):
    """Non-zero binned probabilities."""
    h, _ = np.histogram(v, bins=e)
    p = h/h.sum()
    return p[p > 0]


def joint_p(u, v, eu, ev):
    """Non-zero joint binned probabilities."""
    h, _, _ = np.histogram2d(u, v, bins=[eu, ev])
    p = h/h.sum()
    return p[p > 0]


def ami(x, k, nb=32):
    """Average mutual information of x with itself, delays 0 to k.

    x  : scalar time series
    k  : maximum delay to consider
    nb : number of bins (default 32)

    Returns Ixx, length k+1, in bits. Ixx[0] is the entropy of x, since
    I(X;X) = H(X) -- a useful check on any binning.
    """
    x = np.asarray(x, float).ravel()
    n = x.size
    xx = bin_edges(x, nb)
    Px = binned_p(x, xx)
    Ixx = np.zeros(k + 1)
    for i in range(k + 1):
        Pxx = joint_p(x[:n - i], x[i:], xx, xx)
        Ixx[i] = np.sum(Pxx*np.log2(Pxx)) - 2*np.sum(Px*np.log2(Px))
    return Ixx
