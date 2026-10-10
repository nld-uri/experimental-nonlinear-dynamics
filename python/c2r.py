"""Correlation sum from a scalar time series.

Python port of c2r.m (Appendix A.2). Euclidean distances, temporal
correlations removed by a Theiler window.

Copyright by David Chelidze. Python port for the second edition.
"""
import numpy as np


def c2r(x, d, m, n=None, r=32, p=None, q=None, rng=None):
    """Return (lnr, lnc): ln of the ball sizes and of the correlation sum.

    x : scalar time series
    d : embedding dimension
    m : delay time
    n : number of reconstructed points to use (None for the maximum)
    r : number of length scales, or an array of ball sizes to use
    p : Theiler window; 0 to keep temporal correlations. Default d*m,
        and only needed for flow data.
    q : if given, populate the attractor with q random points instead of
        removing temporal correlations (p is then ignored)
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, float).ravel()
    x = (x - x.mean())/x.std()          # zero mean, unit variance
    nm = x.size - (d - 1)*m             # maximum number of points possible
    if n is None:
        n = nm
    elif n > nm:
        print("WARNING: Max number of embedded vectors exceeded.\n"
              "ACTION:  Continuing with max number of vectors possible.")
        n = nm
    if p is None:
        p = d*m                         # still open for debate

    # reconstruct the phase space
    if q is not None:
        indx = rng.permutation(n)[:q]   # random points from the attractor
        n, p = q, 0                     # no temporal correlations to remove
    else:
        indx = np.arange(n)
    y = np.array([x[indx + i*m] for i in range(d)])      # (d, n)

    # correlation sum
    npt = n - p
    r = np.atleast_1d(np.asarray(r, float))
    auto = r.size == 1
    if auto:
        nbins = int(r[0])
        c = np.zeros(nbins)
    else:
        r = r**2                        # work in squared distances
        c = np.zeros(r.size)
    for i in range(npt - 1):
        j = np.arange(i + p + 1, n)
        dist = np.sum((y[:, j] - y[:, i:i + 1])**2, axis=0)
        if i == 0 and auto:
            r = np.exp(np.linspace(np.log(dist.min()/npt),
                                   np.log(dist.max() + dist.min()), nbins))
        h, _ = np.histogram(dist, bins=np.concatenate([[0.0], r]))
        c = c + np.cumsum(h)
    with np.errstate(divide="ignore"):  # empty bins give -inf, as in MATLAB
        lnc = np.log(2*c/npt/(npt - 1)) # ln of the correlation sum
    lnr = np.log(r)/2                   # ln of the ball sizes
    return lnr, lnc
