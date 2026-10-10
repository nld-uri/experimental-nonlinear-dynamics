"""Recurrence plot and RQA measures (Chapter 18). NumPy only; O(M^2) memory.

    R = recmat(X, eps, w)                 recurrence matrix with Theiler window w
    eps = eps_for_rr(X, rr, w)            threshold giving recurrence rate rr
    P, Pv = line_hist(R, w)               diagonal / vertical line-length histograms
    m = rqa(R, w, lmin=2, vmin=2)         dict RR, DET, L, Lmax, ENTR, LAM, TT
    K2, Pc, ells = k2_from_lines(P, dt, lmin, lmax)  correlation entropy from P(l)

X is (M, d): one delay vector per row. Distances are Euclidean.
"""
import numpy as np


def _dist(X):
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] < X.shape[1]:
        X = X.T
    sq = np.sum(X*X, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2*X@X.T
    return np.sqrt(np.maximum(D2, 0.0))


def eps_for_rr(X, rr, w=0):
    """Threshold eps such that the fraction rr of admissible pairs (|i-j|>w) recur."""
    D = _dist(X); M = D.shape[0]
    i, j = np.triu_indices(M, k=w+1)
    return np.quantile(D[i, j], rr)


def recmat(X, eps, w=0):
    D = _dist(X); M = D.shape[0]
    R = D < eps
    idx = np.arange(M)
    R[np.abs(idx[:, None]-idx[None, :]) <= w] = False
    return R


def _runs(v):
    """Lengths of runs of True in a boolean vector."""
    v = np.concatenate(([False], v, [False])).astype(int)
    d = np.diff(v)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def line_hist(R, w=0):
    M = R.shape[0]
    P = np.zeros(M+1, int); Pv = np.zeros(M+1, int)
    for k in range(w+1, M):                       # upper triangle diagonals
        for L in _runs(np.diagonal(R, k)):
            P[L] += 1
    P *= 2                                         # symmetric: count both triangles
    for j in range(M):
        for L in _runs(R[:, j]):
            Pv[L] += 1
    return P, Pv


def rqa(R, w=0, lmin=2, vmin=2):
    M = R.shape[0]
    P, Pv = line_hist(R, w)
    ell = np.arange(M+1)
    nadm = (M-w)*(M-w-1)                          # admissible ordered pairs
    RR = R.sum()/nadm
    tot = np.sum(ell*P)
    DET = np.sum(ell[lmin:]*P[lmin:])/tot if tot else np.nan
    nl = P[lmin:].sum()
    L = np.sum(ell[lmin:]*P[lmin:])/nl if nl else np.nan
    Lmax = ell[P > 0].max() if P.any() else 0
    p = P[lmin:]/nl if nl else P[lmin:]
    ENTR = -np.sum(p[p > 0]*np.log(p[p > 0])) if nl else np.nan
    totv = np.sum(ell*Pv)
    LAM = np.sum(ell[vmin:]*Pv[vmin:])/totv if totv else np.nan
    nv = Pv[vmin:].sum()
    TT = np.sum(ell[vmin:]*Pv[vmin:])/nv if nv else np.nan
    return dict(RR=RR, DET=DET, L=L, Lmax=Lmax, ENTR=ENTR, LAM=LAM, TT=TT, P=P, Pv=Pv)


def k2_from_lines(P, dt, lmin, lmax):
    """Correlation entropy from the cumulative diagonal-line distribution
    P^c(l) = sum_{l'>=l} P(l') ~ exp(-K2 l dt) (Faure & Korn 1998). Least-squares
    slope of ln P^c over lmin <= l <= lmax; returns K2 (per unit time), P^c, l."""
    ell = np.arange(len(P))
    Pc = np.cumsum(P[::-1])[::-1].astype(float)
    sel = (ell >= lmin) & (ell <= lmax) & (Pc > 0)
    slope = np.polyfit(ell[sel]*dt, np.log(Pc[sel]), 1)[0]
    return -slope, Pc, ell
