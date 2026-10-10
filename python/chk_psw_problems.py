"""The numbers quoted in Problems 16.7, 17.7 and 18.7.

Every value stated as an expected answer in those three problems is
produced here, from data/psw_fatigue_2005.mat alone. Run from python/;
about a minute.

Copyright by David Chelidze. Written for the second edition.
"""
import numpy as np
import scipy.linalg as la
from scipy.spatial.distance import pdist, squareform
import _psw2005 as P

H = 1200                                  # the healthy stage of Section mt_exp


def sod(A):
    """Smooth orthogonal decomposition of A (rows = observations).

    Solves (A^T A) psi = lam^2 (D^T D) psi with D the first difference
    along rows, which is the generalized eigenvalue problem of
    Chapter 13 with nu = 1. Returns the values in decreasing order and
    the corresponding vectors as columns.
    """
    A = A - A.mean(0)
    D = np.diff(A, axis=0)
    w, V = la.eig(A.T @ A, D.T @ D)
    w = np.real(w)
    k = np.argsort(-w)
    return w[k], np.real(V[:, k])


def recover(A, v1, v2):
    """R^2 of each voltage on the leading two smooth coordinates."""
    lam, W = sod(A)
    S = (A - A.mean(0)) @ W[:, :2]
    X = np.column_stack([S, np.ones(A.shape[0])])
    out = []
    for v in (v1, v2):
        b, *_ = np.linalg.lstsq(X, v, rcond=None)
        r = v - X @ b
        out.append(1.0 - r @ r/((v - v.mean()) @ (v - v.mean())))
    return lam, S, out


def det_rqa(R, lmin=2, w=1):
    """Determinism of a recurrence matrix, Theiler window w."""
    M = R.shape[0]
    tot = dt = 0
    for k in range(-M + 1, M):
        if abs(k) <= w:
            continue
        d = np.diag(R, k)
        if d.size == 0:
            continue
        idx = np.flatnonzero(np.diff(np.concatenate(([0], d, [0]))))
        L = idx[1::2] - idx[0::2]
        tot += L.sum()
        dt += L[L >= lmin].sum()
    return dt/tot if tot else 0.0


if __name__ == "__main__":
    Y, t, v1, v2 = P.tracking()
    print("tracking matrix %d x %d, %.2f h\n" % (Y.shape + (t[-1],)))

    print("Problem 17.7(1): healthy stage, H = %d" % H)
    lam, S, R2 = recover(Y[:H], v1[:H], v2[:H])
    print("  five largest SOVs %s" % np.round(lam[:5], 1))
    print("  R2 on the two voltages %.4f %.4f\n" % tuple(R2))

    print("Problem 17.7(2): sensitivity to the healthy-stage cut")
    V = np.column_stack([v1 - v1.mean(), v2 - v2.mean()])
    for h in (800, 1000, 1200, 1400, 1600, 1728):
        lam, S, R2 = recover(Y[:h], v1[:h], v2[:h])
        ang = la.subspace_angles(S, V[:h])
        print("  H=%4d  SOV %7.1f %6.1f   R2 %.3f %.3f   angles %.2f %.2f rad"
              % (h, lam[0], lam[1], R2[0], R2[1], ang[0], ang[1]))
    print()

    print("Problem 17.7(4): how many features are needed")
    rng = np.random.default_rng(1)
    for L in (5, 10, 20, 60, 150, 405):
        n = 20 if L < 405 else 1
        a = np.empty((n, 2))
        for j in range(n):
            c = rng.choice(405, L, replace=False) if L < 405 else np.arange(405)
            a[j] = recover(Y[:H][:, c], v1[:H], v2[:H])[2]
        print("  L=%3d  R2 %.3f +- %.3f  /  %.3f +- %.3f"
              % (L, a[:, 0].mean(), a[:, 0].std(),
                 a[:, 1].mean(), a[:, 1].std()))
    print()

    print("Problem 16.7(2): row-shuffle surrogates, leading SOV")
    lam0 = sod(Y[:H])[0][0]
    ens = np.array([sod(Y[rng.permutation(Y.shape[0])][:H])[0][0]
                    for _ in range(19)])
    print("  data %.1f, surrogate max %.2f over %d surrogates\n"
          % (lam0, ens.max(), ens.size))

    print("Problem 18.7(1): recurrence of the tracking rows")
    A = (Y - Y.mean(0))/Y.std(0)
    Dm = squareform(pdist(A))
    eps = np.percentile(Dm, 10)
    R = (Dm < eps).astype(np.uint8)
    np.fill_diagonal(R, 0)
    print("  eps %.2f   RR %.3f   DET %.3f" % (eps, R.mean(), det_rqa(R)))
    print("Problem 18.7(4): recurrence of later blocks with the first 300")
    f = [(t[i], (Dm[:300, i:i+50] < eps).mean()) for i in range(0, 1700, 200)]
    print("  " + "  ".join("%.1fh:%.2f" % z for z in f))
