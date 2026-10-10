"""Does the spatial-decorrelation step of Section spatd change the answer?

Kennel and Abarbanel (Phys. Rev. E 66, 026209, Appendix) write the delay
vectors as the rows of an N x (d+1) matrix A, take A = U D V^T, and set

    T = sqrt(N) V D^{-1},      B = A T Q,      Q = H(x - |x| e_{d+1}),

where x is the LAST ROW of T and H(z) = I - 2 z z^T / |z|^2 is a Householder
reflection.  T whitens: (1/N) (AT)^T (AT) = I.  Q is orthogonal and is built so
that T Q has zeros along its whole bottom row except the last entry, which makes
the first d coordinates of B functions of the first d coordinates of A alone.

Three variants are compared with doing nothing.  Y here is (d+1, M), one delay
vector per column, so A = Y^T and the transform is applied as Z = (T Q)^T Y.

    ka            Z = (T Q)^T Y, the published construction
    householder   Z = Q U^T Y, an orthogonal rotation alone (no whitening)
    whiten        Z = sqrt(M) Sigma^{-1} U^T Y, whitening alone (no rotation)

Run from python/. About a minute.
"""
import numpy as np
from fnn import embed, nnsearch
import _ch10data as C


def transform(Y, how):
    """Y is (d+1, M): one delay vector per column."""
    if how == "none":
        return Y
    U, s, _ = np.linalg.svd(Y - Y.mean(1, keepdims=True), full_matrices=False)
    M = Y.shape[1]
    W = np.diag(1.0/s)*np.sqrt(M) @ (U.T @ Y)      # whitened coordinates
    if how == "whiten":
        return W
    if how == "ka":
        # T = sqrt(M) U Sigma^{-1} in this column convention; x is its last row.
        x = np.sqrt(M)*U[-1, :]/s
        Q = _house(x)
        return Q.T @ W
    u = U[:, -1]                                   # thin direction of the tube
    Q = _house(u)
    return Q @ (U.T @ Y)


def _house(x):
    """H(z) with z = x - |x| e_last, so that x H = |x| e_last."""
    e = np.zeros(len(x)); e[-1] = 1.0
    v = x - np.linalg.norm(x)*e
    nv = v @ v
    return np.eye(len(x)) - (2.0/nv)*np.outer(v, v) if nv > 1e-300 else np.eye(len(x))


def fnn_frac(x, tau, dmax, w, how, r=10.0, s=0.5):
    """The composite fraction R_nn(d; r, s) after the given transform."""
    x = (np.asarray(x, float).ravel() - np.mean(x))/np.std(x)
    N = x.size
    out = np.zeros(dmax)
    for d in range(1, dmax + 1):
        n = N - d*tau
        Y = embed(x, d + 1, tau, n).T               # (d+1, n)
        Yb = transform(Y, how)
        P = Yb[:d].T.copy()                         # first d coordinates
        q = Yb[d]                                   # the (d+1)-th
        J, D = nnsearch(P, w, 1)
        j, eps = J[:, 0], D[:, 0]
        dlt = np.abs(q - q[j])
        out[d-1] = np.mean((dlt > r*eps) | (dlt > s))
    return out


def read(f, thr=0.02):
    k = np.flatnonzero(f < thr)
    return int(k[0] + 1) if len(k) else -1


if __name__ == "__main__":
    # The step exists to repair a too-small lag, so the decisive test is at
    # small tau, where the reconstruction is a thin tube along the hyperdiagonal.
    print("Composite fraction R_nn(d; 10, 0.5), Lorenz, Delta t = 0.02.")
    print("The AMI minimum is tau = 8; the true embedding dimension is 3.\n")
    print("  tau  transform       d=1      d=2      d=3      d=4      d=5   d_E")
    for tau in (1, 2, 3, 5, 8):
        x = C.record("Lorenz"); w = max(3*tau, 10)
        for h in ("none", "ka", "householder", "whiten"):
            F = fnn_frac(x, tau, 6, w, h)
            print("  %-4d %-13s %s   %d"
                  % (tau, h, " ".join("%8.4f" % v for v in F[:5]), read(F)))
        print()
    for name in ("Lorenz", "Duffing"):
        x = C.record(name); tau, w = C.tau_w(name)
        print("%s at its own delay (tau = %d, w = %d):" % (name, tau, w))
        hs = ("none", "ka", "householder", "whiten")
        F = {h: fnn_frac(x, tau, 8, w, h) for h in hs}
        print("  d_E:  none %d   ka %d   householder %d   whiten %d"
              % tuple(read(F[h]) for h in hs))
