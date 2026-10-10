"""Is cd.mat the raw record behind psw_fatigue_2005.mat?

The file `cd.mat` holds one variable, `strn`: 7,080,000 strain samples. This
script tests whether it is the run that produced the 1728 x 405 tracking matrix
of `data/psw_fatigue_2005.mat`, by rebuilding the two-stage analysis of
Chapter 17 from the raw record with the book's own characteristic-distance
features and comparing every number with the supplied matrix.

Run from python/. The raw record is located by _psw2005.raw_path(); see
data/README.md for where to put it. Takes about a minute.
"""
import sys
import numpy as np
from scipy.io import loadmat
from scipy.linalg import subspace_angles
from gsvd import sod
from _psw2005 import tracking, raw, raw_path, NHEALTHY

FS, NW, NPTS, DIM, TAU, L, SUB = 160.0, 1728, 4096, 5, 6, 300, 2


def sod_rows(M):
    return sod(M.T - M.T.mean(axis=1, keepdims=True))


def stages(F, v1, v2, t, n0=NHEALTHY):
    """The two stages of Section 17.4. Returns a dict of the reported numbers."""
    Fc = F - F[:n0].mean(axis=0)
    _, Psi, lam, _ = sod_rows(F[:n0])
    soc = Fc @ Psi
    V = np.c_[v1, v2][:n0]
    A = np.c_[soc[:n0, :2], np.ones(n0)]
    G = np.linalg.lstsq(A, V, rcond=None)[0]
    R2 = 1 - ((V - A @ G).var(0)/V.var(0))
    B = np.linalg.lstsq(Fc[:n0], V - V.mean(0), rcond=None)[0]
    ang = subspace_angles(np.linalg.qr(Fc[:n0] @ B)[0],
                          np.linalg.qr(soc[:n0, :2])[0])
    Fb = Fc @ Psi[:, 2:]
    _, Psi2, lam2, _ = sod_rows(Fb)
    q = (Fb - Fb[:n0].mean(axis=0)) @ Psi2[:, 0]
    q = q*np.sign(np.polyfit(t, q, 1)[0])
    q = (q - q.min())/(q.max() - q.min())
    m = t >= 10.0
    return dict(sov=lam[:3], R2=R2, ang=ang, sov2=lam2[:2],
                r=np.corrcoef(t[m], 1/q[m])[0, 1])


def report(tag, d):
    print("  %-22s SOV %7.1f %6.1f (next %4.1f) | R2 %.4f %.4f | angles "
          "%.4f %.4f | SOV' %6.1f vs %5.1f | r %.4f"
          % (tag, d["sov"][0], d["sov"][1], d["sov"][2], d["R2"][0], d["R2"][1],
             d["ang"][0], d["ang"][1], d["sov2"][0], d["sov2"][1], d["r"]))


def main():
    Y, t, v1, v2 = tracking()
    x = raw()
    print("raw record: %d samples = %.4f h at %.0f Hz" % (len(x), len(x)/FS/3600, FS))
    print("  quantization step x 65536 = %.4f (10 -> 16 bit over +-5 V)"
          % (np.diff(np.unique(x)).min()*65536))
    print("  the tracking matrix needs %d x %d = %d samples, leaving %d (%.1f s)"
          % (NW, NPTS, NW*NPTS, len(x) - NW*NPTS, (len(x) - NW*NPTS)/FS))

    W = x[:NW*NPTS].reshape(NW, NPTS)
    M = NPTS - (DIM - 1)*TAU
    rng = np.random.default_rng(0)
    pool = np.concatenate([np.stack([W[l, j*TAU:j*TAU + M] for j in range(DIM)],
                                    1)[::30] for l in range(0, NW, 30)])
    lo, hi = pool.min(0), pool.max(0)
    ref = 0.5*(lo + hi) + (2*rng.random((L, DIM)) - 1)*0.5*(hi - lo)*1.5
    F = np.empty((NW, L))
    for l in range(NW):
        Yl = np.stack([W[l, j*TAU:j*TAU + M] for j in range(DIM)], 1)[::SUB]
        F[l] = np.sqrt(((Yl[:, None, :] - ref[None])**2).sum(-1)).mean(0)

    print("\ntwo-stage analysis, supplied PSW features against rebuilt "
          "characteristic distances (L = %d):" % L)
    report("supplied PSW (405)", stages(Y, v1, v2, t))
    report("rebuilt chardist", stages(F, v1, v2, t))
    print("\nIf the two rows agree, cd.mat is the run behind the tracking matrix.")


if __name__ == "__main__":
    main()
