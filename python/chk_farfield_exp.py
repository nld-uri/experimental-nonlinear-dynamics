"""The far-field question on the measured record (Chapter 17, Section 17.4).

Section sec:mt_farfield expands the characteristic distance for a reference
point far from the attractor and finds that it reports only the first two
moments of the window's natural measure. python/chk_farfield.py tests that on
the two simulated examples. This script tests it on the fatigue record, using
the raw strain data behind the tracking matrix (see chk_rawmatch.py).

Five feature sets are put through the two stages of Section 17.4 and compared:

    all        L reference points by the rule of Section sec:mt_alg
    near       the half of them closest to the attractor
    far        the other half
    farfield   L points at FAR bounding-box half-widths, where the expansion
               of eq:farfield certainly applies
    moments    the 5 window means and the 15 distinct covariance entries

and the far-point matrix is regressed on the moment features, which is the
direct test of the expansion.

Run from python/. About four minutes.
"""
import numpy as np
from scipy.io import loadmat
from scipy.linalg import subspace_angles
from gsvd import sod
from _psw2005 import tracking, raw, raw_path, NHEALTHY

FS, NW, NPTS, DIM, TAU, SUB = 160.0, 1728, 4096, 5, 6, 2
L, ENLARGE, FAR = 300, 1.5, 8.0
SEED = 0


def windows(x):
    W = x[:NW*NPTS].reshape(NW, NPTS)
    M = NPTS - (DIM - 1)*TAU
    return W, M


def embed(W, l, M, sub=SUB):
    return np.stack([W[l, j*TAU:j*TAU + M] for j in range(DIM)], 1)[::sub]


def box(W, M):
    pool = np.concatenate([embed(W, l, M, 30) for l in range(0, NW, 30)])
    lo, hi = pool.min(0), pool.max(0)
    return pool, 0.5*(lo + hi), 0.5*(hi - lo)


def chardist(W, M, ref):
    F = np.empty((NW, ref.shape[0]))
    for l in range(NW):
        Y = embed(W, l, M)
        F[l] = np.sqrt(((Y[:, None, :] - ref[None])**2).sum(-1)).mean(0)
    return F


def moments(W, M):
    iu = np.triu_indices(DIM)
    F = np.empty((NW, DIM + len(iu[0])))
    for l in range(NW):
        Y = embed(W, l, M)
        F[l] = np.r_[Y.mean(0), np.cov(Y, rowvar=False)[iu]]
    return F


def rho(ref, pool):
    """Distance to the nearest attractor point, in attractor RMS radii."""
    d = np.empty(len(ref))
    for j in range(len(ref)):
        d[j] = np.sqrt(((pool - ref[j])**2).sum(1)).min()
    return d/np.sqrt(((pool - pool.mean(0))**2).sum(1).mean())


def sod_rows(Mx):
    return sod(Mx.T - Mx.T.mean(axis=1, keepdims=True))


def stages(F, v1, v2, t, n0=NHEALTHY):
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
    return dict(L=F.shape[1], sov=lam[:3], R2=R2, ang=ang, sov2=lam2[:2],
                r=np.corrcoef(t[m], 1/q[m])[0, 1], q=q)


def main():
    Y, t, v1, v2 = tracking()
    x = raw()
    W, M = windows(x)
    pool, mid, half = box(W, M)
    rng = np.random.default_rng(SEED)

    ref = mid + (2*rng.random((L, DIM)) - 1)*half*ENLARGE
    r = rho(ref, pool[::5])
    med = np.median(r)
    print("reference points: rho from %.2f to %.2f attractor radii, median %.2f"
          % (r.min(), r.max(), med))

    # points deliberately in the far field: on a shell at FAR half-widths
    u = rng.normal(size=(L, DIM))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    reff = mid + u*half*FAR
    rf = rho(reff, pool[::5])
    print("far-field shell:  rho from %.1f to %.1f attractor radii" % (rf.min(), rf.max()))

    Fall = chardist(W, M, ref)
    Fsh0 = chardist(W, M, reff)
    sets = [("supplied PSW (405)", Y),
            ("all", Fall),
            ("near (rho < %.2f)" % med, Fall[:, r < med]),
            ("far  (rho > %.2f)" % med, Fall[:, r >= med]),
            ("far field (%.0f widths)" % FAR, Fsh0),
            ("moments (5 + 15)", moments(W, M))]

    print("\n%-24s %4s  %-22s %-15s %-15s %-14s %7s"
          % ("features", "L", "healthy SOVs", "R2 on v1, v2", "angles (rad)",
             "damage SOVs", "Paris r"))
    out = {}
    for name, F in sets:
        d = stages(F, v1, v2, t)
        out[name] = d
        print("%-24s %4d  %7.1f %6.1f %5.1f  %.4f %.4f   %.4f %.4f   "
              "%6.1f %5.1f   %7.4f"
              % (name, d["L"], d["sov"][0], d["sov"][1], d["sov"][2],
                 d["R2"][0], d["R2"][1], d["ang"][0], d["ang"][1],
                 d["sov2"][0], d["sov2"][1], d["r"]))

    Fsh, Fm = Fsh0, moments(W, M)

    # Control 1. How much of each tracking matrix is a function of the twenty
    # moment features of the same windows? For a far reference point eq:farfield
    # is linear in the covariance, so R^2 near one is the expansion confirmed,
    # not a discovery. The informative rows are the near half and the PSW matrix.
    A = np.c_[(Fm - Fm.mean(0))/Fm.std(0), np.ones(NW)]
    print("\nControl 1 -- R^2 of each tracking matrix on the 20 moment features:")
    for lab, F in (("chardist, near half", Fall[:, r < med]),
                   ("chardist, far half", Fall[:, r >= med]),
                   ("chardist, shell", Fsh),
                   ("PSW (source paper)", Y)):
        Fc = F - F.mean(0)
        res = Fc - A @ np.linalg.lstsq(A, Fc, rcond=None)[0]
        per = 1 - res.var(0)/np.maximum(Fc.var(0), 1e-300)
        print("   %-20s total %.5f   per column: median %.4f"
              % (lab, 1 - res.var(0).sum()/Fc.var(0).sum(), np.median(per)))

    # Control 2. Is the failure of the twenty moment features about the FEATURES
    # or about their NUMBER? Random column subsets of the characteristic-distance
    # matrices, at the same L.
    rng2 = np.random.default_rng(1)
    print("\nControl 2 -- random column subsets, mean of 8 draws:")
    print("   %-24s %4s  %-17s %s" % ("features", "L", "SOVs 1,2,3", "R2 on v1, v2"))
    for name, F in (("chardist, mixed", Fall), ("chardist, shell", Fsh)):
        for Ls in (10, 20, 40, 80, 150, 300):
            R2, SV = [], []
            for _ in range(8 if Ls < F.shape[1] else 1):
                c = (rng2.choice(F.shape[1], Ls, replace=False)
                     if Ls < F.shape[1] else np.arange(F.shape[1]))
                dd = stages(F[:, c], v1, v2, t)
                R2.append(dd["R2"]); SV.append(dd["sov"][:3])
            R2, SV = np.mean(R2, 0), np.mean(SV, 0)
            print("   %-24s %4d  %6.1f %5.1f %4.1f   %.4f %.4f"
                  % (name, Ls, SV[0], SV[1], SV[2], R2[0], R2[1]))
    dd = stages(Fm, v1, v2, t)
    print("   %-24s %4d  %6.1f %5.1f %4.1f   %.4f %.4f"
          % ("moment features", 20, dd["sov"][0], dd["sov"][1], dd["sov"][2],
             dd["R2"][0], dd["R2"][1]))
    np.save("/tmp/ff_exp.npy", out, allow_pickle=True)


if __name__ == "__main__":
    main()
