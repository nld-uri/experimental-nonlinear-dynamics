"""Does the curvature correction of eq. (curvcorr) earn its place?

Section sec:nr_curv derives a correction that moves the centre of the local
projection from c_k to 2 c_k - c'_k, with c'_k the mean of the mass centres of
the r SPATIAL neighbours of the point. podfilt shipped only a TEMPORAL variant,
c'_k = (b_{k-1} + b_{k+1})/2, which diverges under iteration in the cloud
scheme. This script implements the spatial version (podfilt u = 2) and measures
all three against the clean record.

Run from python/. About three minutes.
"""
import numpy as np
from podfilt import podfilt
import _ch14data as C

NAMES = {1: "none", 0: "temporal", 2: "spatial (curvcorr)"}


def snr(clean, y):
    e = y - clean
    return 10*np.log10(np.var(clean)/np.var(e))


def run(name, p, d=6, q=3, r=64, npass=10):
    xc, x = C.clean(name), C.noisy(name, p)
    tau = C.TAU[name]
    out = {}
    for u in (1, 0, 2):
        y = x.copy(); s = [snr(xc, y)]
        for _ in range(npass):
            y = podfilt(y, r, tau, d, q, 1, u)
            s.append(snr(xc, y))
        out[u] = np.array(s)
    return out


if __name__ == "__main__":
    for name in ("Lorenz", "Duffing"):
        for p in (0.05, 0.20):
            out = run(name, p)
            print("\n%s, %d%% noise: SNR (dB) against iteration" % (name, round(100*p)))
            print("  correction        it0    it1    it2    it5   it10   best (at)")
            for u in (1, 0, 2):
                s = out[u]; b = int(np.argmax(s))
                print("  %-18s %5.1f  %5.1f  %5.1f  %5.1f  %5.1f   %5.1f (%d)"
                      % (NAMES[u], s[0], s[1], s[2], s[5], s[10], s[b], b))


def sweep_r(name="Lorenz", p=0.20, d=6, q=3, npass=8):
    """The correction scales as kappa eps^2/6, so it should pay only when the
    neighbourhood is large enough for the arc to bend inside it."""
    xc, x = C.clean(name), C.noisy(name, p)
    tau = C.TAU[name]
    print("\n%s, %d%% noise, d=%d q=%d: best SNR (dB) over %d passes"
          % (name, round(100*p), d, q, npass))
    print("  r      none    spatial   gain   eps/sigma")
    for r in (32, 64, 128, 256, 512, 1024):
        best = {}
        for u in (1, 2):
            y = x.copy(); s = [snr(xc, y)]
            for _ in range(npass):
                y = podfilt(y, r, tau, d, q, 1, u)
                s.append(snr(xc, y))
            best[u] = max(s)
        # typical neighbourhood radius at this r
        from nlutil import embed as _emb
        from scipy.spatial import cKDTree
        Y, _n = _emb(x, d, tau)
        dd, _ = cKDTree(Y.T).query(Y.T[::37], k=r)
        eps = np.median(dd[:, -1])/np.std(x)
        print("  %-6d %6.1f  %7.1f  %+6.1f   %7.3f"
              % (r, best[1], best[2], best[2] - best[1], eps))
