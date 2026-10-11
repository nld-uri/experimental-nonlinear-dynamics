"""Reruns of the Chapter 17 tracking numbers quoted in Section 17.4.

Uses the shared pipeline of _ch17data.py unchanged (same drift laws, same
reference-point rule, same centring + SOD), so the baselines reproduce
mkfig_track2.py and mkfig_track3.py.

  R0  Rossler, L = 50, n = 5000 points per window (the figure setting)
  R1  Rossler, L = 50, n = 20000 points per window (windows lengthened)
  R2  Rossler, L = 200, n = 5000 (features added; the first 50 reference
      points are those of R0, since refpoints draws row by row from the same
      seeded generator); R2s1, R2s2 repeat R2 with reference seeds 1, 2
  D0  Duffing, L = 400, n = 12000 (the figure setting)

For each Rossler run: r = corr(SOC_1, a), R^2 of affine / quadratic fits of
SOC_1 on a, and |corr(SOC_1, cos(pi l/(N-1)))|, the slowest mode of the
record (a quadratic in a for the quarter-sine drift).

    python3 chk_track_rerun.py            # all runs (about 6 min on 2 cores)
    python3 chk_track_rerun.py R0 R2      # selected runs
"""
import sys
import time
import numpy as np
from _ch17data import (rossler_setup, duffing_setup, chardist, track, corr,
                       R_N, D_N)


def r2(y, x, deg):
    p = np.polyfit(x, y, deg)
    return 1 - np.var(y - np.polyval(p, x))/np.var(y)


def rossler(tag, L, n, seed=0):
    t0 = time.time()
    a, traj, ref = rossler_setup(L=L, n=n, seed=seed)
    Y = chardist(traj, ref)
    del traj
    lam, Q = track(Y)
    q = Q[:, 0]*np.sign(corr(Q[:, 0], a))
    c1 = np.cos(np.pi*np.arange(R_N)/(R_N - 1))
    res = q - np.polyval(np.polyfit(a, q, 1), a)
    print(f"{tag}: Rossler L={L}, n={n}, N={R_N}, reference seed {seed}")
    print(f"    SOV1, SOV2 = {lam[0]:.4g}, {lam[1]:.4g}")
    print(f"    r = corr(SOC1, a) = {corr(q, a):.4f}")
    print(f"    R^2 affine in a = {r2(q, a, 1):.4f}, quadratic in a = {r2(q, a, 2):.4f}")
    print(f"    |corr(SOC1, cos(pi l/(N-1)))| = {abs(corr(q, c1)):.4f}")
    print(f"    affine residual std / SOC range = "
          f"{100*res.std()/np.ptp(q):.1f} percent   ({time.time()-t0:.0f} s)")
    sys.stdout.flush()


def duffing(tag):
    t0 = time.time()
    f, W, ref = duffing_setup()
    Y = chardist(W, ref)
    del W
    lam, Q = track(Y)
    q = Q[:, 0]*np.sign(corr(Q[:, 0], f))
    res = q - np.polyval(np.polyfit(f, q, 1), f)
    t = np.arange(D_N)
    c1 = np.cos(np.pi*t/(D_N - 1))
    print(f"{tag}: Duffing L=400, n=12000, N={D_N}")
    print(f"    SOV1, SOV2 = {lam[0]:.4g}, {lam[1]:.4g}  (ratio {lam[0]/lam[1]:.1f})")
    print(f"    r = corr(SOC1, f) = {corr(q, f):.5f}")
    print(f"    R^2 affine in f = {r2(q, f, 1):.5f}, quadratic = {r2(q, f, 2):.5f}")
    print(f"    |corr(SOC1, cos(pi l/(N-1)))| = {abs(corr(q, c1)):.4f}")
    print(f"    affine residual std / SOC range = "
          f"{100*res.std()/np.ptp(q):.2f} percent   ({time.time()-t0:.0f} s)")
    sys.stdout.flush()


RUNS = {"R0": lambda: rossler("R0", 50, 5000),
        "R1": lambda: rossler("R1", 50, 20000),
        "R2": lambda: rossler("R2", 200, 5000),
        "R2s1": lambda: rossler("R2s1", 200, 5000, seed=1),
        "R2s2": lambda: rossler("R2s2", 200, 5000, seed=2),
        "D0": lambda: duffing("D0")}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(RUNS)):
        RUNS[k]()
