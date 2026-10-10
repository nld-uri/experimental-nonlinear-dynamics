"""Does the far-field expansion hold? Split the reference points by distance
from the attractor and track with each group separately.

Section sec:mt_farfield predicts that a reference point far from the attractor
sees only the first two moments of the natural measure, so far points alone
should recover the slow variable as well as the full set. The measured record
of Section 17.4 cannot test this -- its raw vibration data are lost and only
the finished tracking matrix survives -- so the test is done here, on the two
simulated examples, where the reference points are ours to place.

    python3 chk_farfield.py
"""
import time
import numpy as np
from _ch17data import (rossler_setup, duffing_setup, chardist, track, corr,
                       R_N, D_N)


def split(ref, pool):
    """rho_j = distance from reference point j to the nearest attractor point,
    in units of the attractor's RMS radius."""
    d = np.sqrt(((ref[:, None, :] - pool[None, :, :])**2).sum(-1)).min(1)
    rms = np.sqrt(((pool - pool.mean(0))**2).sum(1).mean())
    return d/rms


def run(name, drift, win, ref, pool):
    rho = split(ref, pool)
    med = np.median(rho)
    groups = [("all", np.ones(len(ref), bool)),
              ("near (rho < %.2f)" % med, rho < med),
              ("far  (rho > %.2f)" % med, rho >= med)]
    print("\n%s: %d reference points, rho from %.2f to %.2f (attractor RMS radii)"
          % (name, len(ref), rho.min(), rho.max()))
    for lab, m in groups:
        Y = chardist(win, ref[m])
        lam, Q = track(Y)
        q = Q[:, 0]
        q = q*np.sign(corr(q, drift))
        p = np.polyfit(drift, q, 1)
        res = (q - np.polyval(p, drift)).std()/(q.max() - q.min())
        print("   %-22s L = %3d   r = %.5f   SOV1 = %8.1f   "
              "affine residual %5.2f %%"
              % (lab, m.sum(), corr(q, drift), lam[0], 100*res))


if __name__ == "__main__":
    t0 = time.time()
    a, traj, ref = rossler_setup()
    pool = traj.reshape(-1, 3)[::200]
    run("Rossler", a, traj, ref, pool)

    f, W, refd = duffing_setup()
    poold = W.reshape(-1, W.shape[2])[::200]
    run("Duffing", f, W, refd, poold)
    print("\n(%.0f s)" % (time.time() - t0))
