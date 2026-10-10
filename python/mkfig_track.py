"""Figure track.pdf (Chapter 17, Section 17.6): the first SOC against the
Rossler parameter a for the clean record and at 20, 60 and 100 percent
measurement noise.  Runs in about a minute."""
import time
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _ch17data import rossler_setup, chardist, track, corr

LEVELS = [0.0, 0.2, 0.6, 1.0]
t0 = time.time()
a, traj, ref = rossler_setup()
flat = traj.reshape(-1, 3)
rms = float(np.sqrt(np.mean((flat - flat.mean(0))**2)))
rng = np.random.default_rng(3)

fig, ax = plt.subplots(2, 2, figsize=(6.0, 4.0))
for k, lev in enumerate(LEVELS):
    tr = traj + lev*rms*rng.standard_normal(traj.shape)
    Y = chardist(tr, ref)
    lam, Q = track(Y)
    q = Q[:, 0]
    q = q*np.sign(corr(q, a))
    r = corr(q, a)
    print(f"noise {100*lev:3.0f}%: SOV_1 {lam[0]:.4g}, SOV_2 {lam[1]:.4g}, "
          f"correlation of SOC_1 with a {r:.4f}")
    axi = ax.ravel()[k]
    axi.plot(a, q, ".", color=COL[0], ms=1.2)
    axi.set_xlabel("$a$"); axi.set_ylabel(r"$\mathrm{SOC}_1$")
    axi.set_xlim(0.1, 0.4)
    panel(axi, "abcd"[k], x=-0.2)
fig.tight_layout()
fig.savefig(FIGDIR + "track.pdf")
print("wrote figures/track.pdf  (%.0f s)" % (time.time() - t0))
