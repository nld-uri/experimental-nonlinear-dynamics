"""Figure cdistance.pdf (Chapter 17, Section 17.4): characteristic distance to
three of the L = 50 reference points against the Rossler parameter a."""
import time
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _ch17data import (rossler_setup, chardist, R_N, R_NPTS, R_DT, R_L,
                       ENLARGE)

t0 = time.time()
a, traj, ref = rossler_setup()
Y = chardist(traj, ref)
print(f"N = {R_N} windows, {R_NPTS} points each at t_s = {R_DT}, L = {R_L} "
      f"reference points, box enlarged by {ENLARGE}")
print(f"tracking matrix {Y.shape}, D in [{Y.min():.2f}, {Y.max():.2f}]")
rho = np.linalg.norm(ref - traj.reshape(-1, 3).mean(axis=0), axis=1)
jj = list(np.argsort(rho)[:3])          # the three reference points nearest
for j in jj:                            # the attractor, where D is not a
                                        # pure moment statistic
    print(f"  column {j}: reference point {np.round(ref[j], 2)}, "
          f"D in [{Y[:, j].min():.2f}, {Y[:, j].max():.2f}]")

fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.1))
for k, j in enumerate(jj):
    ax[k].plot(a, Y[:, j], ".", color=COL[0], ms=1.2)
    ax[k].set_xlabel("$a$")
    ax[k].set_xlim(0.1, 0.4)
    panel(ax[k], "abc"[k], x=-0.28)
ax[0].set_ylabel("$D$")
fig.tight_layout()
fig.savefig(FIGDIR + "cdistance.pdf")
print("wrote figures/cdistance.pdf  (%.0f s)" % (time.time() - t0))
