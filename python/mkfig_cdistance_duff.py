"""Figure cdistance_duff.pdf (Chapter 17, Section 17.4): characteristic distance
to three of the L = 400 reference points in the reconstructed Duffing phase
space, against the forcing amplitude f.  Runs in about two minutes."""
import time
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _ch17data import (duffing_setup, chardist, D_N, D_NPTS, D_L, D_TAU,
                       D_DIM, ENLARGE)

t0 = time.time()
f, W, ref = duffing_setup()
Y = chardist(W, ref)
print(f"N = {D_N} windows, {D_NPTS} points each (72 per forcing period, 166 periods), "
      f"d = {D_DIM}, tau = {D_TAU}, L = {D_L}, box enlarged by {ENLARGE}")
print(f"tracking matrix {Y.shape}, D in [{Y.min():.3f}, {Y.max():.3f}]")
rho = np.linalg.norm(ref - W.reshape(-1, D_DIM).mean(axis=0), axis=1)
jj = list(np.argsort(rho)[:3])
for j in jj:
    print(f"  column {j}: rho = {rho[j]:.2f}, D in "
          f"[{Y[:, j].min():.3f}, {Y[:, j].max():.3f}]")

fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.1))
for k, j in enumerate(jj):
    ax[k].plot(f, Y[:, j], ".", color=COL[0], ms=1.2)
    ax[k].set_xlabel("$f$")
    ax[k].set_xlim(0.35, 0.40)
    panel(ax[k], "abc"[k], x=-0.28)
ax[0].set_ylabel("$D$")
fig.tight_layout()
fig.savefig(FIGDIR + "cdistance_duff.pdf")
print("wrote figures/cdistance_duff.pdf  (%.0f s)" % (time.time() - t0))
