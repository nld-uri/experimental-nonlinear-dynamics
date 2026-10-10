"""Figure rossler_bifu.pdf (Chapter 17, Section 17.4): bifurcation diagram of
the Rossler system with b = 0.6, c = 6.0 against a in [0.1, 0.4].

The plotted quantity is the local MINIMUM of y(t), which is the branch the
first-edition figure showed (its axis label y_max was wrong: y_max is positive
for every a in this range).  Runs in about a minute."""
import time
import numpy as np
from _figstyle import COL, plt, FIGDIR
from _ch17data import rossler_windows

NA, NPTS, DT = 1200, 4000, 0.06
t0 = time.time()
a = np.linspace(0.1, 0.4, NA)
traj = rossler_windows(a, n=NPTS, dt=DT)          # (NA, NPTS, 3)
y = traj[:, :, 1]
loc = (y[:, 1:-1] < y[:, :-2]) & (y[:, 1:-1] < y[:, 2:])   # local minima
ai, ni = np.nonzero(loc)
ymin = y[ai, ni + 1]
print(f"{NA} parameter values, {NPTS} points each, dt = {DT}")
print(f"local minima of y: {ymin.size} points, range "
      f"[{ymin.min():.2f}, {ymin.max():.2f}]")
print(f"max of y over the record: {y.max():.2f} (never negative, so the "
      "first-edition label y_max cannot be this quantity)")

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(a[ai], ymin, ",", color=COL[0], alpha=0.5)
ax.set_xlabel("$a$")
ax.set_ylabel(r"$y_{\min}$")
ax.set_xlim(0.1, 0.4)
fig.tight_layout()
fig.savefig(FIGDIR + "rossler_bifu.pdf", dpi=600)
print("wrote figures/rossler_bifu.pdf  (%.0f s)" % (time.time() - t0))
