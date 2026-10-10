"""Figure position.pdf (Chapter 17, Discussion): the Rossler parameter a
recovered from the characteristic position at only twenty reference points,
each contributing m = 3 columns to the tracking matrix.  The companion panel
(b) of the same figure is position_exp.pdf, the measured counterpart, built by
mkfig_position_exp.py.  Runs in about ten seconds."""
import time
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _ch17data import (rossler_setup, chardist, charpos, track, corr,
                       R_N, R_NPTS)

LP = 20
t0 = time.time()
a, traj, ref = rossler_setup(L=LP, seed=5)
Yp = charpos(traj, ref)
lam, Q = track(Yp)
q = Q[:, 0]
q = q*np.sign(corr(q, a))
print(f"characteristic position, L = {LP} reference points -> "
      f"{Yp.shape[1]} columns, N = {R_N} windows of {R_NPTS} points")
print(f"SOVs 1..5: {np.array2string(lam[:5], precision=1)}")
print(f"correlation of SOC_1 with a : {corr(q, a):.5f}")
Yd = chardist(traj, ref)
lam_d, Qd = track(Yd)
qd = Qd[:, 0]*np.sign(corr(Qd[:, 0], a))
print(f"same {LP} points, characteristic DISTANCE only: r = {corr(qd, a):.5f}, "
      f"SOV_1 = {lam_d[0]:.4g}")

SMALL = 8            # the size on the page; the figure is drawn at its
                     # include width, so LaTeX does not rescale the labels
fig, ax = plt.subplots(figsize=(2.9, 2.2))
ax.plot(a, q, ".", color=COL[0], ms=1.2)
ax.set_xlabel("$a$"); ax.set_ylabel(r"$\mathrm{SOC}_1$")
ax.set_xlim(0.1, 0.4)
ax.xaxis.label.set_size(SMALL); ax.yaxis.label.set_size(SMALL)
ax.tick_params(labelsize=SMALL)
panel(ax, "a", x=-0.22)
ax.texts[-1].set_fontsize(SMALL)
fig.tight_layout(pad=0.3)
fig.savefig(FIGDIR + "position.pdf")
print("wrote figures/position.pdf  (%.0f s)" % (time.time() - t0))
