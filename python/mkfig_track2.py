"""Figure track2.pdf (Chapter 17, Section 17.4): SOD of the Rossler tracking
matrix.  (a) ten largest SOVs, (b) the drift a_l, (c) the first SOC,
(d) the first SOC against a with a least-squares affine fit."""
import time
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _ch17data import rossler_setup, chardist, track, corr, R_N, R_L

t0 = time.time()
a, traj, ref = rossler_setup()
Y = chardist(traj, ref)
lam, Q = track(Y)
q = Q[:, 0]
q = q*np.sign(corr(q, a))
r = corr(q, a)
p = np.polyfit(a, q, 1)
res = q - np.polyval(p, a)
print(f"SOVs 1..10: {np.array2string(lam[:10], precision=1)}")
print(f"correlation of SOC_1 with a : {r:.5f}")
print(f"correlation of SOC_1 with the window index : {corr(q, np.arange(R_N)):.5f}")
print(f"affine fit residual {res.std():.4g} over an SOC range of "
      f"{q.max() - q.min():.4g} ({100*res.std()/(q.max()-q.min()):.1f} percent)")
print(f"correlation of SOC_2 with a : {corr(Q[:, 1], a):.4f}")

fig, ax = plt.subplots(2, 2, figsize=(6.0, 4.0))
ax[0, 0].semilogy(np.arange(1, 11), lam[:10], "o", color=COL[0], ms=4)
ax[0, 0].set_xlabel("index $p$"); ax[0, 0].set_ylabel(r"SOV $\lambda_p^2$")
ax[0, 1].plot(np.arange(R_N), a, color=COL[0])
ax[0, 1].set_xlabel(r"window index $\ell$"); ax[0, 1].set_ylabel("$a$")
ax[1, 0].plot(np.arange(R_N), q, color=COL[0])
ax[1, 0].set_xlabel(r"window index $\ell$"); ax[1, 0].set_ylabel(r"$\mathrm{SOC}_1$")
ax[1, 1].plot(a, q, ".", color=COL[1], ms=1.5)
ax[1, 1].plot(a, np.polyval(p, a), "k-", lw=0.8)
ax[1, 1].set_xlabel("$a$"); ax[1, 1].set_ylabel(r"$\mathrm{SOC}_1$")
for axi, L in zip(ax.ravel(), "abcd"):
    panel(axi, L, x=-0.2)
fig.tight_layout()
fig.savefig(FIGDIR + "track2.pdf")
print("wrote figures/track2.pdf  (%.0f s)" % (time.time() - t0))
