"""Figure track3.pdf (Chapter 17, Section 17.4): SOD of the Duffing tracking
matrix.  (a) ten largest SOVs, (b) the drift f_l, (c) the first SOC,
(d) the first SOC against f with a least-squares affine fit.
Runs in about two minutes."""
import time
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _ch17data import duffing_setup, chardist, track, corr, D_N

t0 = time.time()
f, W, ref = duffing_setup()
Y = chardist(W, ref)
lam, Q = track(Y)
q = Q[:, 0]
q = q*np.sign(corr(q, f))
p = np.polyfit(f, q, 1)
res = q - np.polyval(p, f)
print(f"SOVs 1..10: {np.array2string(lam[:10], precision=1)}")
print(f"correlation of SOC_1 with f : {corr(q, f):.5f}")
print(f"affine fit residual {res.std():.4g} over an SOC range of "
      f"{q.max() - q.min():.4g} ({100*res.std()/(q.max()-q.min()):.2f} percent)")
print(f"correlation of SOC_2 with f : {corr(Q[:, 1], f):.4f}")

fig, ax = plt.subplots(2, 2, figsize=(6.0, 4.0))
ax[0, 0].semilogy(np.arange(1, 11), lam[:10], "o", color=COL[0], ms=4)
ax[0, 0].set_xlabel("index $p$"); ax[0, 0].set_ylabel(r"SOV $\lambda_p^2$")
ax[0, 1].plot(np.arange(D_N), f, color=COL[0])
ax[0, 1].set_xlabel(r"window index $\ell$"); ax[0, 1].set_ylabel("$f$")
ax[1, 0].plot(np.arange(D_N), q, color=COL[0])
ax[1, 0].set_xlabel(r"window index $\ell$"); ax[1, 0].set_ylabel(r"$\mathrm{SOC}_1$")
ax[1, 1].plot(f, q, ".", color=COL[1], ms=1.5)
ax[1, 1].plot(f, np.polyval(p, f), "k-", lw=0.8)
ax[1, 1].set_xlabel("$f$"); ax[1, 1].set_ylabel(r"$\mathrm{SOC}_1$")
for axi, L in zip(ax.ravel(), "abcd"):
    panel(axi, L, x=-0.2)
fig.tight_layout()
fig.savefig(FIGDIR + "track3.pdf")
print("wrote figures/track3.pdf  (%.0f s)" % (time.time() - t0))
