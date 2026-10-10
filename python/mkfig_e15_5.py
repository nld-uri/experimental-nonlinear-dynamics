"""Figure e15_5.pdf (Chapter 16, Section 16.1.1).

(a) Welch spectra (256-point Hann segments, 50 percent overlap) of
    s_n = x_n^3 and of one iteratively refined (IAAFT) surrogate whose
    histogram is exactly that of the data;
(b) the normalized one-step-ahead local-linear prediction error of the data
    against 199 IAAFT surrogates -- the double-sided test of null
    hypothesis three.

This PDF replaces the pair e15_5 + e15_6 of the first edition.
Runs in about two minutes.
"""
import time
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _ch16data import ar1_cubed, prederr, lag1
from irsurr import irsurr
from psdwelch import psdwelch

B, NIT = 199, 60                 # double-sided test, alpha = 0.01
L = 256                          # Welch segment length
D3, TAU, KNN = 3, 1, 8           # embedding and neighbourhood of the predictor
t0 = time.time()
rng = np.random.default_rng(2016)
x, s = ar1_cubed()

surr = [irsurr(s, NIT, rng)[0] for _ in range(B)]
print("histogram exact for every surrogate:",
      all(np.allclose(np.sort(u), np.sort(s)) for u in surr))
f, Gd = psdwelch(s, 1.0, L)
Gbar = np.mean([psdwelch(u, 1.0, L)[1] for u in surr], axis=0)
bias = np.maximum(Gbar/Gd, Gd/Gbar)
print(f"systematic Welch mismatch after {NIT} iterations (ensemble mean of {B} "
      f"surrogates vs data): worst bin factor {bias.max():.2f}, "
      f"median bin factor {np.median(bias):.3f}")
Gbar2 = np.mean([psdwelch(irsurr(s, 200, rng)[0], 1.0, L)[1]
                 for _ in range(B)], axis=0)
b2 = np.maximum(Gbar2/Gd, Gd/Gbar2)
print(f"same after 200 iterations: worst bin factor {b2.max():.2f}, "
      f"median {np.median(b2):.3f}")
rat1 = np.maximum(psdwelch(surr[0], 1.0, L)[1]/Gd, Gd/psdwelch(surr[0], 1.0, L)[1])
print(f"single-surrogate scatter (surrogate 1 vs data): worst bin factor "
      f"{rat1.max():.2f}, median {np.median(rat1):.2f}")
c1s = np.array([lag1(u) for u in surr])
print(f"lag-one autocorrelation: data {lag1(s):.4f}; surrogates "
      f"{c1s.mean():.4f} +- {c1s.std():.4f}")

e_data = prederr(s, d=D3, tau=TAU, k=KNN)
e_sur = np.array([prederr(u, d=D3, tau=TAU, k=KNN) for u in surr])
rank = int(np.sum(e_sur < e_data) + 1)
print(f"prediction error: data {e_data:.4f}; surrogates {e_sur.min():.4f} to "
      f"{e_sur.max():.4f} (mean {e_sur.mean():.4f}); rank {rank} of {B+1}")
print("two-sided rejection at alpha = 0.01 requires rank 1 or %d" % (B + 1))

_, Gs1 = psdwelch(surr[0], 1.0, L)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.3))
ax[0].semilogy(f, Gd, color=COL[0], lw=0.8, label="data")
ax[0].semilogy(f, Gs1, color=COL[1], lw=0.8, ls="--", label="IAAFT surrogate")
ax[0].set_xlabel("$f$ (cycles/sample)")
ax[0].set_ylabel(r"$G_{ss}(f)$ (units of $s^2$/cycle)")
ax[0].legend(loc="upper right", fontsize=7)
ax[1].hist(e_sur, bins=25, color=COL[0], alpha=0.75, edgecolor="none",
           label=f"{B} surrogates")
ax[1].axvline(e_data, color=COL[1], lw=1.5, label="data")
ax[1].set_xlabel("normalized one-step prediction error")
ax[1].set_ylabel("count")
ax[1].legend(loc="upper right", fontsize=7)
for a, Lb in zip(ax, "ab"):
    panel(a, Lb, x=-0.15)
fig.tight_layout()
fig.savefig(FIGDIR + "e15_5.pdf")
print("wrote figures/e15_5.pdf  (%.0f s)" % (time.time() - t0))
