"""stationaritytest.pdf (label averages), replacing the pair stationaritytest.pdf +
stationarity.pdf. (a) A record x_n of N = 2000 independent Gaussian samples with mean 1
and unit variance (seed 0; the first 400 samples are shown) with one sliding window of
M = 100 samples = t_w marked. (b) The simple stationarity test on that record: the 20
non-overlapping window means mu_i with bars +- sigma_i/sqrt(M), and the grand mean
(dashed). (c) The same test on the nonstationary record x_n + 0.5 sin(2 pi n/N)."""
import numpy as np
from _figstyle import *

N, M, MU, SIG, TREND = 2000, 100, 1.0, 1.0, 0.5
rng = np.random.default_rng(0)
x = MU + SIG*rng.standard_normal(N)
n = np.arange(N)
y = x + TREND*np.sin(2*np.pi*n/N)

def windows(z):
    Z = z.reshape(-1, M)
    return Z.mean(axis=1), Z.std(axis=1, ddof=1)/np.sqrt(M)

plt.rcParams.update({
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
})

fig = plt.figure(figsize=(6, 2.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], wspace=0.25, hspace=0.15)
ax = fig.add_subplot(gs[:, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 1], sharex=axb)
NS, W0 = 400, 150
sketch_axes(ax, "$n$", "$x_n$")
ax.plot(n[:NS], x[:NS], color=COL[0], lw=0.5)
ax.axvline(W0, color="k", lw=0.6); ax.axvline(W0+M, color="k", lw=0.6)
ax.annotate("", (W0, -2.9), (W0+M, -2.9), arrowprops=dict(arrowstyle="<->", lw=0.6))
ax.text(W0+M/2, -3.3, "$t_w$", ha="center", va="top", fontsize=8)
ax.text(W0+M/2, 4.3, r"$M\ \rm{samples}$", ha="center", va="bottom", fontsize=8)
ax.set_ylim(-3.8, 5.0); ax.set_xlim(-5, NS+5); ax.yaxis.set_label_coords(0.06, 1.0); panel(ax, "a", x=-0.04)
for a, z, lab in ((axb, x, "b"), (axc, y, "c")):
    m, s = windows(z)
    i = np.arange(1, m.size+1)
    a.errorbar(i, m, yerr=s, fmt="o", color=COL[0], ms=3, mfc="w", capsize=2, lw=0.8)
    a.axhline(z.mean(), color="k", lw=0.6, ls="--")
    out = np.mean(abs(m - z.mean()) > s)
    print("panel (%s): mean of window sigma_mu = %.3f, fraction of means outside the band about the grand mean = %.2f, spread of means (std) = %.3f" % (lab, s.mean(), out, m.std(ddof=1)))
    a.set_ylabel(r"$\widehat\mu_i \pm \sigma_{\widehat\mu,i}$"); panel(a, lab, x=-0.28)
    a.set_ylim(0.4, 1.7)
axb.tick_params(labelbottom=False); axc.set_xlabel(r"$i\ \rm{window\ index}$"); axc.set_xticks([1, 5, 10, 15, 20])
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.17, top=0.9)
fig.savefig(FIGDIR+"stationaritytest.pdf")
