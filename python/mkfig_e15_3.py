"""Figure e15_3.pdf (Chapter 16, Section 16.1.1).

(a) periodogram of s_n = x_n^3 and of one phase-randomized (FT) surrogate,
    which are identical bin by bin;
(b) the ORDINARY sample autocorrelation at lag one of the data against 99 FT
    surrogates, with the circular autocorrelation -- the quantity phase
    randomization preserves exactly -- marked;
(c) rank of the data's ordinary C_1 among its 99 FT surrogates over 40
    independent records of the same process, showing that the wrap-around
    term biases the ordinary statistic upward without making it informative.

This PDF replaces the pair e15_3 + e15_4 of the first edition.
Runs in about half a minute.
"""
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _ch16data import ar1_cubed, ft_surr, lag1, lag1_circ

B, NREC = 99, 40
rng = np.random.default_rng(23)
x, s = ar1_cubed()
N = s.size

P = lambda y: np.abs(np.fft.rfft(y - y.mean()))**2
f = np.arange(N//2 + 1)/N                     # cycles per sample
one = ft_surr(s, rng)
print("max |periodogram(data) - periodogram(surrogate)| / max periodogram: "
      f"{np.max(np.abs(P(s) - P(one)))/P(s).max():.2e}")

c_data, cc_data = lag1(s), lag1_circ(s)
surr = [ft_surr(s, rng) for _ in range(B)]
c_sur = np.array([lag1(u) for u in surr])
cc_sur = np.array([lag1_circ(u) for u in surr])
rank = int(np.sum(c_sur < c_data) + 1)
print(f"ordinary C_1: data {c_data:.6f}; surrogates {c_sur.min():.6f} to "
      f"{c_sur.max():.6f}; rank {rank} of {B+1}")
print(f"circular C_1: data {cc_data:.8f}; surrogate spread "
      f"{cc_sur.max() - cc_sur.min():.2e} (exactly preserved)")

ranks = []
for t in range(NREC):
    _, st = ar1_cubed(seed=1000 + t)
    d = lag1(st)
    ranks.append(int(np.sum([lag1(ft_surr(st, rng)) < d for _ in range(B)]) + 1))
ranks = np.array(ranks)
print(f"rank of the data over {NREC} records: mean {ranks.mean():.1f}, "
      f"median {np.median(ranks):.0f}, rank {B+1} in {int((ranks == B+1).sum())} "
      f"of {NREC}, rank 1 in {int((ranks == 1).sum())}")

fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.1))
ax[0].semilogy(f[1:], P(s)[1:], color=COL[0], lw=0.4, label="data")
ax[0].semilogy(f[1:], P(one)[1:], color=COL[1], lw=0.4, ls="--",
               label="FT surrogate")
ax[0].set_xlabel("$f$ (cycles/sample)")
ax[0].set_ylabel("periodogram")
ax[0].legend(loc="upper right", fontsize=7)

ax[1].hist(c_sur, bins=15, color=COL[0], alpha=0.75, edgecolor="none",
           label=f"{B} surrogates")
ax[1].axvline(c_data, color=COL[1], lw=1.5, label="data")
ax[1].axvline(cc_data, color=COL[4], lw=1.0, ls=":", label="circular $C_1$")
ax[1].set_xlabel("$C_1$"); ax[1].set_ylabel("count")
ax[1].legend(loc="upper left", fontsize=6)

ax[2].hist(ranks, bins=np.arange(0, B + 2, 10), color=COL[2], alpha=0.85,
           edgecolor="none")
ax[2].set_xlabel("rank of the data, %d records" % NREC)
ax[2].set_ylabel("count")
for a, L in zip(ax, "abc"):
    panel(a, L, x=-0.24)
fig.tight_layout()
fig.savefig(FIGDIR + "e15_3.pdf")
print("wrote figures/e15_3.pdf")
