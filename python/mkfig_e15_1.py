"""Figure e15_1.pdf (Chapter 16, Section 16.1.1).

(a) the AR(1) record x_n = 0.99 x_{n-1} + eta_n, N = 4096, mean removed;
(b) its nonlinear observation s_n = x_n^3;
(c) the lag-one sample autocorrelation of s against 99 shuffle (unconstrained
    randomization) surrogates -- the test of null hypothesis one.

This PDF replaces the pair e15_1 + e15_2 of the first edition.
Prints every number the text quotes.  Runs in about a second.
"""
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _ch16data import ar1_cubed, shuffle_surr, lag1

B = 99                                   # single-sided test, alpha = 0.01
rng = np.random.default_rng(11)
x, s = ar1_cubed()
N = x.size

c_data = lag1(s)
c_sur = np.array([lag1(shuffle_surr(s, rng)) for _ in range(B)])
rank = int(np.sum(c_sur < c_data) + 1)
print(f"N = {N};  std(x) = {x.std():.3f};  std(s) = {s.std():.1f}")
print(f"C_1 of data      : {c_data:.5f}")
print(f"C_1 of {B} shuffles: min {c_sur.min():.5f}  max {c_sur.max():.5f}  "
      f"sd {c_sur.std():.5f}  (1/sqrt(N) = {1/np.sqrt(N):.5f})")
print(f"rank of the data : {rank} of {B+1}")

fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.0))
n = np.arange(1, N + 1)
ax[0].plot(n, x, color=COL[0], lw=0.4)
ax[0].set_xlabel("$n$"); ax[0].set_ylabel("$x_n$")
ax[1].plot(n, s, color=COL[0], lw=0.4)
ax[1].set_xlabel("$n$"); ax[1].set_ylabel("$s_n = x_n^3$")
for a in ax[:2]:
    a.set_xlim(0, N)
ax[2].hist(c_sur, bins=15, color=COL[0], alpha=0.75, edgecolor="none",
           label=f"{B} surrogates")
ax[2].axvline(c_data, color=COL[1], lw=1.5, label="data")
ax[2].set_xlim(-0.1, 1.05)
ax[2].set_xlabel("$C_1$"); ax[2].set_ylabel("count")
ax[2].legend(loc="upper center", fontsize=7)
for a, L in zip(ax, "abc"):
    panel(a, L, x=-0.22)
fig.tight_layout()
fig.savefig(FIGDIR + "e15_1.pdf")
print("wrote figures/e15_1.pdf")
