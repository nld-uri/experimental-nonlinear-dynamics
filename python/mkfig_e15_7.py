"""Figure e15_7.pdf (Chapter 16, Section 16.1.2): the Lorenz observation
s_n = x_n + eta_n (top) and one iteratively refined (IAAFT) surrogate of it
(bottom).  Runs in a few seconds."""
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _ch16data import lorenz_record, LOR_NOISE
from irsurr import irsurr

rng = np.random.default_rng(2016)
s = lorenz_record()
N = s.size
r, _ = irsurr(s, 60, rng)
print(f"N = {N}, noise = {100*LOR_NOISE:.0f}% of RMS, std = {s.std():.2f}")
print("surrogate histogram exact:", np.allclose(np.sort(r), np.sort(s)))

fig, ax = plt.subplots(2, 1, figsize=(6.0, 3.0), sharex=True, sharey=True)
n = np.arange(1, N + 1)
ax[0].plot(n, s, color=COL[0], lw=0.4)
ax[0].set_ylabel("$s_n$")
ax[1].plot(n, r, color=COL[1], lw=0.4)
ax[1].set_ylabel("surrogate")
ax[1].set_xlabel("$n$")
ax[1].set_xlim(0, N)
for a, L in zip(ax, "ab"):
    panel(a, L, x=-0.08)
fig.tight_layout()
fig.savefig(FIGDIR + "e15_7.pdf")
print("wrote figures/e15_7.pdf")
