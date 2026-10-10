"""Figure e15_9.pdf (Chapter 16, Section 16.1.2): normalized one-step-ahead
local-linear prediction error of the noisy Lorenz record against 199 IAAFT
surrogates.  Runs in about two minutes."""
import time
import numpy as np
from _figstyle import COL, plt, FIGDIR
from _ch16data import lorenz_record, prederr, LOR_D, LOR_TAU, LOR_NOISE
from irsurr import irsurr

B, NIT = 199, 60
t0 = time.time()
rng = np.random.default_rng(2016)
s = lorenz_record()
e_data = prederr(s, d=LOR_D, tau=LOR_TAU)
e_sur = np.array([prederr(irsurr(s, NIT, rng)[0], d=LOR_D, tau=LOR_TAU)
                  for _ in range(B)])
rank = int(np.sum(e_sur < e_data) + 1)
print(f"noise {100*LOR_NOISE:.0f}% of RMS, d = {LOR_D}, tau = {LOR_TAU}, "
      f"horizon 1 step")
print(f"prediction error: data {e_data:.4f}; surrogates {e_sur.min():.4f} to "
      f"{e_sur.max():.4f} (mean {e_sur.mean():.4f}); rank {rank} of {B+1}")

fig, ax = plt.subplots(figsize=(4.5, 2.3))
ax.hist(e_sur, bins=20, color=COL[0], alpha=0.75, edgecolor="none",
        label=f"{B} surrogates")
ax.axvline(e_data, color=COL[1], lw=1.5, label="data")
ax.set_xlabel("normalized one-step prediction error")
ax.set_ylabel("count")
ax.legend(loc="upper center", fontsize=7)
fig.tight_layout()
fig.savefig(FIGDIR + "e15_9.pdf")
print("wrote figures/e15_9.pdf  (%.0f s)" % (time.time() - t0))
