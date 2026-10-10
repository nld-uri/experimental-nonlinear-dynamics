"""Figure e15_8.pdf (Chapter 16, Section 16.1.2): histogram and Welch spectrum
of the noisy Lorenz record and of one IAAFT surrogate, showing what the
generator preserves.  Runs in a few seconds."""
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _ch16data import lorenz_record, lag1
from irsurr import irsurr
from psdwelch import psdwelch

L = 256
rng = np.random.default_rng(2016)
s = lorenz_record()
r, _ = irsurr(s, 60, rng)
f, Gd = psdwelch(s, 1.0, L)
_, Gr = psdwelch(r, 1.0, L)
bias = np.maximum(Gd/Gr, Gr/Gd)
print("histogram exact:", np.allclose(np.sort(r), np.sort(s)))
print(f"Welch bin ratio data/surrogate: worst {bias.max():.2f}, "
      f"median {np.median(bias):.2f}")
print(f"lag-one autocorrelation: data {lag1(s):.4f}, surrogate {lag1(r):.4f}")

bins = np.linspace(min(s.min(), r.min()), max(s.max(), r.max()), 80)
fig, ax = plt.subplots(2, 2, figsize=(6.0, 3.6))
for j, (y, name, c) in enumerate([(s, "data", COL[0]),
                                  (r, "IAAFT surrogate", COL[1])]):
    ax[0, j].hist(y, bins=bins, color=c, edgecolor="none")
    ax[0, j].set_xlabel("$s$"); ax[0, j].set_ylabel("count")
    _, G = psdwelch(y, 1.0, L)
    ax[1, j].semilogy(f, G, color=c, lw=0.8)
    ax[1, j].set_xlabel("$f$ (cycles/sample)")
    ax[1, j].set_ylabel(r"$G_{ss}(f)$")
    ax[1, j].set_ylim(1e-2, 1e4)
for a, Lb in zip(ax.ravel(), "abcd"):
    panel(a, Lb, x=-0.2)
fig.tight_layout()
fig.savefig(FIGDIR + "e15_8.pdf")
print("wrote figures/e15_8.pdf")
