"""Figure surr_twowell.pdf (Chapter 16, Section sec:surrmeasured).

Surrogate test of the linear null on the measured two-well oscillator record
of Section sec:twowelldata.

  (a) 39 iteratively refined (IAAFT, 20 iterations, histogram-exact branch)
      surrogates of the first 24,000 samples, with the time-reversal
      asymmetry at lag 5 as the statistic;
  (b) the same test on the whole 96,000-sample record.

Runs in about a minute.

Copyright by David Chelidze.  Written for the second edition.
"""
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _twowell import twowell
from irsurr import irsurr

B, NIT, LAG, NSEG = 39, 20, 5, 24000
SEED = 5


def trev(y, lag=LAG):
    """Normalized time-reversal asymmetry at the given lag, eq. (eq:trev)."""
    dy = np.asarray(y, float)[lag:] - np.asarray(y, float)[:-lag]
    return float(np.mean(dy**3)/np.mean(dy**2)**1.5)


x = twowell()
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.4))
for a, y, name, L in [(ax[0], x[:NSEG], "first 24,000 samples", "a"),
                      (ax[1], x, "whole record", "b")]:
    rng = np.random.default_rng(SEED)
    s = np.array([trev(irsurr(y, NIT, rng)[0]) for _ in range(B)])
    d = trev(y)
    rank = int((s < d).sum() + 1)
    z = (d - s.mean())/s.std(ddof=1)
    print(f"{name}: data {d:.4f}; {B} surrogates mean {s.mean():.4f}, "
          f"sd {s.std(ddof=1):.4f}, range [{s.min():.4f}, {s.max():.4f}]; "
          f"rank {rank} of {B+1}, z = {z:.1f}")
    a.hist(s, bins=13, color=COL[0], alpha=0.75, edgecolor="none",
           label=f"{B} IAAFT surrogates")
    a.axvline(d, color=COL[1], lw=1.5, label="data")
    a.set_xlabel(r"$T_{\mathrm{rev}}$ at lag $5$")
    a.set_ylabel("count")
    a.set_xlim(-0.09, 0.09)
    a.legend(loc="upper left", fontsize=7)
    panel(a, L, x=-0.15)
fig.tight_layout()
fig.savefig(FIGDIR + "surr_twowell.pdf")
print("wrote figures/surr_twowell.pdf")
