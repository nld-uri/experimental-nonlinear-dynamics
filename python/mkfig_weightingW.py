"""weightingW.pdf -- the three strand weightings of sodfilt (Chapter 14, Wweighting).

The weights are taken from nlutil.strand_weights, the routine sodfilt actually
uses, for half-length l = 10 (a 21-point strand), and normalized to sum to one
so that the three are comparable. Note that the tent w_j = l - |j| vanishes at
j = +-l, so those two points are absent from the logarithmic panel; the linear
panel shows them.

    python3 mkfig_weightingW.py
"""
import numpy as np
from _figstyle import plt, COL, FIGDIR, panel
from nlutil import strand_weights

L = 10
NAMES = [(0, "comb"), (1, "tent"), (2, "exponential")]

if __name__ == "__main__":
    j = np.arange(-L, L + 1)
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.3))
    for c, (w, nm) in enumerate(NAMES):
        wd = strand_weights(L, w)
        wd = wd/wd.sum()
        print("%-12s" % nm, " ".join("%.4g" % v for v in wd))
        for a, log in zip(ax, (False, True)):
            m = wd > 0 if log else np.ones_like(wd, bool)
            a.plot(j[m], wd[m], "o", color=COL[c], ms=3.5, mfc="none",
                   mew=0.9, label=nm)
            a.vlines(j[m], 0 if not log else 1e-6, wd[m], color=COL[c], lw=0.7)
    ax[0].set_ylim(0, 1.05)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-5, 2)
    for k, a in enumerate(ax):
        a.set_xlim(-L - 1, L + 1)
        a.set_xticks(np.arange(-L, L + 1, 5))
        a.set_xlabel("position within the strand, $j$")
        a.set_ylabel("$w_j$ (normalized)")
        panel(a, "ab"[k])
    ax[0].legend(loc="upper right", handlelength=1.0)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGDIR + "weightingW.pdf")
