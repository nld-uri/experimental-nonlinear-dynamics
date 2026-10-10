"""Lorenz_SNR.pdf and Duffing_SNR.pdf -- signal-to-noise ratio against iteration
number (Chapter 14, figures fig_L_SNRs and fig_D_SNRs), and the numbers of
Tables TL and TD.

Left panel: the local projective filter (podfilt, 64-point neighbour clouds).
Right panel: the smooth projective filter (sodfilt, bundles of nine 11-point
strands, tent weighting). Both at d = 6, q = 3, ten iterations, on the
N = 20 000 records of _ch14data.py at the five noise levels.

Running it also prints Tables TL and TD: the SNR of the unfiltered record and
after ten iterations of each filter, at 5 and 20 per cent noise, together with
the whole iteration history.

    python3 mkfig_nr_snr.py
"""
import numpy as np
from _figstyle import plt, COL, FIGDIR, panel
import _ch14data as C

MARK = "os^v<"


def one(name, out):
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
    hist = {}
    for a, meth, ttl in zip(ax, ("POD", "SOD"), ("local projective",
                                                 "smooth projective")):
        for j, p in enumerate(C.NOISE):
            s = C.snr_curve(name, p, meth)
            hist[(meth, p)] = s
            a.plot(np.arange(s.size), s, "-" + MARK[j], color=COL[j], ms=3.2,
                   mfc="none", mew=0.8, label=r"$%d\,\%%$ noise" % round(100*p))
        a.set_xlabel("iteration number")
        a.set_xlim(-0.4, C.ITER + 0.4)
        a.grid(True, lw=0.3, color="0.85")
    ax[0].set_ylabel("SNR (dB)")
    ax[1].legend(loc="lower right", ncol=2, handlelength=1.4,
                 columnspacing=1.0, frameon=True, framealpha=0.92,
                 edgecolor="none")
    panel(ax[0], "a", x=-0.14)
    panel(ax[1], "b", x=-0.06)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGDIR + out)
    plt.close(fig)
    return hist


if __name__ == "__main__":
    for name, out in (("Lorenz", "Lorenz_SNR.pdf"),
                      ("Duffing", "Duffing_SNR.pdf")):
        h = one(name, out)
        print("\n%s: SNR (dB), iteration 0 (unfiltered) to %d" % (name, C.ITER))
        for meth in ("POD", "SOD"):
            for p in C.NOISE:
                print("  %s %4.0f %%  " % (meth, 100*p) +
                      " ".join("%6.2f" % v for v in h[(meth, p)]))
        print("  Table: noise | none | POD(10) | SOD(10) | gains")
        for p in C.NOISE:
            n0 = h[("POD", p)][0]
            print("   %4.0f %%  %7.4f  %7.4f  %7.4f   (+%.2f, +%.2f dB)"
                  % (100*p, n0, h[("POD", p)][-1], h[("SOD", p)][-1],
                     h[("POD", p)][-1] - n0, h[("SOD", p)][-1] - n0))
