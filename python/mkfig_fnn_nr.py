"""fnn_lorenz.pdf and fnn_duffing.pdf -- false nearest neighbours at five noise
levels (Chapter 14, figures fig_L_c2 and fig_D_c2).

Kennel's composite false-nearest-neighbour fraction (ratio threshold
R_tol = 10 AND attractor-size threshold A_tol = 2, plain nearest neighbour,
Theiler window w = 3 tau) against embedding dimension d = 1..10, for the clean
record and the five noise levels of Section sec:nr_eval. The inset repeats
d = 4..10 on a magnified scale, where the noise floor lives. Records and delays
from _ch14data.py (Lorenz tau = 12, Duffing tau = 7, N = 20 000 each).

The plain ratio criterion alone ("orig" in fnn.fnnfrac) has no noise floor: it
falls to zero by d = 7 at every noise level, on both records. The floor the
chapter describes is Kennel's second criterion.

    python3 mkfig_fnn_nr.py
"""
import numpy as np
from _figstyle import plt, COL, FIGDIR
import _ch14data as C

DMAX = 10
LEVELS = (0.0,) + C.NOISE
MARK = "os^v<>"


def one(name, out, width):
    d = np.arange(1, DMAX + 1)
    fig, ax = plt.subplots(figsize=(width, width*0.68))
    ins = ax.inset_axes([0.46, 0.22, 0.51, 0.40])
    for j, p in enumerate(LEVELS):
        f = C.fnn_curve(name, p, DMAX)
        lb = "clean" if p == 0 else r"$%d\,\%%$ noise" % round(100*p)
        for a in (ax, ins):
            a.plot(d, f, "-" + MARK[j], color=COL[j % 7], ms=3.2,
                   mfc="none", mew=0.8, label=lb if a is ax else None)
    ax.set_xlabel("embedding dimension $d$")
    ax.set_ylabel(r"$\hat{f}_{\mathrm{nn}}$ (per cent)")
    ax.set_xlim(0.5, DMAX + 0.5)
    ax.set_ylim(-2, 102)
    ax.legend(ncol=2, loc="upper right", handlelength=1.6, columnspacing=1.0)
    ins.set_xlim(3.6, DMAX + 0.4)
    ins.set_ylim(-0.8, 20)
    ins.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGDIR + out)
    plt.close(fig)
    return {p: C.fnn_curve(name, p, DMAX) for p in LEVELS}


if __name__ == "__main__":
    for name, out, w in (("Lorenz", "fnn_lorenz.pdf", 3.5),
                         ("Duffing", "fnn_duffing.pdf", 3.5)):
        r = one(name, out, w)
        print(name, "  Kennel f_hat (per cent), d = 1..%d" % DMAX)
        for p, f in r.items():
            print("  %4.0f %%  " % (100*p) +
                  " ".join("%6.2f" % v for v in f))
