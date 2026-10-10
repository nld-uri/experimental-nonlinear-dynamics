"""Fig. edtwowell (edtwowell.pdf): embedding dimension of the two-well oscillator record
of Section sec:twowelldata (N = 96 000 samples, tau = 5 samples, Theiler window w = 32).

 (a) Kennel fraction f_hat(d; r) of eq. (OrKennel), a = 2, r = 2, 4, ..., 20 (x), with the
     simple surrogate of Section sec:ed_records (circles); dotted vertical line d_max =
     ln N / ln 10;
 (b) composite fraction R_nn(d; r, 0.5) of eq. (new), k = 1, l = 0, same r family and
     surrogate, and the IAAFT surrogate of the whole record (irsurr, 20 iterations) at
     r = 10 (dashed);
 (c) Cao's E1 and E2 for the record (filled) and for the same IAAFT surrogate (open);
 (d) delay and strand checks: R_nn(d; 10, 0.5) with k = 1 at tau = 3, 5 and 10 (w = 32
     fixed) and with k = 10, l = tau at tau = 5.

Run from python/: writes ../figures/edtwowell.pdf and prints every reported quantity.
Runtime about 8 minutes on two cores.
"""
import time
import numpy as np
from _figstyle import *
from _twowell import twowell
import fnn as F
from irsurr import irsurr

t0 = time.time()
TAU, W, DMAX = 5, 32, 8
RS = np.arange(2, 21, 2)
d = np.arange(1, DMAX + 1)
x = twowell()
N = x.size
dmax_bound = np.log(N)/np.log(10)
print("N = %d, tau = %d, w = %d, d_max = ln N/ln 10 = %.2f" % (N, TAU, W, dmax_bound))


def family(ax, curves, sur=None):
    for c in curves:
        ax.plot(d, c, "-x", color=COL[0], lw=0.6, ms=3, mew=0.6)
    if sur is not None:
        for c in sur:
            ax.plot(d, c, "-o", color=COL[1], lw=0.6, ms=2.5, mfc="none", mew=0.6)
    ax.set_xlim(0, DMAX + 1); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("$d$")


def sweep(eps, dlt, which, rs, s=0.5):
    return np.array([F.fnnfrac(eps, dlt, r=r, s=s)[which] for r in rs])


fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.8))
(a, b), (c, dd) = axs

eps, dlt, dsur = F.fnn(x, TAU, DMAX, W, 1, 0)
for ax, which, lab in [(a, "kennel", r"$\hat{f}_{nn}(d;r)$"),
                       (b, "comp", r"$R_{nn}(d;r,0.5)$")]:
    family(ax, sweep(eps, dlt, which, RS), sweep(eps, dsur, which, RS))
    ax.axvline(dmax_bound, color="k", ls=":", lw=0.7)
    ax.set_ylabel(lab)
    print("%-7s r=10:" % which, np.round(F.fnnfrac(eps, dlt, r=10, s=0.5)[which], 4))
    print("%-7s r=10 simple surrogate:" % which,
          np.round(F.fnnfrac(eps, dsur, r=10, s=0.5)[which], 3))
print("f_hat over all r>=4, d=3..6:",
      np.round(sweep(eps, dlt, "kennel", RS[1:])[:, 2:6].max(axis=0), 4))
print("R_nn  over all r>=4, d=3..6:",
      np.round(sweep(eps, dlt, "comp", RS[1:])[:, 2:6].max(axis=0), 4))
print("R_nn  over s=0.1..1 at r=4, d=4:",
      np.round([F.fnnfrac(eps, dlt, r=4, s=s)["comp"][3] for s in
                (0.1, 0.3, 0.5, 1.0)], 4))

# IAAFT surrogate of the whole record: composite fraction and Cao statistics
xs = irsurr(x, 20, np.random.default_rng(1))[0]
es, ds, _ = F.fnn(xs, TAU, DMAX, W, 1, 0)
fs_ = F.fnnfrac(es, ds, r=10, s=0.5)["comp"]
b.plot(d, fs_, "--", color=COL[2], lw=1.0)
print("IAAFT surrogate R_nn(d;10,0.5):", np.round(fs_, 3))

# (c) Cao
E1, E2 = F.cao(x, TAU, DMAX, W)
E1s, E2s = F.cao(xs, TAU, DMAX, W)
c.plot(d, E1, "-o", color=COL[0], ms=3.5, lw=0.8, label="$E_1$")
c.plot(d, E2, "-s", color=COL[1], ms=3.5, lw=0.8, label="$E_2$")
c.plot(d, E1s, "--o", color=COL[0], ms=3.5, mfc="none", lw=0.8, label="$E_1$ surr.")
c.plot(d, E2s, "--s", color=COL[1], ms=3.5, mfc="none", lw=0.8, label="$E_2$ surr.")
c.axhline(1, color="k", ls=":", lw=0.6); c.set_ylim(0, 1.5)
c.legend(fontsize=7, loc="lower right", ncol=2)
c.set_xlim(0, DMAX + 1); c.set_xlabel("$d$"); c.set_ylabel("$E_1(d)$, $E_2(d)$")
print("Cao E1:", np.round(E1, 3)); print("Cao E2:", np.round(E2, 3))
print("Cao E1 surrogate:", np.round(E1s, 3)); print("Cao E2 surrogate:", np.round(E2s, 3))

# (d) delay check and strand selection
for m, t in enumerate([3, TAU, 2*TAU]):
    e, dl, _ = (eps, dlt, dsur) if t == TAU else F.fnn(x, t, DMAX, W, 1, 0)
    fr = F.fnnfrac(e, dl, r=10, s=0.5)["comp"]
    dd.plot(d, fr, "-x", color=COL[m], ms=3.5, lw=0.8, label=r"$\tau=%d$, $k=1$" % t)
    print("tau=%2d k=1  R_nn(d;10,0.5):" % t, np.round(fr, 4))
e10, d10, _ = F.fnn(x, TAU, DMAX, W, 10, TAU)
fr10 = F.fnnfrac(e10, d10, r=10, s=0.5)["comp"]
dd.plot(d, fr10, "-s", color=COL[3], ms=3.5, mfc="none", lw=0.8,
        label=r"$\tau=5$, $k=10$, $l=\tau$")
print("tau= 5 k=10 R_nn(d;10,0.5):", np.round(fr10, 4))
dd.set_xlim(0, DMAX + 1); dd.set_ylim(-0.02, 1.02); dd.set_xlabel("$d$")
dd.set_ylabel(r"$R_{nn}(d;10,0.5)$"); dd.legend(fontsize=7)

for ax, let in zip(axs.ravel(), "abcd"):
    ax.set_xticks(np.arange(0, DMAX + 1, 2))
    panel(ax, let, x=-0.16, y=1.0)
fig.tight_layout(pad=0.4)
fig.savefig(FIGDIR + "edtwowell.pdf")
print("mkfig_edtwowell: %.0f s" % (time.time() - t0))
