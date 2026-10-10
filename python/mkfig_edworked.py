"""Fig. edworked (edworked.pdf): the procedure of Section sec:ed_procedure applied to the
Lorenz x record with 20 % additive Gaussian noise (N = 20 000, dt = 0.02, tau = 8, w = 24).
 (a) composite fraction R_nn(d; r, 0.5), k = 1, r = 2, 4, ..., 20 (x), the simple surrogate
     of Section sec:ed_records (circles), and the AAFT surrogate of the whole record
     (irsurr, 20 iterations) at r = 10 (dashed); d_max = ln N / ln 10 dotted vertical line;
 (b) the same with k = 10, l = tau;
 (c) Cao's E1 (filled) and E2 (open) with the same tau and w;
 (d) delay check: R_nn(d; 10, 0.5), k = 10, l = tau, at tau/2, tau and 2 tau (w fixed).
Run from python/: writes ../figures/edworked.pdf and prints the report quantities.
"""
import time
import numpy as np
from _figstyle import *
import _ch10data as C
import fnn as F
from irsurr import irsurr

t0 = time.time()
NAME, P, DMAX = "Lorenz", 0.2, 12
RS = np.arange(2, 21, 2)
tau, w = C.tau_w(NAME)
d = np.arange(1, DMAX + 1)
dmax_bound = np.log(C.N)/np.log(10)
x = C.record(NAME, P)
fig, axs = plt.subplots(2, 2, figsize=(6, 4.6))
(a, b), (c, dd) = axs

# AAFT surrogate of the noisy record, k = 1
xs = irsurr(x, 20, np.random.default_rng(0))[0]
ea, da, _ = F.fnn(xs, tau, DMAX, w)
fa = F.fnnfrac(ea, da, r=10, s=0.5)["comp"]
for ax, k, l in [(a, 1, 0), (b, 10, tau)]:
    eps, dlt, dsur = C.nn(NAME, noise=P, dmax=DMAX, k=k, l=l)
    fr, fs = C.sweep(eps, dlt, "comp", RS, 0.5), C.sweep(eps, dsur, "comp", RS, 0.5)
    C.family(ax, fr, fs)
    ax.plot(d, fa, "--", color=COL[2], lw=1.0)
    ax.axvline(dmax_bound, color="k", ls=":", lw=0.7)
    ax.set_ylabel(r"$R_{nn}(d;r,0.5)$")
    print("k=%2d l=%2d: R_nn(d;10,0.5) =" % (k, l), np.round(fr[4], 3))
    print("            surrogate        =", np.round(fs[4], 3))
    print("            all r>=4 at d=3..5:", np.round(fr[1:, 2:5].max(axis=0), 3))
print("AAFT surrogate R_nn(d;10,0.5) =", np.round(fa, 3))
print("d_max = ln N/ln 10 = %.2f" % dmax_bound)

# (c) Cao
E1, E2 = F.cao(x, tau, DMAX, w)
c.plot(d, E1, "-o", color=COL[0], ms=3.5, lw=0.8, label="$E_1$")
c.plot(d, E2, "-s", color=COL[1], ms=3.5, mfc="none", lw=0.8, label="$E_2$")
c.axhline(1, color="k", ls=":", lw=0.6); c.set_ylim(0, 1.5); c.legend(fontsize=7, loc="lower right")
c.set_ylabel("$E_1(d)$, $E_2(d)$")
print("Cao E1:", np.round(E1, 2)); print("Cao E2:", np.round(E2, 2))

# (d) delay check
for m, t in enumerate([tau//2, tau, 2*tau]):
    eps, dlt, dsur = C.nn(NAME, noise=P, dmax=DMAX, k=10, l=t, tau=t, w=w)
    fr = F.fnnfrac(eps, dlt, r=10, s=0.5)["comp"]
    dd.plot(d, fr, "-x", color=COL[m], ms=3.5, lw=0.8, label=r"$\tau=%d$" % t)
    print("tau=%2d: R_nn(d;10,0.5), k=10 =" % t, np.round(fr, 3))
dd.set_ylim(-0.02, 1.02); dd.set_ylabel(r"$R_{nn}(d;10,0.5)$"); dd.legend(fontsize=7)
for ax, let in zip(axs.ravel(), "abcd"):
    ax.set_xlim(0, DMAX + 1); ax.set_xticks([0, 2, 4, 6, 8, 10, 12]); ax.set_xlabel("$d$")
    panel(ax, let, x=-0.16, y=1.0)
fig.tight_layout(pad=0.4)
fig.savefig(FIGDIR + "edworked.pdf")
print("mkfig_edworked: %.0f s" % (time.time() - t0))
