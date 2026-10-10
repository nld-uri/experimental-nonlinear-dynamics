"""Fig. caoE (caoE.pdf): Cao's E1(d) (a) and E2(d) (b), eqs. (eq:ed_caoE1)-(eq:ed_caoE2),
maximum norm, nearest neighbor outside the Theiler window w = 3 tau, for the Lorenz,
Duffing, Henon and Gaussian-noise records of Table tab:ed_params (N = 20 000).
Run from python/: writes ../figures/caoE.pdf and prints the values for the text.
"""
import time
import numpy as np
from _figstyle import *
import _ch10data as C
import fnn as F

t0 = time.time()
DMAX = 10
d = np.arange(1, DMAX + 1)
fig, (a, b) = plt.subplots(1, 2, figsize=(6, 2.4))
MK = {"Lorenz": "o", "Duffing": "s", "Henon": "^", "Gaussian": "v"}
for m, name in enumerate(["Lorenz", "Duffing", "Henon", "Gaussian"]):
    tau, w = C.tau_w(name)
    E1, E2 = F.cao(C.record(name), tau, DMAX, w)
    kw = dict(color=COL[m], marker=MK[name], ms=3.5, mfc="none", mew=0.7, lw=0.8,
              label=name.replace("Henon", "H\u00e9non"))
    a.plot(d, E1, **kw); b.plot(d, E2, **kw)
    print("%-9s tau=%2d w=%2d  E1:" % (name, tau, w), np.round(E1, 2))
    print("                    E2:", np.round(E2, 2))
a.axhline(1, color="k", ls=":", lw=0.6); b.axhline(1, color="k", ls=":", lw=0.6)
a.set_ylabel("$E_1(d)$"); b.set_ylabel("$E_2(d)$")
a.set_ylim(0, 1.1); b.set_ylim(0.4, 1.7)
for ax, let in zip((a, b), "ab"):
    ax.set_xlabel("$d$"); ax.set_xlim(0, DMAX + 1); ax.set_xticks(d)
    panel(ax, let, x=-0.18, y=1.0)
a.legend(fontsize=7, loc="lower right")
fig.tight_layout(pad=0.4)
fig.savefig(FIGDIR + "caoE.pdf")
print("mkfig_caoE: %.0f s" % (time.time() - t0))
