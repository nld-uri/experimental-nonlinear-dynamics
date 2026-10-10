"""Fig. CFNN (Fig_10.pdf): composite FNN fraction R_nn(d; r, s), eq. (new), k = 1, l = 0,
Theiler window w = 3 tau, for the six test records. Rows 1-2: s = 0.5 and r = 2, 4, ..., 20;
rows 3-4: r = 4 and s = 0.1, 0.2, ..., 1. Circles: surrogate fractions.
Run from python/: writes ../figures/Fig_10.pdf. Data: _ch10data.py (cached).
"""
import time
import numpy as np
from _figstyle import *
import _ch10data as C

t0 = time.time()
RS, SS, DMAX = np.arange(2, 21, 2), np.arange(0.1, 1.01, 0.1), 20
fig, axs = plt.subplots(4, 3, figsize=(6, 6.4), sharex=True, sharey=True)
letters = iter("abcdefghijkl")
for blk, (rs, ss) in enumerate([(RS, 0.5), (4, SS)]):
    for m, name in enumerate(C.RECORDS):
        ax = axs[2*blk + m//3, m % 3]
        eps, dlt, dsur = C.nn(name, dmax=DMAX)
        fr, fs = C.sweep(eps, dlt, "comp", rs, ss), C.sweep(eps, dsur, "comp", rs, ss)
        C.family(ax, fr, fs)
        panel(ax, next(letters), x=-0.08, y=1.01)
        ax.set_xticks([0, 5, 10, 15, 20])
        if blk == 0:
            print("%-9s r-sweep, s=0.5: floor at d=20 data %.2f-%.2f, surrogate %.2f-%.2f; "
                  "first d with R<0.01 at r=10: %s" % (name, fr[:, -1].min(), fr[:, -1].max(),
                  fs[:, -1].min(), fs[:, -1].max(), 1 + np.argmax(fr[4] < 0.01) if (fr[4] < 0.01).any() else "-"))
            print("          minimum over d (r=10): %.3f at d=%d" % (fr[4].min(), 1 + np.argmin(fr[4])))
        else:
            print("%-9s s-sweep, r=4: R(d=20) at s=0.5: data %.2f, surrogate %.2f; s=1: %.2f / %.2f"
                  % (name, fr[4, -1], fs[4, -1], fr[9, -1], fs[9, -1]))
for ax in axs[:2, 0]:
    ax.set_ylabel(r"$R_{nn}(d;r,0.5)$")
for ax in axs[2:, 0]:
    ax.set_ylabel(r"$R_{nn}(d;4,s)$")
for ax in axs[:3].ravel():
    ax.set_xlabel("")
fig.tight_layout(pad=0.3, h_pad=1.0)
fig.savefig(FIGDIR + "Fig_10.pdf")
print("mkfig_CFNN: %.0f s" % (time.time() - t0))
