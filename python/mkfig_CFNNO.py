"""Fig. CFNNO (Fig_3.pdf): Kennel fraction of false nearest neighbors, eq. (OrKennel),
a = 2, r = 2, 4, ..., 20, for the six test records of Chapter 10, with the Theiler
window w = 3 tau of Table tab:ed_params, and the surrogate fractions.
Run from python/: writes ../figures/Fig_3.pdf. Data: _ch10data.py (cached).
"""
import time
import numpy as np
from _figstyle import *
import _ch10data as C

t0 = time.time()
RS = np.arange(2, 21, 2)
DMAX = 20
fig, axs = plt.subplots(2, 3, figsize=(6, 3.8), sharex=True, sharey=True)
for ax, name, let in zip(axs.ravel(), C.RECORDS, "abcdef"):
    eps, dlt, dsur = C.nn(name, dmax=DMAX)
    fr = C.sweep(eps, dlt, "kennel", RS, 0.5)
    fs = C.sweep(eps, dsur, "kennel", RS, 0.5)
    C.family(ax, fr, fs)
    panel(ax, let, x=-0.08, y=1.01)
    ax.set_xticks([0, 5, 10, 15, 20])
    if name == "Gaussian":
        print("Gaussian Kennel fraction, r=10:", np.round(fr[4], 2))
    if name in ("Lorenz", "Duffing"):
        print(name, "first zero (r=10):", 1 + np.argmax(fr[4] < 0.005))
for ax in axs[:, 0]:
    ax.set_ylabel(r"$\hat f_{nn}(d;r)$")
for ax in axs[0]:
    ax.set_xlabel("")
fig.tight_layout(pad=0.3, h_pad=1.2)
fig.savefig(FIGDIR + "Fig_3.pdf")
print("mkfig_CFNNO: %.0f s" % (time.time() - t0))
