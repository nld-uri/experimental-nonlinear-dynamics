"""Fig. FNN_Noise (Fig_12.pdf): the FNN fractions on noisy records. Rows: Lorenz with 10 %
and 40 % additive Gaussian noise, Duffing with 10 % and 40 %. Columns: (left) Kennel
fraction, eq. (OrKennel), a = 2, r = 2, 4, ..., 20; (middle) composite fraction eq. (new)
at r = 10, s = 1 with k = 1, l = 0; (right) the same with k = 10, l = tau, eq. (CFrac).
Theiler window w = 3 tau; circles: surrogates. Run from python/: writes ../figures/Fig_12.pdf.
"""
import time
import numpy as np
from _figstyle import *
import _ch10data as C

t0 = time.time()
RS, DMAX = np.arange(2, 21, 2), 12
CASES = [("Lorenz", 0.1), ("Lorenz", 0.4), ("Duffing", 0.1), ("Duffing", 0.4)]
fig, axs = plt.subplots(4, 3, figsize=(6, 6.4), sharex=True, sharey=True)
letters = iter("abcdefghijkl")
for row, (name, p) in enumerate(CASES):
    tau, w = C.tau_w(name)
    eps, dlt, dsur = C.nn(name, noise=p, dmax=DMAX)
    C.family(axs[row, 0], C.sweep(eps, dlt, "kennel", RS, 0.5), C.sweep(eps, dsur, "kennel", RS, 0.5))
    f1, s1 = C.sweep(eps, dlt, "comp", 10, 1.0), C.sweep(eps, dsur, "comp", 10, 1.0)
    C.family(axs[row, 1], f1, s1)
    eps, dlt, dsur = C.nn(name, noise=p, dmax=DMAX, k=10, l=tau)
    f10, s10 = C.sweep(eps, dlt, "comp", 10, 1.0), C.sweep(eps, dsur, "comp", 10, 1.0)
    C.family(axs[row, 2], f10, s10)
    for ax in axs[row]:
        panel(ax, next(letters), x=-0.08, y=1.01)
    print("%-8s %3.0f %%  k=1 : R_nn(d)=" % (name, 100*p), np.round(f1[0], 2))
    print("               k=10: R_nn(d)=", np.round(f10[0], 2), " surrogate floor %.2f" % s10[0, -1])
for ax, lab in zip(axs[:, 0], [r"$\hat f_{nn}(d;r)$"]*4):
    ax.set_ylabel(lab)
for ax in axs[:, 1]:
    ax.set_ylabel(r"$R_{nn}(d;10,1)$")
for ax in axs[:, 2]:
    ax.set_ylabel(r"$R_{nn}(d;10,1)$")
for ax in axs[:3].ravel():
    ax.set_xlabel("")
for ax in axs.ravel():
    ax.set_xticks([0, 3, 6, 9, 12])
fig.tight_layout(pad=0.3, h_pad=1.0)
fig.savefig(FIGDIR + "Fig_12.pdf")
print("mkfig_FNN_Noise: %.0f s" % (time.time() - t0))
