"""Fig. nnd (Fig_6.pdf): nearest-neighbor statistics against embedding dimension for the
six test records, Theiler window w = 3 tau (Table tab:ed_params).
 (a) mean d-dimensional separation <eps_i> with the heuristic eq. (nnds) for Gaussian noise
     and the fits of eq. (nndd) to the Lorenz and Duffing curves (printed);
 (b) mean appended separation <delta_i>, with the w = 0 curves of Lorenz and Duffing dashed
     and the white-noise levels 2/sqrt(pi), 2/sqrt(3) dotted;
 (c) mean ratio <delta_i/eps_i>;
 (d) <delta_i> for the Lorenz record with 0, 5, 10, 20 and 40 % additive Gaussian noise.
Run from python/: writes ../figures/Fig_6.pdf. Data: _ch10data.py (cached).
"""
import time
import numpy as np
from scipy.special import gamma
from scipy.optimize import curve_fit
from _figstyle import *
import _ch10data as C

t0 = time.time()
DMAX = 20
d = np.arange(1, DMAX + 1)
MK = {"Lorenz": "o", "Duffing": "s", "Gaussian": "^", "Uniform": "v", "AR(0.05)": "d",
      "AR(0.5)": "<"}
fig, axs = plt.subplots(2, 2, figsize=(6, 4.6))
(a, b), (c, dd) = axs
for m, name in enumerate(C.RECORDS):
    eps, dlt, _ = C.nn(name, dmax=DMAX)
    me = np.array([e.mean() for e in eps]); md = np.array([e.mean() for e in dlt])
    mr = np.array([np.mean(dl[e > 0]/e[e > 0]) for e, dl in zip(eps, dlt)])
    kw = dict(color=COL[m], marker=MK[name], ms=3, mfc="none", mew=0.7, lw=0.8, label=name)
    a.plot(d, me, **kw); b.plot(d, md, **kw); c.plot(d, mr, **kw)
    if name in ("Lorenz", "Duffing"):
        dmin = 3 if name == "Lorenz" else 4
        f = lambda x, e0, g: e0*(np.exp(g*x) - 1)
        (e0, g), _ = curve_fit(f, d[dmin-1:], me[dmin-1:], p0=(0.03, 0.1))
        print("%s fit of eq. nndd (d >= %d, w = 3 tau): eps0 = %.3f, lambda tau dt = %.3f"
              % (name, dmin, e0, g))
        eps0, dlt0, _ = C.nn(name, dmax=DMAX, w=0)
        md0 = np.array([e.mean() for e in dlt0]); me0 = np.array([e.mean() for e in eps0])
        (e0, g0), _ = curve_fit(f, d[dmin-1:], me0[dmin-1:], p0=(0.03, 0.1),
                                bounds=([0, 1e-4], [10, 2]), maxfev=20000)
        print("   without window: eps0 = %.3f, lambda tau dt = %.3f" % (e0, g0))
        b.plot(d, md0, "--", color=COL[m], lw=0.7)
        print("   <delta> minimum %.3f at d = %d (w=0: %.3f at d = %d)"
              % (md.min(), d[np.argmin(md)], md0.min(), d[np.argmin(md0)]))
    if name in ("Gaussian",):
        print("Gaussian <eps>:", np.round(me, 3)); print("Gaussian <delta>:", np.round(md, 3))
heur = np.sqrt(2)*gamma((d + 1)/2)/gamma(d/2)*(np.sqrt(d)/C.N)**(1/d)
a.plot(d, heur, "k--", lw=0.8, label="eq. (nnds)")
b.axhline(2/np.sqrt(np.pi), color="k", ls=":", lw=0.7); b.axhline(2/np.sqrt(3), color="k", ls=":", lw=0.7)
a.set_ylabel(r"$\langle\epsilon_i\rangle/\sigma_x$"); a.legend(ncol=2, fontsize=7, loc="upper left")
b.set_ylabel(r"$\langle\delta_i\rangle/\sigma_x$"); b.set_yscale("log"); b.set_ylim(0.008, 1.5)
c.set_ylabel(r"$\langle\delta_i/\epsilon_i\rangle$"); c.set_yscale("log")
for ax in (a, b, c):
    ax.set_xlim(0, DMAX + 1); ax.set_xlabel("$d$")
# (d) noisy Lorenz
for m, p in enumerate([0, 0.05, 0.1, 0.2, 0.4]):
    eps, dlt, _ = C.nn("Lorenz", noise=p, dmax=12)
    md = np.array([e.mean() for e in dlt])
    dd.plot(np.arange(1, 13), md, "-o", color=COL[m], ms=3, mfc="none", mew=0.7, lw=0.8,
            label="%d %%" % (100*p) if p else "clean")
    print("Lorenz noise %3.0f %%: <delta> min %.3f at d=%d, at d=12: %.3f"
          % (100*p, md.min(), 1 + np.argmin(md), md[-1]))
dd.set_yscale("log"); dd.set_ylim(0.008, 6); dd.set_xlim(0, 13); dd.set_xlabel("$d$")
dd.set_xticks([0, 2, 4, 6, 8, 10, 12])
dd.set_ylabel(r"$\langle\delta_i\rangle/\sigma_x$"); dd.legend(fontsize=7, loc="upper right", ncol=3)
for ax, let in zip(axs.ravel(), "abcd"):
    panel(ax, let, x=-0.16, y=1.0)
fig.tight_layout(pad=0.4)
fig.savefig(FIGDIR + "Fig_6.pdf")
print("mkfig_nnd: %.0f s" % (time.time() - t0))
