"""DmuLDs.pdf (label `Dmu`) -- average pointwise dimension by the fixed-distance
and fixed-mass algorithms (pwdim) for the Lorenz (a-b) and Duffing (c-d) records.

(a),(c): <ln mu_i(eps)> against ln eps (fixed distance) and ln(n/N) against
<ln eps_i(n)> (fixed mass, Grassberger's psi(n) - ln N in place of ln(n/N));
(b),(d): their local slopes, the estimates of D_P. Records, tau, w, d as in
_ch11data (Lorenz d = 3, Duffing d = 4); pwdim with k = 500 neighbours / bins and
Nq = 4096 random reference points, seed 0, records normalized to unit variance.
The scaling ranges RANGE were read off the slope plateaus and are shaded; the
mean +- sd of the local slopes over them is printed for the caption.
"""
from _figstyle import *
from _ch11data import *

RANGE = {"Lorenz": {"fd": (-3.0, -1.5), "fm": (-3.0, -1.5)},
         "Duffing": {"fd": (-2.2, -1.0), "fm": (-2.2, -1.0)}}
LAB = {"fd": "fixed distance", "fm": "fixed mass"}

fig, axs = plt.subplots(2, 2, figsize=(6.0, 4.4))
for row, name in enumerate(["Lorenz", "Duffing"]):
    a, b = axs[row]
    lnD, lnN = pointwise(name, DEMB[name])
    res = {}
    for c, key in enumerate(["fd", "fm"]):
        x, y = lnD[:, c], lnN[:, c]
        ok = np.isfinite(y) & (y != 0)
        x, y = x[ok], y[ok]
        a.plot(x, y, color=COL[c], lw=1.0, label=LAB[key])
        s = np.gradient(y, x)
        b.plot(x, s, color=COL[c], lw=0.9, label=LAB[key])
        lo, hi = RANGE[name][key]
        m = (x >= lo) & (x <= hi)
        res[key] = (s[m].mean(), s[m].std())
        b.axvspan(lo, hi, color=COL[c], alpha=0.10, lw=0)
    a.set_xlabel(r"$\ln\epsilon,\ \langle\ln\epsilon_i(n)\rangle$")
    a.set_ylabel(r"$\langle\ln\hat\mu_i(\epsilon)\rangle,\ \psi(n)-\ln N$")
    a.legend(loc="lower right", fontsize=7)
    b.set_xlabel(r"$\ln\epsilon,\ \langle\ln\epsilon_i(n)\rangle$")
    b.set_ylabel(r"$\hat D_P,\ \bar D_P$")
    b.set_ylim(0, 3)
    b.legend(loc="lower left", fontsize=7)
    panel(a, "ac"[row], x=-0.22); panel(b, "bd"[row], x=-0.22)
    print(name, "d = %d:" % DEMB[name],
          "  ".join("%s over %s: %.2f +- %.2f" % (LAB[k], RANGE[name][k], *res[k]) for k in res))
fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(FIGDIR + "DmuLDs.pdf")
print("wrote DmuLDs.pdf")
