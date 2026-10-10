"""D2Plots2.pdf (label `D2LDs`) -- correlation sums and local slopes for embedding
dimensions d = 1..DMAX, Lorenz (a-b) and Duffing (c-d): the plateau of the slope
stops moving once d is large enough, which is the embedding-dimension criterion of
Section sec:embedfromD2.

Records, tau, w: _ch11data. The plateau value for each d is the mean of the local
slope over the scaling range of D2Plots (RANGE) and is printed for the caption.
"""
from _figstyle import *
from _ch11data import *

DMAX = 6
RANGE = {"Lorenz": (-4.5, -2.0), "Duffing": (-3.5, -1.0)}

fig, axs = plt.subplots(2, 2, figsize=(6.0, 4.4))
for row, name in enumerate(["Lorenz", "Duffing"]):
    a, b = axs[row]
    lo, hi = RANGE[name]
    plateau = []
    for d in range(1, DMAX + 1):
        lnr, lnc = corrsum(name, d)
        ok = np.isfinite(lnc)
        a.plot(lnr[ok], lnc[ok], color=COL[d - 1], lw=0.9, label="$d=%d$" % d)
        r, s = local_slope(lnr, lnc)
        b.plot(r, s, color=COL[d - 1], lw=0.8)
        m = (r >= lo) & (r <= hi)
        plateau.append((s[m].mean(), s[m].std()))
    a.set_xlabel(r"$\ln\epsilon$"); a.set_ylabel(r"$\ln\hat C(\epsilon)$")
    a.set_xlim(-8, 2); a.set_ylim(-20, 1); a.legend(loc="lower right", fontsize=7, ncol=2)
    b.axvspan(lo, hi, color="0.5", alpha=0.12, lw=0)
    b.set_xlabel(r"$\ln\epsilon$"); b.set_ylabel(r"$\mathrm{d}\ln\hat C/\mathrm{d}\ln\epsilon$")
    b.set_xlim(-8, 2); b.set_ylim(0, 4)
    panel(a, "ac"[row], x=-0.22); panel(b, "bd"[row], x=-0.22)
    print(name, "plateau (mean +- sd of local slope over %s) for d = 1..%d:" % (RANGE[name], DMAX))
    print("   " + "  ".join("%.2f+-%.2f" % p for p in plateau))
fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(FIGDIR + "D2Plots2.pdf")
print("wrote D2Plots2.pdf")
