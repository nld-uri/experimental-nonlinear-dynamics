"""D2Plots.pdf (label `scaling`) -- correlation sum and local slope for the Lorenz
and Duffing records, with the scaling ranges shaded and the fitted D2 marked.

Records, tau, w, d: _ch11data (Lorenz N = 20 000, dt = 0.02, tau = 8, w = 24, d = 3;
Duffing N = 20 000, dt = 0.1, tau = 14, w = 42, d = 4), unit variance, 101 ball
sizes with ln eps from -8 to 2. The scaling ranges RANGE were read off the local
slope plateaus by hand and are stated in the caption. Printed: least-squares slope
+- standard error over the range, mean +- sd of the local slopes, and Takens'
estimator at three eps_0 spanning the upper half of the range.
"""
from _figstyle import *
from _ch11data import *

RANGE = {"Lorenz": (-4.5, -2.0), "Duffing": (-3.5, -1.0)}

fig, axs = plt.subplots(2, 2, figsize=(6.0, 4.4))
out = {}
for row, name in enumerate(["Lorenz", "Duffing"]):
    d = DEMB[name]
    lnr, lnc = corrsum(name, d)
    lo, hi = RANGE[name]
    slope, se, ms, ss = fit_range(lnr, lnc, lo, hi)
    tk = [takens(name, d, np.exp(e0)) for e0 in np.linspace((lo + hi)/2, hi, 3)]
    out[name] = (slope, se, ms, ss, tk)
    a, b = axs[row]
    ok = np.isfinite(lnc)
    a.plot(lnr[ok], lnc[ok], color=COL[0])
    a.axvspan(lo, hi, color=COL[0], alpha=0.12, lw=0)
    # least-squares line through the scaling range, extended a little
    m = ok & (lnr >= lo) & (lnr <= hi)
    c0 = np.mean(lnc[m] - slope*lnr[m])
    xx = np.array([lo - 1.5, hi + 1.5])
    a.plot(xx, slope*xx + c0, "--", color=COL[1], lw=0.8)
    a.set_xlabel(r"$\ln\epsilon$"); a.set_ylabel(r"$\ln\hat C(\epsilon)$")
    a.set_xlim(-8, 2); a.set_ylim(-20, 1)
    r, s = local_slope(lnr, lnc)
    b.plot(r, s, ".-", color=COL[0], ms=2.5, lw=0.6)
    b.axvspan(lo, hi, color=COL[0], alpha=0.12, lw=0)
    b.axhline(slope, color=COL[1], ls="--", lw=0.8)
    b.text(1.9, slope + 0.08, r"$\hat D_2=%.2f$" % slope, ha="right", va="bottom",
           color=COL[1], fontsize=8)
    b.set_xlabel(r"$\ln\epsilon$")
    b.set_ylabel(r"$\mathrm{d}\ln\hat C/\mathrm{d}\ln\epsilon$")
    b.set_xlim(-8, 2); b.set_ylim(0, 4)
    panel(a, "ac"[row], x=-0.22); panel(b, "bd"[row], x=-0.22)
fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(FIGDIR + "D2Plots.pdf")
for name, (slope, se, ms, ss, tk) in out.items():
    print("%s d=%d range %s: LS slope %.3f +- %.3f (SE); local slopes %.3f +- %.3f (sd)"
          % (name, DEMB[name], RANGE[name], slope, se, ms, ss))
    for (D, e, M) in tk:
        print("   Takens: D_T = %.3f formal +- %.4f from M = %d pairs" % (D, e, M))
print("wrote D2Plots.pdf")
