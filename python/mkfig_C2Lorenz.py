"""C2Lorenz.pdf -- the three regions of ln C(eps) vs ln eps on one record.

Lorenz x record of _ch11data (N = 20 000, dt = 0.02) with NOISE * sigma_x of Gaussian
measurement noise added so that a noise-dominated region exists at small eps,
embedded with d = 3, tau = 8, Theiler window w = 24. The scaling range RANGE is
shaded, the least-squares line through it is drawn, and the slope +- standard
error is printed and written on the figure. Below the shaded range the local slope
rises toward d (noise fills the reconstruction space); above it, C -> 1.
"""
from _figstyle import *
from _ch11data import *

NOISE = 0.003
RANGE = (-3.5, -2.0)
d = DEMB["Lorenz"]
lnr, lnc = corrsum("Lorenz", d, noise=NOISE)
slope, se, ms, ss = fit_range(lnr, lnc, *RANGE)
r, s = local_slope(lnr, lnc)
noise_sl = s[(r > -6) & (r < -4.5)]
print("slope in scaling range %.3f +- %.3f (SE), local slopes %.3f +- %.3f; "
      "local slope in the noise region [-6,-4.5]: %.2f..%.2f (mean %.2f)"
      % (slope, se, ms, ss, noise_sl.min(), noise_sl.max(), noise_sl.mean()))

fig, ax = plt.subplots(figsize=(4.4, 3.2))
ok = np.isfinite(lnc)
ax.plot(lnr[ok], lnc[ok], color=COL[0], lw=1.3)
ax.axvspan(*RANGE, color=COL[0], alpha=0.12, lw=0)
m = ok & (lnr >= RANGE[0]) & (lnr <= RANGE[1])
c0 = np.mean(lnc[m] - slope*lnr[m])
xx = np.array([-6.0, 0.5])
ax.plot(xx, slope*xx + c0, "--", color=COL[1], lw=0.9)
ax.text(-1.4, slope*(-1.4) + c0 - 0.4, r"$\hat D_2 = %.3f\pm%.3f$" % (slope, se), color=COL[1],
        fontsize=8.5, ha="left", va="top")
# slope triangle
x0 = -3.3; y0 = slope*x0 + c0
ax.plot([x0, x0 + 1, x0 + 1], [y0, y0, y0 + slope], color="k", lw=0.7)
ax.text(x0 + 0.5, y0 - 0.3, "$1$", ha="center", va="top", fontsize=8)
ax.text(x0 + 1.1, y0 + slope/2, r"$D\approx D_2$", ha="left", va="center", fontsize=8)
ax.text(-6.6, -9.5, "noise\ndominated\n$D > D_2$", ha="center", va="center", fontsize=8.5)
ax.annotate("", xy=(RANGE[0], -17.5), xytext=(RANGE[1], -17.5),
            arrowprops=dict(arrowstyle="<->", lw=0.7))
ax.text(np.mean(RANGE), -17.2, "scaling region", ha="center", va="bottom", fontsize=8.5)
ax.text(0.9, -2.2, r"$\hat C\to 1$", ha="center", fontsize=8.5)
ax.set_xlabel(r"$\ln\epsilon$"); ax.set_ylabel(r"$\ln\hat C(\epsilon)$")
ax.set_xlim(-7.6, 2); ax.set_ylim(-19, 0.5)
fig.tight_layout(pad=0.3)
fig.savefig(FIGDIR + "C2Lorenz.pdf")
print("wrote C2Lorenz.pdf  noise = %g sigma_x, ln(noise) = %.2f" % (NOISE, np.log(NOISE)))
