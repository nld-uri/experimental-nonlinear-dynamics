"""twowell_d2.pdf (label `twowelld2`) -- correlation sum and local slope of the
two-well oscillator record of Section 1.4.1, embedded with tau = 5 samples in
d = 2,...,8.

Settings: c2r(x, d, 5, n=30000, r=R, p=5*d) with R = 76 ball sizes log-spaced
over ln r in [-6.5, 1.0], the record normalized to zero mean and unit variance
inside c2r, so r is in units of the standard deviation of the record. The
scaling range ln r in [-1.45, -0.35] was read off the local-slope plateau.

Printed: least-squares slope over the scaling range for every d, the mean and
scatter of the local slopes there, the depth and position of the dip below the
range, and the pair count at the small-scale end. Also printed is the control
experiment of the text: the two-well Duffing model of Section 1.4 integrated
noise-free and sampled at 16 and at 64 points per forcing period, showing that
the dip is produced by the sampling rate and not by the measurement.

Runtime about twenty minutes: c2r is O(n^2).
"""
from _figstyle import *
from _twowell import twowell
from c2r import c2r

LO, HI = -1.45, -0.35            # scaling range in ln r, units of sigma
DS = list(range(2, 9))
R = np.exp(np.linspace(-6.5, 1.0, 76))
NPTS = 30000

x = twowell()
res, fit = {}, {}
for d in DS:
    lnr, lnc = c2r(x, d, 5, n=NPTS, r=R, p=5*d, rng=np.random.default_rng(7))
    g = np.isfinite(lnc)
    a, c = lnr[g], lnc[g]
    s = np.gradient(c, a)
    res[d] = (a, c, s)
    m = (a >= LO) & (a <= HI)
    fit[d] = (np.polyfit(a[m], c[m], 1)[0], s[m].mean(), s[m].std())

fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.6))
for k, d in enumerate(DS):
    a, c, s = res[d]
    col = COL[k % 7]
    axs[0].plot(a, c, color=col, lw=0.9, label=r"$d=%d$" % d)
    axs[1].plot(a, s, ".-", color=col, ms=2.5, lw=0.7)
for ax in axs:
    ax.axvspan(LO, HI, color="0.5", alpha=0.18, lw=0)
    ax.set_xlabel(r"$\ln (\epsilon/\sigma)$")
    ax.set_xlim(-6.5, 1.0)
axs[0].set_ylabel(r"$\ln\hat C(\epsilon)$")
axs[0].set_ylim(-20, 1)
axs[0].legend(loc="lower right", ncol=2, columnspacing=1.0, handlelength=1.4,
              fontsize=7)
axs[1].axhline(2.68, color="k", ls="--", lw=0.7)
axs[1].text(0.9, 2.75, r"$\hat D_2 = 2.68$", fontsize=8, ha="right")
axs[1].set_ylabel(r"$\mathrm{d}\ln\hat C/\mathrm{d}\ln\epsilon$")
axs[1].set_ylim(0.5, 4.0)
panel(axs[0], "a", x=-0.20)
panel(axs[1], "b", x=-0.20)
fig.tight_layout(pad=0.4, w_pad=1.4)
fig.savefig(FIGDIR + "twowell_d2.pdf")

npair = (NPTS - 1)*(NPTS - 2)/2
print("scaling range ln r in [%.2f, %.2f], r/sigma %.2f to %.2f"
      % (LO, HI, np.exp(LO), np.exp(HI)))
for d in DS:
    a, c, s = res[d]
    k = (a > -4.35) & (a < -2.0)
    print("d=%d  D2(LS) %.4f  local %.4f +- %.4f | dip %.3f at ln r = %.2f"
          % (d, fit[d][0], fit[d][1], fit[d][2], s[k].min(), a[k][np.argmin(s[k])]))
v = np.array([fit[d][0] for d in (4, 5, 6, 7, 8)])
print("d = 4..8: mean %.3f, half-spread %.3f" % (v.mean(), (v.max() - v.min())/2))
a, c, s = res[5]
for e in (-4.3, -5.3, -6.3):
    j = np.argmin(abs(a - e))
    print("d=5: at ln r = %.1f, ln C = %.2f, about %d pairs of %.2g"
          % (e, c[j], np.exp(c[j])*npair, npair))

# ---- control: the model sampled at two rates, no measurement noise ----
from scipy.integrate import solve_ivp
T = 2*np.pi


def duffing(npp, ncyc):
    ts = np.arange(0, ncyc*T, T/npp)
    sol = solve_ivp(lambda t, u: [u[1], -0.25*u[1] + u[0] - u[0]**3 + 0.3*np.cos(t)],
                    [0, ts[-1] + 1], [-0.6708, -0.3693], t_eval=ts,
                    rtol=1e-9, atol=1e-10, max_step=0.05)
    return sol.y[0][200*npp:]


Rc = np.exp(np.linspace(-7.0, 0.5, 51))
for npp, ncyc, tau in ((16, 6000, 5), (64, 1600, 20)):
    lnr, lnc = c2r(duffing(npp, ncyc), 5, tau, n=20000, r=Rc, p=25,
                   rng=np.random.default_rng(7))
    g = np.isfinite(lnc)
    a, s = lnr[g], np.gradient(lnc[g], lnr[g])
    m = (a >= LO) & (a <= HI)
    below = a[(a < LO) & (s < 1.7)]
    print("Duffing model, %d samples per forcing period, tau = %d: plateau %.3f,"
          " leaves it at ln r = %.2f, level below %.2f"
          % (npp, tau, s[m].mean(), below.max(), s[(a > -6) & (a < below.max())].mean()))

# ---- the record in halves, same settings at d = 5 ----
for nm, seg in (("first half", x[:48000]), ("second half", x[48000:])):
    lnr, lnc = c2r(seg, 5, 5, n=15000, r=R, p=25, rng=np.random.default_rng(7))
    g = np.isfinite(lnc)
    a, c = lnr[g], lnc[g]
    m = (a >= LO) & (a <= HI)
    print("%s of the record, d = 5, n = 15000: D2 = %.3f"
          % (nm, np.polyfit(a[m], c[m], 1)[0]))
print("wrote twowell_d2.pdf")
