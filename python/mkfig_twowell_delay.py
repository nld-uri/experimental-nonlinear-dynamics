"""twowell_delay.pdf (label twowelldelay): delay selection on the two-well
oscillator record of Section sec:twowelldata.

(a) average mutual information I(tau) in bits, plug-in estimator of ami.py at
    n_b = 16, 32 and 64 bins, lags 0 to 40 samples, whole record (N = 96,000).
    The first minimum is at tau = 5 samples for all three bin counts.
(b) autocorrelation coefficient rho(tau) over the same lags: first zero
    crossing at 12 samples, first crossing of 1/e at 8 samples.
(c), (d), (e) two-dimensional delay reconstructions (s_n, s_{n+tau}) of the
    first 16,000 samples (1000 forcing periods) at tau = 2, 5 and 12 samples.

Vertical markers in (a) and (b) sit at one and two forcing periods (16 and 32
samples), so that the ripple locked to the drive is visible.
"""
import numpy as np
from _figstyle import *
from _twowell import twowell, NPP
from ami import ami

KMAX = 40
TAUS = (2, 5, 12)
NREC = 16000                      # samples used in the reconstruction panels

x = twowell()
N = x.size
Iami = {nb: ami(x, KMAX, nb=nb) for nb in (16, 32, 64)}
xd = x - x.mean()
rho = np.correlate(xd, xd, "full")[N-1:N+KMAX]/np.dot(xd, xd)

for nb, I in Iami.items():
    mins = [k for k in range(1, KMAX) if I[k] < I[k-1] and I[k] < I[k+1]]
    print("n_b = %3d: H = %.3f bit, first AMI minimum at tau = %d (%.3f bit), "
          "all minima %s, N/n_b^2 = %.0f"
          % (nb, I[0], mins[0], I[mins[0]], mins, N/nb**2))
print("rho: first zero crossing at %d, first below 1/e at %d, rho(5) = %.3f, "
      "rho(12) = %.4f" % (int(np.argmax(rho < 0)), int(np.argmax(rho < np.exp(-1))),
                          rho[5], rho[12]))

fig = plt.figure(figsize=(6.0, 4.5))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.55, wspace=1.6)
axI = fig.add_subplot(gs[0, :3]); axR = fig.add_subplot(gs[0, 3:])
lag = np.arange(KMAX + 1)
for nb, c, ls in zip((16, 32, 64), (COL[2], COL[0], COL[4]), ("--", "-", ":")):
    axI.plot(lag, Iami[nb], ls, color=c, lw=1.0, label="$n_b = %d$" % nb)
axI.plot([5], [Iami[32][5]], "o", color=COL[1], ms=5, mfc="w", mew=1.2)
axI.annotate(r"$\tau = 5$", (5, Iami[32][5]), textcoords="offset points",
             xytext=(-4, -14), fontsize=8, color=COL[1])
axI.set_ylabel(r"$I(\tau)$ (bits)"); axI.set_ylim(0, 2.6)
axI.legend(loc="upper right")
axR.plot(lag, rho, "-", color=COL[0], lw=1.0)
axR.axhline(0.0, color="k", lw=0.5)
axR.axhline(np.exp(-1), color="k", lw=0.5, ls=":")
axR.text(30, np.exp(-1) + 0.03, "$1/e$", fontsize=8)
for k, c in ((8, COL[3]), (12, COL[1])):
    axR.plot([k], [rho[k]], "o", color=c, ms=5, mfc="w", mew=1.2)
axR.annotate("$8$", (8, rho[8]), textcoords="offset points", xytext=(-14, -12),
             fontsize=8, color=COL[3])
axR.annotate("$12$", (12, rho[12]), textcoords="offset points", xytext=(3, 6),
             fontsize=8, color=COL[1])
axR.set_ylabel(r"$\rho(\tau)$"); axR.set_ylim(-0.35, 1.05)
for a, p in ((axI, "a"), (axR, "b")):
    for k in (NPP, 2*NPP):
        a.axvline(k, color="0.6", lw=0.5, ls="-.")
    a.set_xlim(0, KMAX); a.set_xlabel(r"$\tau$ (samples)"); panel(a, p, x=-0.2)

s = x[:NREC]
for j, tau in enumerate(TAUS):
    a = fig.add_subplot(gs[1, 2*j:2*j+2])
    a.plot(s[:-tau], s[tau:], ".", color=COL[0], ms=0.5, alpha=0.25, rasterized=True)
    a.set_xlabel("$s_n$ (V)"); a.set_ylabel(r"$s_{n+\tau}$ (V)")
    a.set_xlim(-3.6, 3.6); a.set_ylim(-3.6, 3.6)
    a.set_xticks([-3, 0, 3]); a.set_yticks([-3, 0, 3])
    a.set_aspect("equal")
    a.text(0.5, 1.02, r"$\tau = %d$" % tau, transform=a.transAxes, ha="center",
           va="bottom", fontsize=8)
    panel(a, "cde"[j], x=-0.3)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.94)
fig.savefig(FIGDIR + "twowell_delay.pdf", dpi=400)
