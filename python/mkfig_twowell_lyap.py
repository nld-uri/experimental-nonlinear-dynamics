"""twowell_lyap.pdf (label `twowelllyap`) -- mean logarithmic divergence and its
local slope for the two-well oscillator record of Section 1.4.1, embedded with
tau = 5 samples in d = 4, 5, 6, 7.

Settings: lyapunov(x, d, 5, N=5000, r=1, n=160, w=32), the record normalized to
zero mean and unit variance inside the routine. Step numbers are converted to
forcing periods by the 16 samples per period measured from the record, so
nothing here depends on the sampling rate in hertz. The fit window is steps
10 to 64, read off the local-slope plateau of panel (b).

Printed: the fitted exponent at every d, the slope over the saturated tail, the
sensitivity to the fit window, the two halves of the record separately, and the
scatter over independent draws of the fiducial points. Runtime a few minutes.
"""
from _figstyle import *
from _twowell import twowell, NPP
from lyapunov import lyapunov

DS = [4, 5, 6, 7]
LO, HI = 10, 64                  # fit window in steps
NSTEP, NFID, WIN = 160, 5000, 32
SEED = 3

x = twowell()
lam, curves = {}, {}
for d in DS:
    ns, lnd = lyapunov(x, d, 5, N=NFID, r=1, n=NSTEP, w=WIN,
                       rng=np.random.default_rng(SEED))
    curves[d] = lnd
    m = (ns >= LO) & (ns <= HI)
    p = np.polyfit(ns[m], lnd[m], 1)
    lam[d] = (p[0]*NPP, p[1])

fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.6))
for k, d in enumerate(DS):
    t = ns/NPP
    axs[0].plot(t, curves[d], color=COL[k], lw=0.9, label=r"$d=%d$" % d)
    sl = np.gradient(curves[d], ns)*NPP
    axs[1].plot(t, sl, color=COL[k], lw=0.9)
p = lam[5]
tt = np.array([0.0, float(NSTEP)])
axs[0].plot(tt/NPP, (p[0]/NPP)*tt + p[1], "k--", lw=0.9)
axs[0].set_ylim(-4, 1.5)
for ax in axs:
    ax.axvspan(LO/NPP, HI/NPP, color="0.5", alpha=0.18, lw=0)
    ax.set_xlabel("$n$, forcing periods")
    ax.set_xlim(0, NSTEP/NPP)
axs[0].set_ylabel(r"$\Delta_n$")
axs[0].legend(loc="lower right", ncol=2)
axs[1].axhline(0.61, color="k", ls="--", lw=0.7)
axs[1].text(6.2, 0.66, r"$\hat\lambda_1 = 0.61$", fontsize=8, ha="right")
axs[1].set_ylabel(r"$\mathrm{d}\Delta_n/\mathrm{d}n$, nats per period")
axs[1].set_ylim(0, 1.2)
panel(axs[0], "a", x=-0.18)
panel(axs[1], "b", x=-0.18)
fig.tight_layout(pad=0.4, w_pad=1.4)
fig.savefig(FIGDIR + "twowell_lyap.pdf")

print("fit window steps %d-%d = %.2f-%.2f forcing periods" % (LO, HI, LO/NPP, HI/NPP))
for d in DS:
    m2 = (ns >= 100) & (ns <= NSTEP)
    tail = np.polyfit(ns[m2], curves[d][m2], 1)[0]*NPP
    print("d=%d  lambda_1 = %.3f nats/period, tail slope (n=100-160) %.3f"
          % (d, lam[d][0], tail))
v = np.array([lam[d][0] for d in DS])
print("d = 4..7: mean %.3f, half-spread %.3f" % (v.mean(), (v.max() - v.min())/2))

lnd = curves[5]
print("window sensitivity at d = 5 (nats per forcing period):")
for lo, hi in ((6, 64), (10, 48), (10, 64), (10, 80), (16, 64), (20, 72),
               (1, 160), (100, 160)):
    m = (ns >= lo) & (ns <= hi)
    print("   steps %3d-%3d  %.3f" % (lo, hi, np.polyfit(ns[m], lnd[m], 1)[0]*NPP))
for nm, seg in (("first half", x[:48000]), ("second half", x[48000:])):
    a, b = lyapunov(seg, 5, 5, N=NFID, r=1, n=NSTEP, w=WIN,
                    rng=np.random.default_rng(SEED))
    m = (a >= LO) & (a <= HI)
    print("   %s of the record: %.3f" % (nm, np.polyfit(a[m], b[m], 1)[0]*NPP))
for w in (16, 32, 64, 128):
    a, b = lyapunov(x, 5, 5, N=NFID, r=1, n=NSTEP, w=w,
                    rng=np.random.default_rng(SEED))
    m = (a >= LO) & (a <= HI)
    print("   Theiler window w = %3d samples: %.3f"
          % (w, np.polyfit(a[m], b[m], 1)[0]*NPP))
v = []
for s in range(6):
    a, b = lyapunov(x, 5, 5, N=NFID, r=1, n=NSTEP, w=WIN,
                    rng=np.random.default_rng(s))
    m = (a >= LO) & (a <= HI)
    v.append(np.polyfit(a[m], b[m], 1)[0]*NPP)
v = np.array(v)
print("   six draws of the fiducial points: mean %.3f, sd %.4f" % (v.mean(), v.std()))
print("wrote twowell_lyap.pdf")
