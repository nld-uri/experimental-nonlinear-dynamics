"""stroboscopic.pdf (label stroboscopic): stroboscopic Poincare section of the book's
two-well Duffing oscillator x'' + 0.25 x' - x + x^3 = 0.3 cos t (T = 2 pi).
(a) the forcing f(t) = 0.3 cos t and (b) the response x(t) over six forcing periods after a
    transient of 200 time units (solve_ivp RK45, rtol 1e-9, from (x, x') = (-0.67, -0.37)),
    with the sampling instants t mod T = phi, phi = 0, marked; the samples of (b) lie in Sigma.
(c) the section itself: (x, x') at t mod T = 0 for 5000 consecutive periods after 50
    transient periods, fixed-step RK4 with 200 steps per period (_ch08data.duffing_strobe).
The col2 circles in (c) are the seven samples of (b) (t/T = 0, ..., 6)."""
import numpy as np, time
from _figstyle import *
from _ch08data import duffing, duffing_strobe

GAMMA, F, PHI = 0.25, 0.3, 0.0
T = 2*np.pi; DT = 0.02; NPER_TS = 6; TTRANS = 200.0
t0 = time.time()
# time series: integrate from t = 0, keep [TTRANS, TTRANS + 6T]; start sampling on a strobe instant
tstart = T*np.ceil(TTRANS/T)
N = int(round(NPER_TS*T/DT)) + 1
Y = duffing(N, DT, gamma=GAMMA, F=F, ttrans=tstart)
t = tstart + DT*np.arange(N)
x = Y[0]
tstrobe = tstart + PHI + T*np.arange(NPER_TS + 1)
istrobe = np.round((tstrobe - tstart)/DT).astype(int)
istrobe = istrobe[istrobe < N]
S = duffing_strobe(5000, 200, GAMMA, F, PHI, 50)
print("time series %d samples, dt %g; section %d points, x in [%.3f, %.3f], x' in [%.3f, %.3f]; run %.1f s"
      % (N, DT, S.shape[1], S[0].min(), S[0].max(), S[1].min(), S[1].max(), time.time()-t0))

fig = plt.figure(figsize=(6, 4.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.45, wspace=0.3)
axf = fig.add_subplot(gs[0, 0]); axx = fig.add_subplot(gs[0, 1]); axs = fig.add_subplot(gs[1, :])
tt = (t - tstart)/T
axf.plot(tt, F*np.cos(t), color=COL[0], lw=0.9)
axf.set_ylabel("$f(t) = 0.3\\cos t$"); axf.set_ylim(-0.4, 0.4)
axx.plot(tt, x, color=COL[0], lw=0.9)
axx.plot(tt[istrobe], x[istrobe], "o", color=COL[1], ms=4, mfc="w", mew=1.2, label=r"samples in $\Sigma$")
axx.set_ylabel("$x(t)$"); axx.set_ylim(-1.7, 1.7); axx.legend(loc="upper right", fontsize=7)
for ax, p in ((axf, "a"), (axx, "b")):
    for k in range(NPER_TS + 1): ax.axvline(k + PHI/T, color="k", lw=0.5, ls="--")
    ax.set_xlim(0, NPER_TS); ax.set_xlabel("$t/T$ (forcing periods)"); panel(ax, p, x=-0.2)
axf.text(1.5, 0.37, r"$t\ \mathrm{mod}\ T = \varphi$", fontsize=7, ha="center", va="top", bbox=dict(fc="w", ec="none", pad=0.5))
axs.plot(S[0], S[1], ".", color=COL[0], ms=1.2, rasterized=True)
axs.plot(Y[0, istrobe], Y[1, istrobe], "o", color=COL[1], ms=4, mfc="w", mew=1.2)
axs.set_xlabel("$x$"); axs.set_ylabel(r"$\dot x$"); axs.set_xlim(-1.7, 1.7); axs.set_ylim(-0.8, 1.0)
panel(axs, "c", x=-0.08)
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.09, top=0.95)
fig.savefig(FIGDIR + "stroboscopic.pdf", dpi=300)
