"""Figure duffing_bifu.pdf (Chapter 17, Section 17.4): bifurcation diagram of
the two-well Duffing oscillator xdd + 0.25 xd - x + x^3 = f cos t against the
forcing amplitude f, sampled once per forcing period (Poincare section at
phase zero).  Runs in about half a minute."""
import time
import numpy as np
from _figstyle import COL, plt, FIGDIR

NF, NPER, NTRANS = 1200, 200, 400
DT = np.pi/36                      # 72 samples per forcing period
GAM = 0.25
t0 = time.time()
f = np.linspace(0.32, 0.40, NF)
ff = f[:, None]


def g(u, tt):
    x, v = u[:, 0:1], u[:, 1:2]
    return np.hstack([v, -GAM*v + x - x**3 + ff*np.cos(tt)])


rng = np.random.default_rng(0)
u = np.tile(np.array([1.0, 0.0]), (NF, 1)) + 0.01*rng.standard_normal((NF, 2))
tt = 0.0
pts = []
for period in range(NTRANS + NPER):
    for _ in range(72):
        k1 = g(u, tt); k2 = g(u + 0.5*DT*k1, tt + 0.5*DT)
        k3 = g(u + 0.5*DT*k2, tt + 0.5*DT); k4 = g(u + DT*k3, tt + DT)
        u = u + (DT/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        tt += DT
    if period >= NTRANS:
        pts.append(u[:, 0].copy())
X = np.array(pts)                                   # (NPER, NF)
print(f"{NF} forcing amplitudes, {NTRANS} periods discarded, {NPER} kept")
print(f"x on the section: [{X.min():.2f}, {X.max():.2f}]")

fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.plot(np.repeat(f[None, :], NPER, axis=0).ravel(), X.ravel(), ",",
        color=COL[0], alpha=0.4)
ax.set_xlabel("$f$"); ax.set_ylabel("$x$")
ax.set_xlim(0.32, 0.40)
fig.tight_layout()
fig.savefig(FIGDIR + "duffing_bifu.pdf", dpi=600)
print("wrote figures/duffing_bifu.pdf  (%.0f s)" % (time.time() - t0))
