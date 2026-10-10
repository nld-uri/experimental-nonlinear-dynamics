"""delaycorr.pdf (label delaycorr): the quarter-period delay for a harmonic signal.
(a) x(t) = sin t (col1) and its autocorrelation coefficient rho_xx(tau) (col2), estimated
    with the 1/(N-m) normalization from a record of 100 periods sampled at dt = 2 pi/200
    (N = 20,000 samples); the estimate equals cos(tau) to 1e-3, and its first zero crossing
    is at tau = T/4 = pi/2 (dotted line), the delay that shifts the phase by pi/2.
(b) the two-dimensional forward delay reconstruction (x_n, x_{n+tau}) with tau = T/4: a circle."""
import numpy as np
from _figstyle import *
from _ch03data import acf

NPER, NS = 100, 200                    # periods, samples per period
dt = 2*np.pi/NS; N = NPER*NS
t = dt*np.arange(N); x = np.sin(t)
mmax = NS
r = acf(x, mmax); lag = dt*np.arange(mmax + 1)
print("max |rho_hat - cos| = %.1e; first zero crossing at lag %d samples = %.4f (pi/2 = %.4f)"
      % (np.abs(r - np.cos(lag)).max(), np.argmax(r < 0), dt*np.argmax(r < 0), np.pi/2))

fig, axs = plt.subplots(1, 2, figsize=(6, 2.3), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]
ax.plot(t[:NS+1], x[:NS+1], color=COL[0], lw=1.2, label="$\\sin t$")
ax.plot(lag, r, color=COL[1], lw=1.2, label=r"$\rho_{xx}(\tau)$")
ax.axhline(0, color="k", lw=0.5); ax.axvline(np.pi/2, color="k", lw=0.6, ls=":")
ax.set_xlim(0, 2*np.pi); ax.set_ylim(-1.05, 1.05)
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]); ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax.set_xlabel(r"$t$, $\tau$"); ax.set_ylabel(r"$\sin t$, $\rho_{xx}(\tau)$"); ax.legend(loc="upper right")
ax.text(np.pi/2 + 0.1, -0.9, r"$\tau = T/4$", fontsize=8)
panel(ax, "a", x=-0.15)
ax = axs[1]
k = NS//4
ax.plot(x[:N-k], x[k:], color=COL[0], lw=1.2)
ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
ax.set_xlabel(r"$x_n = \sin t_n$"); ax.set_ylabel(r"$x_{n+\tau} = \sin(t_n + \pi/2)$")
panel(ax, "b", x=-0.3)
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.9, wspace=0.45)
fig.savefig(FIGDIR + "delaycorr.pdf")
