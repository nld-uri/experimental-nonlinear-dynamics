"""visdel.pdf (label visdel): choosing the delay by eye on the Lorenz x record.
Lorenz system (10, 28, 8/3) from (5, -5, 20), dt = 0.02, first 20 time units discarded,
N = 10,000 samples. Two-dimensional forward delay plots (x_n, x_{n+tau}) for
(a) tau = 1 sample (0.02): too small, the points bunch along the diagonal;
(b) tau = 9 samples (0.18, the first AMI minimum): about right;
(c) tau = 60 samples (1.2, more than a lobe circuit): too large, the structure is folded over."""
import numpy as np
from _figstyle import *
from _ch08data import lorenz

N, DT = 10000, 0.02
TAUS = (1, 9, 60)
x = lorenz(N, DT, ttrans=20.0)[0]
fig, axs = plt.subplots(1, 3, figsize=(6, 2.25))
for ax, tau, p in zip(axs, TAUS, "abc"):
    ax.plot(x[:N-tau], x[tau:], color=COL[0], lw=0.3)
    ax.plot([-22, 22], [-22, 22], color="0.6", lw=0.4, ls=":")
    ax.set_xlim(-22, 22); ax.set_ylim(-22, 22); ax.set_aspect("equal")
    ax.set_xticks([-20, 0, 20]); ax.set_yticks([-20, 0, 20])
    ax.set_xlabel("$x_n$"); ax.set_ylabel(r"$x_{n+\tau}$")
    ax.text(0.04, 0.95, r"$\tau = %d$" % tau, transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(fc="w", ec="none", pad=1))
    panel(ax, p, x=-0.28)
    print("tau = %d: correlation of (x_n, x_{n+tau}) = %.3f" % (tau, np.corrcoef(x[:N-tau], x[tau:])[0, 1]))
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.9, wspace=0.5)
fig.savefig(FIGDIR + "visdel.pdf")
