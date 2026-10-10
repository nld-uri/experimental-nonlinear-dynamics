"""ergo.pdf (label ergo): five sample functions of one stationary, ergodic random process,
the AR(2) process of Chapter 3 with rho = 0.2 (x_{n+1} = (2 - (pi/10)^2 - 0.2) x_n
- 0.8 x_{n-1} + eta_n, unit white eta), 400 samples each after a 200-sample transient,
seeds 0..4. The dotted vertical line marks a fixed time t_j at which the ensemble average
over the sample functions is taken; the bracket marks the interval over which the time
average of the single bottom record is taken."""
import numpy as np
from _figstyle import *
from _ch03data import ar2

K, N, NB, RHO, TJ = 5, 400, 200, 0.2, 120
X = np.array([ar2(RHO, N+NB, np.random.default_rng(k))[NB:] for k in range(K)])
X /= X.std()
n = np.arange(N)
print("ensemble mean at t_j over %d records: %.3f; time mean of record 5 over %d samples: %.3f (process mean 0)" % (K, X[:, TJ].mean(), N, X[-1].mean()))
fig, axs = plt.subplots(K, 1, figsize=(6, 3.0), sharex=True)
for k, ax in enumerate(axs):
    ax.plot(n, X[k], color=COL[k], lw=0.6)
    sketch_axes(ax, "$t$", "$x_{%d}(t)$" % (k+1))
    ax.yaxis.set_label_coords(-0.005, 0.55); ax.yaxis.label.set_ha("right")
    ax.xaxis.set_label_coords(1.01, 0.5)
    ax.set_ylim(-3.5, 3.5); ax.set_xlim(-2, N+2)
    ax.axvline(TJ, color=COL[1], lw=0.7, ls=":")
    ax.plot(TJ, X[k, TJ], "o", color=COL[1], ms=3)
axs[-1].annotate("", (200, -3.2), (390, -3.2), arrowprops=dict(arrowstyle="<->", lw=0.6, color=COL[0]))
axs[-1].text(295, -4.0, "time average", ha="center", va="top", fontsize=8, color=COL[0])
axs[-1].annotate("ensemble average", xy=(TJ, -3.3), xytext=(TJ, -6.5), ha="center", fontsize=8, color=COL[1],
                 arrowprops=dict(arrowstyle="->", lw=0.6, color=COL[1]), annotation_clip=False)
axs[0].text(TJ+4, 3.2, "$t = t_j$", fontsize=8, color=COL[1], va="top")
fig.subplots_adjust(left=0.08, right=0.96, bottom=0.16, top=0.97, hspace=0.15)
fig.savefig(FIGDIR+"ergo.pdf")
