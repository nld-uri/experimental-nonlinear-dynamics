"""delayrec.pdf (label reconst): delay reconstruction of the Lorenz x record.
Lorenz system (10, 28, 8/3) from (5, -5, 20), dt = 0.02, first 20 time units discarded,
N = 4000 samples kept (80 time units).
(a) the record x_n and its two forward shifts x_{n+tau}, x_{n+2 tau} with tau = 9 samples
    (0.18 time units, the first minimum of the AMI of this record, Chapter delay), drawn one
    under the other over the first 30 time units; the three col2 dots are the coordinates of one delay vector
    x_n = [x_n, x_{n+tau}, x_{n+2tau}]^T (n = 300), the dotted lines join them.
(b) the three-dimensional reconstruction (x_n, x_{n+tau}, x_{n+2tau}); the col2 dot is x_300.
(c) the true state (x_n, y_n, z_n) over the same 80 time units, for comparison."""
import numpy as np
from _figstyle import *
from _ch08data import lorenz

TAU, N, DT = 9, 4000, 0.02
U = lorenz(N, DT, ttrans=20.0)
x = U[0]; t = DT*np.arange(N)
n0 = 300
M = N - 2*TAU
X = np.array([x[:M], x[TAU:TAU+M], x[2*TAU:2*TAU+M]])
print("N = %d, tau = %d, %d reconstructed points; x_%d = (%.2f, %.2f, %.2f)" % (N, TAU, M, n0, *X[:, n0]))

fig = plt.figure(figsize=(6, 5.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.3, wspace=0.05)
ax = fig.add_subplot(gs[0, :])
off = 45.0
for j in range(3):
    xs = x[j*TAU:j*TAU+M]
    ax.plot(t[:M], xs - j*off, color=COL[0], lw=0.6)
    ax.plot(t[n0], X[j, n0] - j*off, "o", color=COL[1], ms=4, zorder=3)
    ax.text(-1.0, -j*off, ["$x_n$", r"$x_{n+\tau}$", r"$x_{n+2\tau}$"][j], ha="right", va="center", fontsize=9)
ax.plot([t[n0]]*2, [X[2, n0] - 2*off, X[0, n0]], ":", color=COL[1], lw=0.8)
ax.annotate(r"$\bm{x}_n = [x_n,\, x_{n+\tau},\, x_{n+2\tau}]^{\mathrm{T}}$".replace(r"\bm{x}", r"\mathbf{x}"),
            xy=(t[n0], X[2, n0] - 2*off), xytext=(t[n0] + 3, -2*off - 27), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.6))
ax.set_xlim(-0.5, 30); ax.set_ylim(-2*off - 32, 25)
ax.set_yticks([]); ax.set_xlabel("time $t$")
for s in ("left", "top", "right"): ax.spines[s].set_visible(False)
panel(ax, "a", x=-0.02)

def portrait(axp, A, B, C, labels, p, view=(20, -50)):
    axp.plot(A, B, C, color=COL[0], lw=0.35)
    axp.set_xlabel(labels[0], labelpad=-4); axp.set_ylabel(labels[1], labelpad=-4); axp.set_zlabel(labels[2], labelpad=-4)
    axp.tick_params(labelsize=6, pad=-2); axp.view_init(elev=view[0], azim=view[1])
    axp.text2D(0.02, 0.92, "(%s)" % p, transform=axp.transAxes, fontsize=9)
axb = fig.add_subplot(gs[1, 0], projection="3d")
portrait(axb, X[0], X[1], X[2], ["$x_n$", r"$x_{n+\tau}$", r"$x_{n+2\tau}$"], "b", view=(35, -30))
axb.plot([X[0, n0]], [X[1, n0]], [X[2, n0]], "o", color=COL[1], ms=4)
axc = fig.add_subplot(gs[1, 1], projection="3d")
portrait(axc, U[0], U[1], U[2], ["$x$", "$y$", "$z$"], "c")
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.04, top=0.97)
fig.savefig(FIGDIR + "delayrec.pdf")
