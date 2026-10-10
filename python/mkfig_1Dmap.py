"""1Dmap.pdf -- the piecewise linear map F(x) of eq. (oneDmap), drawn exactly.

F(x) = 2 mu x for x < 1/2 and 2 mu (x - 1) + 1 for x > 1/2, with MU below. The
shaded strip [1/(2 mu), 1 - 1/(2 mu)] of width Delta = (mu - 1)/mu is mapped out
of [0, 1] in one iteration (F > 1 on the left branch, F < 0 on the right one).
"""
from _figstyle import *

MU = 1.25                       # 1/(2 mu) = 0.4, as in the original drawing
a = 1/(2*MU)

fig, ax = plt.subplots(figsize=(2.8, 2.9))
ax.axvspan(a, 1 - a, color="0.85", zorder=0)
x = np.linspace(0, 0.5, 2)
ax.plot(x, 2*MU*x, color=COL[0], lw=1.4)
x = np.linspace(0.5, 1, 2)
ax.plot(x, 2*MU*(x - 1) + 1, color=COL[0], lw=1.4)
ax.plot([0, 1], [1, 1], "k:", lw=0.7)
ax.plot([1, 1], [0, 1], "k:", lw=0.7)
ax.plot([a, a], [0, 1], "k:", lw=0.7)
ax.plot([1 - a, 1 - a], [0, 1], "k:", lw=0.7)
ax.plot([0, 1], [0, 1], color="0.5", lw=0.6, ls="--")
ax.annotate("", xy=(a, 0.45), xytext=(1 - a, 0.45),
            arrowprops=dict(arrowstyle="<->", lw=0.7))
ax.text(1 - a + 0.02, 0.50, r"$\Delta=\frac{\mu-1}{\mu}$", fontsize=8, va="bottom")
ax.set_xlim(-0.02, 1.2); ax.set_ylim(-0.25, 1.28)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_position(("data", 0)); ax.spines["bottom"].set_position(("data", 0))
ax.set_xticks([a, 1 - a, 1]); ax.set_xticklabels([r"$\frac{1}{2\mu}$", "", "$1$"])
ax.text(1 - a, -0.07, r"$1-\frac{1}{2\mu}$", ha="left", va="top", fontsize=9)
ax.set_yticks([1]); ax.set_yticklabels(["$1$"])
ax.text(-0.04, -0.06, "$0$", ha="right", va="top", fontsize=9)
ax.set_xlabel("$x_n$"); ax.set_ylabel("$F(x_n)$")
ax.xaxis.set_label_coords(1.0, 0.2); ax.yaxis.set_label_coords(0.06, 1.0)
ax.yaxis.label.set_rotation(0)
fig.tight_layout(pad=0.3)
fig.savefig(FIGDIR + "1Dmap.pdf")
print("wrote 1Dmap.pdf  mu =", MU, " Delta =", (MU - 1)/MU)
