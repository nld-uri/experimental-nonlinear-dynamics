"""comb.pdf (label comb): the sampling (comb) function s(t; T_s) = sum_n delta(t - n T_s),
drawn as unit arrows at n T_s, n = -4..4, T_s = 1. Pure formula."""
import numpy as np
from _figstyle import *

TS = 1.0
fig, ax = plt.subplots(figsize=(4, 1.3))
for n in range(-4, 5):
    ax.annotate("", (n*TS, 1), (n*TS, 0), arrowprops=dict(arrowstyle="->", lw=1.0, color=COL[0]))
sketch_axes(ax, "$t$", "$s(t)$"); ax.spines["left"].set_visible(False)
ax.plot([0, 0], [0, 1.25], "k", lw=0.6)
ax.text(-5.0, 0.5, r"$\cdots$", fontsize=10, va="center"); ax.text(4.7, 0.5, r"$\cdots$", fontsize=10, va="center")
ax.set_xlim(-5.3, 5.4); ax.set_ylim(-0.45, 1.35)
ax.annotate("", (2*TS, -0.25), (3*TS, -0.25), arrowprops=dict(arrowstyle="<->", lw=0.6)); ax.text(2.5*TS, -0.3, "$T_s$", ha="center", va="top", fontsize=8)
ax.text(0.1, -0.12, "0", fontsize=8, va="top"); ax.yaxis.set_label_coords(0.5, 0.98)
fig.subplots_adjust(left=0.02, right=0.96, bottom=0.05, top=0.95); fig.savefig(FIGDIR+"comb.pdf")
