"""datacollw.pdf (label datacollw): the band-limited signal of _ch05data on [0, 20] s with the
data collection window [4, 16] s marked (the record actually kept)."""
import numpy as np
from _figstyle import *
from _ch05data import signal

T, W0, W1 = 20.0, 4.0, 16.0
t = np.arange(0, T, 1e-3); x = signal(t)
fig, ax = plt.subplots(figsize=(3.5, 1.6))
sketch_axes(ax, "$t$", "$x(t)$")
ax.plot(t, x, color=COL[0], lw=0.7)
ax.axvline(W0, color="k", lw=0.6); ax.axvline(W1, color="k", lw=0.6)
ax.annotate("", (W0, -2.9), (W1, -2.9), arrowprops=dict(arrowstyle="<->", lw=0.6))
ax.text(0.5*(W0+W1), -2.7, "data collection window", ha="center", va="bottom", fontsize=7)
ax.set_xlim(-0.2, T+0.2); ax.set_ylim(-3.3, 3.3)
fig.subplots_adjust(left=0.04, right=0.95, bottom=0.03, top=0.9); fig.savefig(FIGDIR+"datacollw.pdf")
