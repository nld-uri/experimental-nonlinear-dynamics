"""sampled.pdf (label sampled): the band-limited signal of _ch05data (f_BW = 1 Hz) on
[0, 12] s (drawn at 1 ms) and its samples x(n T_s) at T_s = 0.25 s (f_s = 4 f_BW)."""
import numpy as np
from _figstyle import *
from _ch05data import signal, FBW

TS, T = 0.25, 12.0
t = np.arange(0, T, 1e-3); x = signal(t)
tn = np.arange(0, T, TS); xn = signal(tn)
print("f_s = %.2f f_BW, %d samples" % (1/TS/FBW, tn.size))
fig, ax = plt.subplots(figsize=(6, 1.9))
sketch_axes(ax, "$t$", "$x(t)$")
ax.plot(t, x, color=COL[0], lw=0.8); ax.plot(tn, xn, "o", color=COL[1], ms=3.5, mfc="none", mew=0.8)
k = 8; ax.axvline(tn[k], color="k", lw=0.5); ax.axvline(tn[k+1], color="k", lw=0.5)
ax.annotate("", (tn[k]-0.3, -2.9), (tn[k]+0.07, -2.9), arrowprops=dict(arrowstyle="<-", lw=0.6))
ax.annotate("", (tn[k+1]-0.07, -2.9), (tn[k+1]+0.3, -2.9), arrowprops=dict(arrowstyle="->", lw=0.6))
ax.text(tn[k+1]+0.3, -2.95, "$T_s$", va="center", fontsize=8)
ax.set_xlim(-0.2, T+0.2); ax.set_ylim(-3.3, 3.3)
fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.9); fig.savefig(FIGDIR+"sampled.pdf")
