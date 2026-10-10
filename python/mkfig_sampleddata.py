"""sampleddata.pdf (label sampleddata): a record of N = 32 samples x_n = x(n Delta) of the
band-limited signal of _ch05data, Delta = 0.25 s (f_s = 4 f_BW), T = N Delta = 8 s; the
signal is shifted by 30 s into the record so that the segment differs from sampled.pdf."""
import numpy as np
from _figstyle import *
from _ch05data import signal

N, D, T0 = 32, 0.25, 30.0
t = np.arange(0, N*D, 1e-3); x = 0.98*signal(t+T0)
tn = D*np.arange(N); xn = 0.98*signal(tn+T0)
fig, ax = plt.subplots(figsize=(6, 2.0))
sketch_axes(ax, "$t,\ n$", "$x(t)$")
for tt in tn: ax.axvline(tt, color="0.6", lw=0.4, ls="--")
ax.plot(t, x, color=COL[0], lw=0.8); ax.plot(tn, xn, "o", color=COL[1], ms=3.5, mfc="none", mew=0.8)
for k, dy in zip(range(4), (-0.3, -0.3, -0.3, 0.2)): ax.text(tn[k]+0.02, xn[k]+dy, "$x_%d$" % k, fontsize=7, va="center")
ax.text(tn[-1]+0.4, xn[-1]+0.4, "$x_{N-1}$", fontsize=7, ha="right", va="top")
ax.annotate("", (tn[10]+0.05, -2.4), (tn[10]-0.2, -2.4), arrowprops=dict(arrowstyle="->", lw=0.6)); 
ax.annotate("", (tn[11]+0.2, -2.4), (tn[11]-0.05, -2.4), arrowprops=dict(arrowstyle="<-", lw=0.6)); 
ax.text(tn[10]+D/2, -2.6, r"$\Delta$", ha="center", va="bottom", fontsize=8)
ax.text(-0.1, 0, "0", fontsize=8, ha="right", va="center")
ax.set_xlim(-0.2, N*D+0.1); ax.set_ylim(-3.0, 3.0)
fig.subplots_adjust(left=0.03, right=0.96, bottom=0.03, top=0.9); fig.savefig(FIGDIR+"sampleddata.pdf")
