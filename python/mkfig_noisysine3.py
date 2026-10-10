"""noisysine3.pdf (label nsine3): x(t) = a sin t + n(t), a = 1, sigma_n = 0.1 a, dt = 0.01,
seed 0, over 15 periods, t in [0, 30 pi]; a single sample function in which two averaging
scales are marked: Delta T_1 = pi/2 (a quarter period, over which the mean is not constant)
and Delta T_2 = 14 periods = 28 pi (over which the mean is nearly zero)."""
import numpy as np
from _figstyle import *

A, SIG, DT = 1.0, 0.1, 0.01
DT1, T2A, T2B = np.pi/2, np.pi/2, np.pi/2 + 28*np.pi
rng = np.random.default_rng(0)
t = np.arange(0, 30*np.pi+DT/2, DT)
x = A*np.sin(t) + SIG*rng.standard_normal(t.size)
m1 = x[(t >= T2A) & (t < T2A+DT1)].mean(); m2 = x[(t >= T2A) & (t < T2B)].mean()
print("mean over Delta T_1 = %.3f, over Delta T_2 = %.4f" % (m1, m2))
fig, ax = plt.subplots(figsize=(6, 2.4))
sketch_axes(ax, "$t$", "$x(t)$")
ax.plot(t, x, color=COL[0], lw=0.4)
ax.set_yticks([-A, A]); ax.set_yticklabels(["$-a$", "$a$"]); ax.tick_params(axis="y", length=3)
ax.set_ylim(-2.0*A, 1.5*A); ax.set_xlim(-0.5, 30*np.pi+0.5)
for tt in (T2A, T2B):
    ax.axvline(tt, color="k", lw=0.5)
ax.vlines(x=T2A+DT1, ymin=-1.65*A, ymax=1.65*A, color="k", lw=0.5)
ax.annotate("", (T2A-1.8, -1.45*A), (T2A+0.5, -1.45*A), arrowprops=dict(arrowstyle="<-", lw=0.6))
ax.annotate("", (T2A+DT1-0.5, -1.45*A), (T2A+DT1+1.8, -1.45*A), arrowprops=dict(arrowstyle="->", lw=0.6))
ax.text(T2A+DT1+2, -1.45*A, r"$\Delta T_1$", va="center", fontsize=8)
ax.annotate("", (T2A-0.5, -1.8*A), (T2B+0.5, -1.8*A), arrowprops=dict(arrowstyle="<->", lw=0.6))
ax.text(0.5*(T2A+T2B), -1.8*A, r"$\Delta T_2$", va="center", ha="center", fontsize=8,
        bbox=dict(fc="w", ec="none", pad=1))
fig.subplots_adjust(left=0.05, right=0.97, bottom=0.03, top=0.9)
fig.savefig(FIGDIR+"noisysine3.pdf")
