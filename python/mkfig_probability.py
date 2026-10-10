"""probability.pdf (label one): a generic signal x(t) beside its probability density p(x).
The signal is the two-well Duffing x record of _ch02data (dt = 0.1); p(x) is a
histogram of N = 40000 samples (80 bins); the plotted segment is the first 100 time
units. The band a <= x <= b (a = 0.8, b = 1.1) is shaded in p(x) and the samples of
x(t) that fall in it are marked, so that Prob{a <= x <= b} = int_a^b p dx is the shaded
area and the fraction of time the signal spends between the two lines."""
import numpy as np
from _figstyle import *
from _ch02data import duffing_record, density

A, B, TSHOW = 0.8, 1.1, 100.0
t, x = duffing_record()
c, h, e = density(x)
frac = np.mean((x >= A) & (x <= B))
area = h[(c > A) & (c < B)].sum()*(e[1]-e[0])
print("fraction of samples in [a,b] = %.4f, histogram area = %.4f" % (frac, area))

fig = plt.figure(figsize=(6, 1.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 4.2], wspace=0.05)
axp = fig.add_subplot(gs[0]); axx = fig.add_subplot(gs[1], sharey=axp)
# density, drawn sideways with x vertical and p(x) increasing to the left
axp.fill_betweenx(c, 0, h, color=COL[0], alpha=0.35, lw=0)
axp.plot(h, c, color=COL[0], lw=0.9)
m = (c >= A) & (c <= B)
axp.fill_betweenx(c[m], 0, h[m], color=COL[0], alpha=0.9, lw=0)
axp.invert_xaxis()
for s in ("top", "right", "left"):
    axp.spines[s].set_visible(False)
axp.set_yticks([]); axp.set_xticks([0])
axp.set_xticklabels([""])
axp.set_xlabel("$p(x)$"); axp.xaxis.set_label_coords(0.5, -0.02)
axp.set_ylabel("$x$", rotation=0); axp.yaxis.set_label_coords(1.0, 1.02)
# signal
sel = t <= TSHOW
sketch_axes(axx, "$t$", "$x(t)$")
axx.plot(t[sel], x[sel], color=COL[0], lw=0.8)
inb = sel & (x >= A) & (x <= B)
axx.plot(t[inb], x[inb], ".", color=COL[1], ms=2.5)
for lev, lab in ((A, "$a$"), (B, "$b$")):
    axx.axhline(lev, color="k", lw=0.5)
    axx.text(-0.8, lev, lab, ha="right", va="center", fontsize=8)
axx.set_xlim(-1, TSHOW+1); axx.set_ylim(-1.7, 1.7)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.14, top=0.9)
fig.savefig(FIGDIR+"probability.pdf")
