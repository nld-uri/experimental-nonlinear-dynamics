"""probability_distribution.pdf (label prob_dist): the same two-well Duffing x record as
probability.pdf beside its probability distribution P(x) = int_{-inf}^x p(s) ds
(empirical distribution of N = 40000 samples). The level a = -0.5 is marked: P(a) is
the fraction of samples below a, shaded in the signal panel."""
import numpy as np
from _figstyle import *
from _ch02data import duffing_record

A, TSHOW = -0.5, 100.0
t, x = duffing_record()
xs = np.sort(x); P = np.arange(1, x.size+1)/x.size
Pa = np.mean(x <= A)
print("P(a) = %.4f at a = %.2f" % (Pa, A))

fig = plt.figure(figsize=(6, 1.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 4.2], wspace=0.05)
axp = fig.add_subplot(gs[0]); axx = fig.add_subplot(gs[1], sharey=axp)
axp.plot(P, xs, color=COL[0], lw=1.0)
axp.plot([0, Pa], [A, A], color="k", lw=0.5, ls=":")
axp.plot([Pa, Pa], [-1.7, A], color="k", lw=0.5, ls=":")
axp.invert_xaxis()
for s in ("top", "right", "left"):
    axp.spines[s].set_visible(False)
axp.set_yticks([]); axp.set_xticks([0, Pa, 1])
axp.set_xticklabels(["0", "$P(a)$", "1"], fontsize=8)
axp.text(0.02, 1.02, "$P(x)$", transform=axp.transAxes, ha="left", va="bottom")
axp.set_ylabel("$x$", rotation=0); axp.yaxis.set_label_coords(1.0, 1.02)
sel = t <= TSHOW
sketch_axes(axx, "$t$", "$x(t)$")
axx.plot(t[sel], x[sel], color=COL[0], lw=0.8)
axx.fill_between(t[sel], A, x[sel], where=x[sel] <= A, color=COL[0], alpha=0.35, lw=0)
axx.axhline(A, color="k", lw=0.5, ls=":")
axx.text(-0.8, A, "$a$", ha="right", va="center", fontsize=8)
axx.set_xlim(-1, TSHOW+1); axx.set_ylim(-1.7, 1.7)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.14, top=0.9)
fig.savefig(FIGDIR+"probability_distribution.pdf")
