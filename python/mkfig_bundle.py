"""bundlefig.pdf -- what the strand bundle buys, against r.

Three panels, one y-axis each (never a dual axis): the two noise levels
are one series each, direct-labelled, in the book's col1/col2 which pass
the CVD and contrast checks as a pair. The clean-record value of each
estimator is the grey reference, because that -- not the literature
value -- is the ceiling a filter can reach at this record length.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL1, COL2, GREY = "#0072BD", "#D95319", "#59595c"
R = np.array([1, 2, 4, 8])
# 20,000 points, l=11, m=12, d=6, q=3, tent weight, three iterations
# weight a^4/(delta^4 + a^4), a = distance to the farthest strand of the
# bundle; numbers from bundle_exp.py
SNR = {20: [19.90, 24.94, 27.67, 27.58], 40: [13.72, 18.72, 21.36, 21.23]}
D2 = {20: [2.836, 2.474, 2.030, 1.926], 40: [2.950, 2.834, 2.389, 2.017]}
L1 = {20: [0.104, 0.322, 0.740, 0.800], 40: [0.086, 0.106, 0.268, 0.388]}
IN = {20: (14.05, 2.962, -0.006), 40: (7.97, 2.974, 0.038)}
CLEAN_D2, CLEAN_L1 = 1.974, 0.828

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "mathtext.fontset": "cm", "font.family": "serif",
    "font.serif": ["DejaVu Serif"]})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))

def panel(a, data, ylab, lab, ref=None, refname=None, dyl=(6, 6),
          at=-1, ha="right", dx=-2, lpos=(0.05, 0.94),
          refat=8, refha="right", refdx=(-2, -11)):
    for (lvl, col, dy) in ((20, COL1, dyl[0]), (40, COL2, dyl[1])):
        a.plot(R, data[lvl], "o-", color=col, ms=4, mfc="white", mew=1.2,
               zorder=3)
        a.annotate(rf"${lvl}\,\%$", xy=(R[at], data[lvl][at]),
                   xytext=(dx, dy), textcoords="offset points",
                   ha=ha, color=col, fontsize=8)
    if ref is not None:
        a.axhline(ref, color=GREY, lw=1.0, ls="--", zorder=2)
        a.annotate(refname, xy=(refat, ref), xytext=refdx, ha=refha,
                   textcoords="offset points", color=GREY, fontsize=7.5)
    a.set_xscale("log", base=2)
    a.set_xticks(R)
    a.set_xticklabels([str(v) for v in R])
    a.set_xlabel(r"bundle size $r$")
    a.set_ylabel(ylab)
    a.grid(True, color="0.88", lw=0.5)
    a.set_axisbelow(True)
    a.text(lpos[0], lpos[1], lab, transform=a.transAxes,
           fontsize=9, va="top")
    a.set_box_aspect(1/1.3)

panel(ax[0], SNR, "SNR (dB)", "(a)", dyl=(7, -12))
panel(ax[1], D2, r"$D_2$", "(b)", CLEAN_D2, "clean\n\n", dyl=(-1, 7),
      at=0, ha="left", dx=4, lpos=(0.88, 0.94), refat=1, refha="left",
      refdx=(4, -11))
panel(ax[2], L1, r"$\hat\lambda_1$", "(c)", CLEAN_L1, "clean",
      dyl=(-11, 7), refat=1, refha="left", refdx=(4, -11),
      lpos=(0.05, 0.80))
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "bundlefig.pdf"))
print("wrote figures/bundlefig.pdf")
