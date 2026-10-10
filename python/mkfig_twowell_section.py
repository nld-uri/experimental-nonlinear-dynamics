"""twowell_section.pdf (label twowellsection): the measured stroboscopic Poincare
section of the two-well oscillator record of Section sec:twowelldata, against the
section of the simulated two-well Duffing oscillator of Section sec:runningdata.

(a) the record strobed once per forcing period (every 16th sample), plotted in the
    delay pair (s_k, s_k') with s_k' the sample a quarter of a forcing period
    later (4 samples); 6000 section points, the whole record.
(b) the same delay pair for the model x'' + 0.25 x' - x + x^3 = 0.3 cos t, strobed
    at t mod T = 0 and t mod T = T/4 of the same periods (_ch08data.duffing_strobe,
    200 steps per period, 50 transient periods, 5000 section points).
(c), (d) the boxes marked in (a) and (b), each about one fifth of the plotted
    range on a side, on the layered part of each set.
(e) mean distance to the nearest neighbour in the section against the assumed
    number of samples per forcing period, all strides truncated to the same
    5300 points so that the comparison is not a point-count comparison.

The comparison of (a) with (b) is qualitative: the rig parameters are not matched
to the model, and the observable is strain, not displacement.
"""
import numpy as np
from scipy.spatial import cKDTree
from _figstyle import *
from _twowell import twowell, section, NPP, LSB
from _ch08data import duffing_strobe

QP = NPP//4                       # quarter forcing period, in samples
NEQ = 5300                        # points kept per stride in panel (e)
STRIDES = np.arange(12, 21)

x = twowell()
s0, s4 = section(x, phase=0), section(x, phase=QP)
n = min(s0.size, s4.size)
s0, s4 = s0[:n], s4[:n]

A = duffing_strobe(5000, 200, 0.25, 0.3, 0.0, 50)        # x at t mod T = 0
B = duffing_strobe(5000, 200, 0.25, 0.3, np.pi/2, 50)    # x at t mod T = T/4
m0, m4 = A[0], B[0]

print("measured section: %d points, s in [%.3f, %.3f] V, s' in [%.3f, %.3f] V"
      % (n, s0.min(), s0.max(), s4.min(), s4.max()))
print("model section:    %d points, x in [%.3f, %.3f]" % (m0.size, m0.min(), m0.max()))
print("quantization step %.6f V, %.0f A/D steps across the section"
      % (LSB, (s0.max() - s0.min())/LSB))


def meannn(u, v):
    """Mean distance to the nearest neighbour of a planar point set."""
    Y = np.column_stack((u, v))
    d, _ = cKDTree(Y).query(Y, k=2)
    return d[:, 1].mean()


dnn = []
for m in STRIDES:
    q = int(round(m/4))
    u, v = x[0::m], x[q::m]
    k = min(u.size, v.size, NEQ)
    dnn.append(meannn(u[:k], v[:k]))
    print("stride %2d (quarter %d), %d points, mean nearest-neighbour distance %.4f V"
          % (m, q, k, dnn[-1]))
dnn = np.array(dnn)
print("phase dependence at stride %d:" % NPP,
      ["%.4f" % meannn(*(lambda a, b: (a[:min(a.size, b.size)], b[:min(a.size, b.size)]))
                       (x[p::NPP], x[p+QP::NPP])) for p in (0, 4, 8, 12)])

# zoom boxes on the layered part of each set
BM = (-0.75, 0.25, 0.40, 1.40)
BS = (-0.75, -0.25, 0.15, 0.65)
inM = ((s0 > BM[0]) & (s0 < BM[1]) & (s4 > BM[2]) & (s4 < BM[3])).sum()
inS = ((m0 > BS[0]) & (m0 < BS[1]) & (m4 > BS[2]) & (m4 < BS[3])).sum()
print("points in the zoom box: measured %d of %d, model %d of %d"
      % (inM, n, inS, m0.size))

fig = plt.figure(figsize=(6.0, 4.9))
gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1.0], hspace=0.45, wspace=1.7)
axa = fig.add_subplot(gs[0, :3]); axb = fig.add_subplot(gs[0, 3:])
for a, (u, v, xl, yl, bx) in ((axa, (s0, s4, "$s_k$ (V)", "$s_k'$ (V)", BM)),
                              (axb, (m0, m4, "$x_k$", "$x_k'$", BS))):
    a.plot(u, v, ".", color=COL[0], ms=0.8, rasterized=True)
    a.plot([bx[0], bx[1], bx[1], bx[0], bx[0]], [bx[2], bx[2], bx[3], bx[3], bx[2]],
           "-", color=COL[1], lw=0.8)
    a.set_xlabel(xl); a.set_ylabel(yl)
axc = fig.add_subplot(gs[1, :2]); axd = fig.add_subplot(gs[1, 2:4])
for a, (u, v, xl, yl, bx) in ((axc, (s0, s4, "$s_k$ (V)", "$s_k'$ (V)", BM)),
                              (axd, (m0, m4, "$x_k$", "$x_k'$", BS))):
    a.plot(u, v, ".", color=COL[0], ms=2.0)
    a.set_xlim(bx[0], bx[1]); a.set_ylim(bx[2], bx[3])
    a.set_xlabel(xl); a.set_ylabel(yl)
    for sp in a.spines.values():
        sp.set_color(COL[1])
axe = fig.add_subplot(gs[1, 4:])
axe.plot(STRIDES, dnn, "o-", color=COL[0], ms=3, lw=0.9)
axe.plot([NPP], [dnn[list(STRIDES).index(NPP)]], "o", color=COL[1], ms=6,
         mfc="w", mew=1.3)
axe.set_xlabel("$n$ (samples per period)")
axe.set_xticks([12, 14, 16, 18, 20])
axe.set_ylabel(r"mean $r_{\mathrm{nn}}$ (V)")
axe.set_ylim(0, 0.04)
for a, p in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d"), (axe, "e")):
    panel(a, p, x=-0.22 if a in (axc, axd, axe) else -0.14)
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.95)
fig.savefig(FIGDIR + "twowell_section.pdf", dpi=400)
