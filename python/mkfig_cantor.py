"""middle_third.pdf and genCantor.pdf -- the first stages of the two Cantor
constructions of Chapter 11, drawn exactly from the interval arithmetic.

middle_third: stages n = 0, 1, 2 of the middle-thirds set; the removed open
middle thirds are drawn as dotted gaps with open brackets, the end points are
labelled, and the total length L_n = (2/3)^n is written at the right.

genCantor: the surviving set of the map of eq. (oneDmap) after n = 0, 1, 2
iterations for mu = MU. At stage 1 the strip [1/(2 mu), 1 - 1/(2 mu)] of width
Delta = (mu - 1)/mu leaves; at stage 2 each remaining interval of length
(1 - Delta)/2 loses its own middle strip of width Delta_1 = Delta (1 - Delta)/2.
"""
from _figstyle import *
from fractions import Fraction as Fr

def stages(gap_fraction, n):
    """Intervals after n stages when each interval loses the central fraction
    gap_fraction of its length; returns a list of lists of (lo, hi)."""
    out = [[(Fr(0), Fr(1))]]
    for _ in range(n):
        new = []
        for lo, hi in out[-1]:
            L = hi - lo
            a = L*(1 - gap_fraction)/2
            new += [(lo, lo + a), (hi - a, hi)]
        out.append(new)
    return out


def draw(ax, ivals, labels, right, ylab):
    for n, row in enumerate(ivals):
        y = -n
        # removed gaps: dotted, with outward brackets
        if n > 0:
            for (lo1, hi1), (lo2, hi2) in zip(row[:-1], row[1:]):
                if lo2 > hi1:
                    ax.plot([float(hi1), float(lo2)], [y, y], ":", color=COL[1], lw=1.0)
        for lo, hi in row:
            ax.plot([float(lo), float(hi)], [y, y], color=COL[0], lw=2.2, solid_capstyle="butt")
            ax.plot([float(lo), float(lo)], [y - 0.12, y + 0.12], color=COL[1], lw=1.0)
            ax.plot([float(hi), float(hi)], [y - 0.12, y + 0.12], color=COL[1], lw=1.0)
        for xpos, txt in labels[n]:
            ax.text(float(xpos), y + 0.18, txt, ha="center", va="bottom", fontsize=8)
        ax.text(1.06, y, right[n], ha="left", va="center", fontsize=9)
        ax.text(-0.06, y, ylab % n, ha="right", va="center", fontsize=9)
    ax.set_xlim(-0.2, 1.4); ax.set_ylim(-len(ivals) + 0.5, 0.6)
    ax.axis("off")


def frac(q):
    q = Fr(q)
    return "$%d$" % q.numerator if q.denominator == 1 else r"$\frac{%d}{%d}$" % (q.numerator, q.denominator)

# ---- middle thirds ----------------------------------------------------------
iv = stages(Fr(1, 3), 2)
labels = []
for row in iv:
    pts = sorted(set([p for seg in row for p in seg]))
    labels.append([(p, frac(p)) for p in pts])
right = ["$L_0=1$", r"$L_1=\frac{2}{3}$", r"$L_2=\frac{4}{9}=\frac{2^2}{3^2}$"]
fig, ax = plt.subplots(figsize=(4.2, 1.6))
draw(ax, iv, labels, right, "$n=%d$")
fig.tight_layout(pad=0.2)
fig.savefig(FIGDIR + "middle_third.pdf")
print("wrote middle_third.pdf; stage-2 intervals:", [(str(a), str(b)) for a, b in iv[2]])

# ---- generalized Cantor set of the map (oneDmap) ---------------------------
MU = Fr(5, 4)
Delta = (MU - 1)/MU
iv = stages(Delta, 2)
labels = [[(0, "$0$"), (1, "$1$")], [], []]
right = ["", r"$\Delta=\frac{\mu-1}{\mu}$", r"$\Delta_1=\frac{\Delta}{2}(1-\Delta)$"]
fig, ax = plt.subplots(figsize=(4.2, 1.6))
draw(ax, iv, labels, right, "$n=%d$")
# gap widths marked with arrows
def mark(ax, x1, x2, y, txt):
    ax.annotate("", xy=(float(x1), y), xytext=(float(x2), y),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="k"))
    ax.text(float(x1 + x2)/2, y + 0.06, txt, ha="center", va="bottom", fontsize=8)
mark(ax, iv[1][0][1], iv[1][1][0], -1 + 0.22, r"$\Delta$")
mark(ax, iv[2][0][1], iv[2][1][0], -2 + 0.22, r"$\Delta_1$")
mark(ax, iv[2][2][1], iv[2][3][0], -2 + 0.22, r"$\Delta_1$")
fig.tight_layout(pad=0.2)
fig.savefig(FIGDIR + "genCantor.pdf")
print("wrote genCantor.pdf  mu =", MU, "Delta =", Delta, "Delta_1 =", Delta*(1 - Delta)/2,
      "stage-2 gaps:", iv[2][1][0] - iv[2][0][1], iv[2][3][0] - iv[2][2][1])
