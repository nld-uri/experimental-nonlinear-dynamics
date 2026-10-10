"""DmuLDDs.pdf -- local slopes of the fixed-distance (left) and fixed-mass (right)
pointwise-dimension curves for embedding dimensions d = 1..DMAX, Lorenz (a-b) and
Duffing (c-d): the plateau stops moving once d exceeds the attractor dimension.

Records, tau, w: _ch11data; pwdim with k = 500, Nq = 4096 reference points, seed 0,
unit-variance records. The plateau value per d (mean of the local slope over the
ranges RANGE of DmuLDs) is printed for the caption.
"""
from _figstyle import *
from _ch11data import *

DMAX = 6
RANGE = {"Lorenz": (-3.0, -1.5), "Duffing": (-2.2, -1.0)}       # fixed mass (as DmuLDs)
RANGE_FD = {"Lorenz": (-3.0, -2.0), "Duffing": (-2.2, -1.4)}    # fixed distance: pwdim counts
# at most k neighbours, so <ln mu(eps)> saturates at eps_i(k) and the slope drops to zero
# there for every d; the narrower range keeps clear of it for d up to DMAX.

fig, axs = plt.subplots(2, 2, figsize=(6.0, 4.4))
for row, name in enumerate(["Lorenz", "Duffing"]):
    lo, hi = RANGE[name]
    plateau = {0: [], 1: []}
    for d in range(1, DMAX + 1):
        lnD, lnN = pointwise(name, d)
        for c in (0, 1):
            x, y = lnD[:, c], lnN[:, c]
            ok = np.isfinite(y) & (y != 0)
            x, y = x[ok], y[ok]
            s = np.gradient(y, x)
            axs[row, c].plot(x, s, color=COL[d - 1], lw=0.8, label="$d=%d$" % d)
            l2, h2 = (lo, hi) if c == 1 else RANGE_FD[name]
            m = (x >= l2) & (x <= h2)
            plateau[c].append(s[m].mean() if m.any() else np.nan)
    for c in (0, 1):
        ax = axs[row, c]
        ax.axvspan(*((lo, hi) if c == 1 else RANGE_FD[name]), color="0.5", alpha=0.12, lw=0)
        ax.set_ylim(0, 4); ax.set_xlim(-6.5, 0.5)
        ax.set_xlabel(r"$\ln\epsilon$" if c == 0 else r"$\langle\ln\epsilon_i(n)\rangle$")
        ax.set_ylabel(r"$\hat D_P$" if c == 0 else r"$\bar D_P$")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        panel(ax, "abcd"[2*row + c], x=-0.18)
    print(name, "plateau over %s (fd) / %s (fm), d = 1..%d" % (RANGE_FD[name], RANGE[name], DMAX))
    print("   fixed distance:", "  ".join("%.2f" % v for v in plateau[0]))
    print("   fixed mass:    ", "  ".join("%.2f" % v for v in plateau[1]))
fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(FIGDIR + "DmuLDDs.pdf")
print("wrote DmuLDDs.pdf")
