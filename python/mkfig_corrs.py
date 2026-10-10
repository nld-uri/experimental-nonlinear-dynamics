"""corrs.pdf (label corrs): scatter plots of N = 300 pairs (x, y). (a) highly correlated:
x ~ N(0,1), y = x + 0.15 e with e ~ N(0,1) independent; (b) uncorrelated: x and y
independent N(0,1). Seed 0. The sample correlation coefficients are printed and quoted
in the caption."""
import numpy as np
from _figstyle import *

N, S = 300, 0.15
rng = np.random.default_rng(0)
x = rng.standard_normal(N); y1 = x + S*rng.standard_normal(N)
x2 = rng.standard_normal(N); y2 = rng.standard_normal(N)
r1 = np.corrcoef(x, y1)[0, 1]; r2 = np.corrcoef(x2, y2)[0, 1]
print("rho_xy (a) = %.3f  (theory %.3f); rho_xy (b) = %.3f" % (r1, 1/np.sqrt(1+S**2), r2))
fig, axs = plt.subplots(1, 2, figsize=(6, 2.9))
for ax, (xx, yy), lab in zip(axs, ((x, y1), (x2, y2)), "ab"):
    ax.plot(xx, yy, ".", color=COL[0], ms=3)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$", rotation=0); panel(ax, lab, x=-0.15)
fig.tight_layout()
fig.savefig(FIGDIR+"corrs.pdf")
