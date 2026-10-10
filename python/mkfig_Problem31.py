"""Problem31.pdf (label p31): N = 30 pairwise observations (X, Y) with X uniform on [0, 10],
Y = 2 X + 1 + 1.5 e, e ~ N(0,1), seed 0, and the least-squares line Y_p = alpha X + beta
fitted to them (numpy.polyfit)."""
import numpy as np
from _figstyle import *

N, A, B, S = 30, 2.0, 1.0, 1.5
rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 10, N)); Y = A*X + B + S*rng.standard_normal(N)
a, b = np.polyfit(X, Y, 1)
print("fit: alpha = %.3f, beta = %.3f; rho_XY = %.3f" % (a, b, np.corrcoef(X, Y)[0, 1]))
fig, ax = plt.subplots(figsize=(3, 2.3))
ax.plot(X, Y, "+", color=COL[0], ms=5, mew=0.9, label="observations")
xx = np.array([0, 10]); ax.plot(xx, a*xx+b, color=COL[1], lw=1.0, label=r"$Y_p = \widehat\alpha X + \widehat\beta$")
ax.set_xlabel("$X$"); ax.set_ylabel("$Y$"); ax.legend(loc="upper left", fontsize=7)
ax.set_xlim(0, 10.5); ax.set_ylim(0, 25)
fig.tight_layout(); fig.savefig(FIGDIR+"Problem31.pdf")
