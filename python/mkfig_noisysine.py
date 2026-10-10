"""noisysine.pdf (label nsine): x(t) = a sin t + n(t) for 100 independent sample
functions plotted on top of each other. a = 1, n(t) Gaussian white noise of standard
deviation sigma_n = 0.1 a, independent between samples and between sample functions,
t in [0, 6 pi] at dt = 0.01 (three periods), seed 0."""
import numpy as np
from _figstyle import *

A, SIG, M, DT = 1.0, 0.1, 100, 0.01
rng = np.random.default_rng(0)
t = np.arange(0, 6*np.pi+DT/2, DT)
X = A*np.sin(t)[None, :] + SIG*rng.standard_normal((M, t.size))
fig, ax = plt.subplots(figsize=(6, 2.2))
sketch_axes(ax, "$t$", "$x(t)$")
ax.plot(t, X.T, color=COL[0], lw=0.3, alpha=0.6)
ax.set_yticks([-A, A]); ax.set_yticklabels(["$-a$", "$a$"])
ax.tick_params(axis="y", length=3)
ax.set_xlim(-0.3, 6*np.pi+0.3); ax.set_ylim(-1.5*A, 1.5*A)
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.9)
fig.savefig(FIGDIR+"noisysine.pdf")
print("ensemble mean at t=pi/2: %.3f (a sin t = %.3f); ensemble std %.3f" % (X[:, np.argmin(abs(t-np.pi/2))].mean(), A, X.std(axis=0).mean()))
