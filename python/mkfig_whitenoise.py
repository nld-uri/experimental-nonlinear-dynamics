"""whitenoise.pdf (label white): the pair S_xx(f) = 1 (flat, all f) and R_xx(tau) = delta(tau),
the delta drawn as a unit arrow at tau = 0. Pure formula, no random data."""
import numpy as np
from _figstyle import *

fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 2.0))
f = np.linspace(-5, 5, 2)
a1.plot(f, np.ones(2), color=COL[0], lw=1.2)
a1.set_ylim(-0.2, 1.6); a1.set_yticks([0, 1]); a1.set_xlim(-5, 5); a1.set_xticks([0])
a1.set_xlabel("$f$"); a1.set_ylabel("$S_{xx}(f)$")
a2.annotate("", (0, 1.4), (0, 0), arrowprops=dict(arrowstyle="->", lw=1.2, color=COL[0]))
a2.set_ylim(-0.1, 1.6); a2.set_yticks([0]); a2.set_xlim(-5, 5); a2.set_xticks([0])
a2.set_xlabel(r"$\tau$"); a2.set_ylabel(r"$R_{xx}(\tau)$")
a2.text(0.15, 1.3, r"$\delta(\tau)$", fontsize=8)
fig.tight_layout(h_pad=0.4); fig.savefig(FIGDIR+"whitenoise.pdf")
