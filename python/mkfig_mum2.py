"""mum2.pdf (label mum2): stability of the P-1 fixed point x = (mu-1)/mu of the logistic
map x_{n+1} = mu x_n (1 - x_n).
(a) the multiplier 2 - mu over the stability range 1 < mu < 3; it changes sign at mu = 2.
(b) iterates x_n for mu = 1.5 from x_0 = 0.1: monotonic approach to (mu-1)/mu = 1/3.
(c) iterates x_n for mu = 2.7 from x_0 = 0.5: oscillatory approach to (mu-1)/mu = 0.6296.
Iterates 0..20 are shown; the dotted line marks the fixed point."""
import numpy as np
from _figstyle import *

def logistic(mu, x0, n):
    x = np.empty(n + 1); x[0] = x0
    for k in range(n): x[k+1] = mu*x[k]*(1 - x[k])
    return x

NIT = 20
fig, axs = plt.subplots(1, 3, figsize=(6, 1.9), gridspec_kw=dict(width_ratios=[1, 1.4, 1.4]))
ax = axs[0]
mu = np.linspace(1, 3, 3)
ax.plot(mu, 2 - mu, color=COL[0], lw=1.4)
ax.axhline(0, color="k", lw=0.5); ax.axvline(2, color="0.6", lw=0.5, ls=":")
ax.set_xlim(0.8, 3.2); ax.set_ylim(-1.2, 1.2); ax.set_xticks([1, 2, 3]); ax.set_yticks([-1, 0, 1])
ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$2-\mu$"); panel(ax, "a", x=-0.3)
for ax, mu, x0, p in [(axs[1], 1.5, 0.1, "b"), (axs[2], 2.7, 0.5, "c")]:
    x = logistic(mu, x0, NIT); xf = (mu - 1)/mu
    ax.plot(np.arange(NIT + 1), x, "-o", color=COL[0], lw=0.8, ms=3.5, mfc="w")
    ax.axhline(xf, color="k", lw=0.5, ls=":")
    ax.set_xlim(-0.5, NIT + 0.5); ax.set_ylim(0, 0.8)
    ax.set_yticks([0, 0.4, 0.8])
    ax.text(1.02, xf, r"$\frac{\mu-1}{\mu}$", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9)
    ax.set_xlabel("$n$"); ax.set_ylabel("$x_n$")
    ax.text(0.97, 0.06, r"$\mu = %g$" % mu, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    panel(ax, p, x=-0.22)
    print("mu = %g, x0 = %g: fixed point %.4f, multiplier %.2f, |x_20 - x*| = %.2e" % (mu, x0, xf, 2 - mu, abs(x[-1] - xf)))
fig.subplots_adjust(left=0.08, right=0.94, bottom=0.22, top=0.9, wspace=0.7)
fig.savefig(FIGDIR + "mum2.pdf")
