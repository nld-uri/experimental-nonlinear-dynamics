"""p1p2b.pdf (label p1p2b): the period-doubling bifurcation of the logistic map
x_{n+1} = mu x_n (1 - x_n) at mu = 3, shown by iterates from x_0 = 0.1, n = 0..40.
(a) mu = 2.9: the P-1 fixed point (mu-1)/mu = 0.6552 is stable (multiplier 2 - mu = -0.9),
    the approach is oscillatory and slow.
(b) mu = 3.1: the P-1 point (dotted) is unstable (multiplier -1.1) and the iterates settle
    on the P-2 orbit x_{a,b} = 1/2 + 1/(2 mu) +- sqrt(mu^2 - 2 mu - 3)/(2 mu) = 0.7646, 0.5580
    (dashed)."""
import numpy as np
from _figstyle import *

def logistic(mu, x0, n):
    x = np.empty(n + 1); x[0] = x0
    for k in range(n): x[k+1] = mu*x[k]*(1 - x[k])
    return x

NIT, X0 = 40, 0.1
fig, axs = plt.subplots(1, 2, figsize=(6, 2.1))
for ax, mu, p in [(axs[0], 2.9, "a"), (axs[1], 3.1, "b")]:
    x = logistic(mu, X0, NIT); xf = (mu - 1)/mu
    ax.plot(np.arange(NIT + 1), x, "-o", color=COL[0], lw=0.8, ms=3, mfc="w")
    ax.axhline(xf, color="k", lw=0.5, ls=":")
    ax.text(1.02, xf, r"$\frac{\mu-1}{\mu}$", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9)
    if mu > 3:
        r = np.sqrt(mu**2 - 2*mu - 3)/(2*mu); xa, xb = 0.5 + 0.5/mu + r, 0.5 + 0.5/mu - r
        for v in (xa, xb): ax.axhline(v, color=COL[1], lw=0.6, ls="--")
        ax.text(1.02, xa + 0.03, "$x_a$", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9, color=COL[1])
        ax.text(1.02, xb - 0.03, "$x_b$", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9, color=COL[1])
        print("mu = %g: P-1 %.4f (multiplier %.2f), P-2 x_a = %.4f, x_b = %.4f; last iterates %.4f %.4f" % (mu, xf, 2-mu, xa, xb, x[-2], x[-1]))
    else:
        print("mu = %g: P-1 %.4f (multiplier %.2f), |x_40 - x*| = %.3f" % (mu, xf, 2-mu, abs(x[-1]-xf)))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlim(-0.5, NIT + 0.5); ax.set_ylim(0, 0.9)
    ax.set_xlabel("$n$"); ax.set_ylabel("$x_n$")
    ax.text(0.97, 0.06, r"$\mu = %g$" % mu, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    panel(ax, p, x=-0.16)
fig.subplots_adjust(left=0.09, right=0.94, bottom=0.2, top=0.9, wspace=0.45)
fig.savefig(FIGDIR + "p1p2b.pdf")
