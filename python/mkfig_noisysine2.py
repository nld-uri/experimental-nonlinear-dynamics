"""noisysine2.pdf (label nsine2): (a) x(t) = a sin(t + t0) for three values of the random
phase t0 drawn uniformly on [0, 2 pi) (seed 0; the drawn values are printed), a = 1,
t in [0, 6 pi]; (b) the density of x at any fixed t when t0 is uniform, the arcsine
density p(x) = 1/(pi sqrt(a^2 - x^2)) on |x| < a, compared with a histogram of
x(t = 0) over 20000 random phases."""
import numpy as np
from _figstyle import *

A, DT = 1.0, 0.01
rng = np.random.default_rng(0)
t0 = rng.uniform(0, 2*np.pi, 3)
print("t0 =", np.round(t0, 3))
t = np.arange(0, 6*np.pi+DT/2, DT)
fig = plt.figure(figsize=(6, 2.2))
gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1], wspace=0.15)
ax = fig.add_subplot(gs[0]); axp = fig.add_subplot(gs[1], sharey=ax)
sketch_axes(ax, "$t$", "$x(t)$")
for k, ph in enumerate(t0):
    ax.plot(t, A*np.sin(t+ph), color=COL[k], lw=0.9, label="$t_0 = %.2f$" % ph)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.04), ncol=3, fontsize=7, handlelength=1.5)
ax.set_yticks([-A, A]); ax.set_yticklabels(["$-a$", "$a$"]); ax.tick_params(axis="y", length=3)
ax.set_xlim(-0.3, 6*np.pi+0.3); ax.set_ylim(-1.5*A, 1.5*A)
ax.yaxis.set_label_coords(0.04, 1.02); panel(ax, "a", x=-0.05)
# arcsine density
xs = np.linspace(-A*0.9995, A*0.9995, 1000)
p = 1/(np.pi*np.sqrt(A**2-xs**2))
h, e = np.histogram(A*np.sin(rng.uniform(0, 2*np.pi, 20000)), bins=25, range=(-A, A), density=True)
axp.barh(0.5*(e[1:]+e[:-1]), h, height=e[1]-e[0], color=COL[0], alpha=0.3, lw=0)
axp.plot(p, xs, color=COL[0], lw=1.0)
for s in ("top", "right"):
    axp.spines[s].set_visible(False)
axp.set_xlim(0, 2.5); axp.set_xticks([0, 1, 2])
axp.tick_params(labelleft=False, axis="y", length=3)
axp.set_yticks([-A, A])
axp.set_xlabel("$p(x)$"); axp.set_ylabel("$x$", rotation=0); axp.yaxis.set_label_coords(0.06, 1.02)
axp.axhline(A, color="k", lw=0.5, ls=":"); axp.axhline(-A, color="k", lw=0.5, ls=":")
panel(axp, "b", x=-0.14)
fig.subplots_adjust(left=0.05, right=0.98, bottom=0.18, top=0.88)
fig.savefig(FIGDIR+"noisysine2.pdf")
