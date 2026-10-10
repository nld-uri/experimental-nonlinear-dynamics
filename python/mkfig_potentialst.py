"""potentialst.pdf (label potentialst): the three local shapes of a potential at an
equilibrium, V(delta) = V''(x_e) delta^2 / 2 with V'' = +1 (minimum, stable), the cubic
V = delta^3 (V'' = 0, inflection, semi-stable) and V'' = -1 (maximum, unstable), for
delta in [-1, 1]. The equilibrium is marked by a dot: col1 stable, col3 semi-stable,
col2 unstable."""
import numpy as np
from _figstyle import *

d = np.linspace(-1, 1, 201)
cases = [(d**2/2, r"$\frac{\mathrm{d}^2V}{\mathrm{d}x^2} > 0$", COL[0], "a"),
         (d**3, r"$\frac{\mathrm{d}^2V}{\mathrm{d}x^2} = 0$", COL[2], "b"),
         (-d**2/2, r"$\frac{\mathrm{d}^2V}{\mathrm{d}x^2} < 0$", COL[1], "c")]
fig, axs = plt.subplots(1, 3, figsize=(6, 1.7))
for ax, (V, lab, c, p) in zip(axs, cases):
    ax.plot(d, V, color="k", lw=1.0)
    ax.plot(0, 0, "o", color=c, ms=4, zorder=3)
    sketch_axes(ax, r"$\delta$", r"$V(\delta)$", yzero=False)
    ax.spines["left"].set_position(("data", 0)); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.6, 0.6)
    ax.yaxis.set_label_coords(0.55, 0.98)
    ax.text(0.97, 0.05 if p != "c" else 0.9, lab, transform=ax.transAxes, ha="right",
            va="bottom" if p != "c" else "top", fontsize=9)
    panel(ax, p, x=0.0, y=0.9)
fig.subplots_adjust(left=0.03, right=0.98, bottom=0.12, top=0.9, wspace=0.25)
fig.savefig(FIGDIR + "potentialst.pdf")
