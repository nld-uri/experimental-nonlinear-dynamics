"""modulation.pdf (label mod): a band-limited spectrum X(f), taken as the triangle
X(f) = 1 - |f|/f_BW on |f| < f_BW (f_BW = 1), and the spectrum of x(t) cos(2 pi f_c t),
Y(f) = [X(f - f_c) + X(f + f_c)]/2, with f_c = 4 f_BW: two half-height copies at +-f_c."""
import numpy as np
from _figstyle import *

FBW, FC = 1.0, 4.0
tri = lambda f: np.clip(1 - np.abs(f)/FBW, 0, None)
f = np.linspace(-6, 6, 4001)
fig = plt.figure(figsize=(6, 1.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.25)
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
for ax, y, lab in ((ax1, tri(f), "$X(f)$"), (ax2, 0.5*(tri(f-FC)+tri(f+FC)), "$Y(f)$")):
    ax.fill_between(f, 0, y, color=COL[0], alpha=0.5, lw=0); ax.plot(f, y, color=COL[0], lw=0.8)
    sketch_axes(ax, "$f$", lab, yzero=False); ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-0.05, 1.25); ax.xaxis.set_label_coords(1.02, 0.1)
ax1.set_xlim(-2.5, 2.5); ax1.set_xticks([-FBW, 0, FBW]); ax1.set_xticklabels(["$-f_{BW}$", "0", "$f_{BW}$"], fontsize=8)
ax2.set_xlim(-6, 6); ax2.yaxis.set_label_coords(0.51, 1.02); ax2.set_xticks([-FC, 0, FC]); ax2.set_xticklabels(["$-f_c$", "0", "$f_c$"], fontsize=8)
ax2.set_yticks([1]); ax2.set_yticklabels(["1"], fontsize=8); ax1.set_yticks([1]); ax1.set_yticklabels(["1"], fontsize=8)
for ax in (ax1, ax2): ax.tick_params(length=2)
ax2.plot([-FC, -FC], [0, 0.5], "k:", lw=0.6); ax2.plot([FC, FC], [0, 0.5], "k:", lw=0.6)
ax2.text(FC+0.35, 0.55, r"$\frac{1}{2}$", fontsize=8, va="center")
fig.text(0.355, 0.55, "modulation\n$\\longrightarrow$", ha="center", va="center", fontsize=8)
fig.subplots_adjust(left=0.03, right=0.97, bottom=0.2, top=0.85)
fig.savefig(FIGDIR+"modulation.pdf")
