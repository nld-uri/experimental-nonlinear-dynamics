"""sampleFT.pdf (label sampleFT): (a) a band-limited transform X(f), the triangle
1 - |f|/f_BW on |f| < f_BW; (b) the transform of the sampled signal, X_s(f) = (1/T_s)
sum_n X(f - n/T_s), for f_s = 1/T_s = 3 f_BW (copies do not overlap: no aliasing) --
heights are scaled by T_s so that the copies have the height of X. The dashed box marks
the Nyquist band |f| < f_s/2."""
import numpy as np
from _figstyle import *

FBW, FS = 1.0, 3.0
tri = lambda f: np.clip(1 - np.abs(f)/FBW, 0, None)
fig = plt.figure(figsize=(6, 1.7))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.15)
a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])
f = np.linspace(-2, 2, 801)
a.fill_between(f, 0, tri(f), color=COL[0], alpha=0.5, lw=0); a.plot(f, tri(f), color=COL[0], lw=0.8)
sketch_axes(a, "$f$", "$X(f)$", yzero=False); a.axhline(0, color="k", lw=0.6); a.xaxis.set_label_coords(1.02, 0.1)
a.set_xticks([-FBW, 0, FBW]); a.set_xticklabels(["$-f_{BW}$", "0", "$f_{BW}$"], fontsize=8); a.tick_params(length=2)
a.set_xlim(-2, 2); a.set_ylim(-0.05, 1.3); a.yaxis.set_label_coords(0.5, 1.02); panel(a, "a", x=0.0, y=-0.35)
f = np.linspace(-2.5*FS, 2.5*FS, 4001)
Xs = sum(tri(f - n*FS) for n in range(-4, 5))
b.fill_between(f, 0, Xs, color=COL[0], alpha=0.5, lw=0); b.plot(f, Xs, color=COL[0], lw=0.8)
sketch_axes(b, "$f$", "$T_s X_s(f)$", yzero=False); b.axhline(0, color="k", lw=0.6); b.xaxis.set_label_coords(1.02, 0.1)
b.yaxis.set_label_coords(0.5, 1.02)
b.set_xticks([-2*FS, -FS, -FBW, 0, FBW, FS, 2*FS]); b.tick_params(length=2)
b.set_xticklabels(["$-2f_s$", "$-f_s$", "$-f_{BW}$", "0", "$f_{BW}$", "$f_s$", "$2f_s$"], fontsize=8)
b.plot([-FS/2, -FS/2, FS/2, FS/2], [0, 1.15, 1.15, 0], "k--", lw=0.6)
b.annotate("", (FBW, 0.3), (FS-FBW, 0.3), arrowprops=dict(arrowstyle="<->", lw=0.6)); b.text(FS/2, 0.36, "$f_s - 2f_{BW}$", ha="center", va="bottom", fontsize=6, bbox=dict(fc="w", ec="none", pad=0.5))
b.set_xlim(-2.5*FS, 2.5*FS); b.set_ylim(-0.05, 1.3); panel(b, "b", x=0.0, y=-0.35)
fig.subplots_adjust(left=0.02, right=0.97, bottom=0.22, top=0.85); fig.savefig(FIGDIR+"sampleFT.pdf")
