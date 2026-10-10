"""sinc.pdf (label sinc): the transform of the rectangular window of width 2a,
W(f) = sin(2 pi f a)/(pi f) = 2a sinc(2 pi f a), on |f| <= 5/(2a); zeros at multiples of
1/(2a), peak 2a at f = 0. The script prints the first sidelobe level."""
import numpy as np
from _figstyle import *

a = 1.0
f = np.linspace(-2.5/a, 2.5/a, 4001)
W = 2*a*np.sinc(2*f*a)   # numpy sinc(u) = sin(pi u)/(pi u)
side = W[(f > 0.5/a) & (f < 1/a)]
print("first sidelobe: %.4f x 2a = %.1f dB" % (side.min()/(2*a), 20*np.log10(abs(side.min())/(2*a))))
fig, ax = plt.subplots(figsize=(6, 2.3))
ax.plot(f, W, color=COL[0], lw=1.0); ax.axhline(0, color="k", lw=0.4)
tk = np.arange(-5, 6)/(2*a)
ax.set_xticks(tk); ax.set_xticklabels([r"$-\frac{5}{2a}$", r"$-\frac{2}{a}$", r"$-\frac{3}{2a}$", r"$-\frac{1}{a}$", r"$-\frac{1}{2a}$", "0",
                                        r"$\frac{1}{2a}$", r"$\frac{1}{a}$", r"$\frac{3}{2a}$", r"$\frac{2}{a}$", r"$\frac{5}{2a}$"], fontsize=8)
ax.set_yticks([0, 2*a]); ax.set_yticklabels(["0", "$2a$"]); ax.set_xlim(f[0], f[-1]); ax.set_ylim(-0.6*a, 2.3*a)
ax.set_xlabel("$f$"); ax.set_ylabel("$W(f)$"); ax.grid(True, lw=0.3, alpha=0.5)
fig.tight_layout(); fig.savefig(FIGDIR+"sinc.pdf")
