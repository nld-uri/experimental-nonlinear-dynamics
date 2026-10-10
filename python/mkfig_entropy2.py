"""entropy2.pdf (label entropy2): Shannon entropy of a two-box partition,
H = -p_1 ln p_1 - p_2 ln p_2 with p_2 = 1 - p_1 (nats), against p_1 (bottom axis) and p_2
(top axis). The maximum ln 2 = 0.6931 nats (= 1 bit) is at p_1 = p_2 = 1/2 (dotted line)."""
import numpy as np
from _figstyle import *

p = np.linspace(0, 1, 1001)
def plogp(q): return np.where(q > 0, q*np.log(np.where(q > 0, q, 1)), 0.0)
H = -plogp(p) - plogp(1 - p)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(p, H, color=COL[0], lw=1.2)
ax.axvline(0.5, color="k", lw=0.6, ls=":"); ax.axhline(np.log(2), color="k", lw=0.6, ls=":")
ax.set_xlim(0, 1); ax.set_ylim(0, 0.75)
ax.set_yticks([0, 0.2, 0.4, np.log(2)]); ax.set_yticklabels(["0", "0.2", "0.4", r"$\ln 2$"])
ax.set_xlabel("probability $p_1$"); ax.set_ylabel(r"entropy $H({\cal A})$ (nats)")
ax2 = ax.secondary_xaxis("top", functions=(lambda q: 1 - q, lambda q: 1 - q))
ax2.set_xlabel("probability $p_2 = 1 - p_1$")
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.18, top=0.82)
fig.savefig(FIGDIR + "entropy2.pdf")
print("max H = %.4f at p_1 = %.3f (ln 2 = %.4f)" % (H.max(), p[H.argmax()], np.log(2)))
