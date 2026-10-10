"""entropy.pdf (label entropy): the summand -p ln p of the Shannon entropy against p on
[0, 1] (natural logarithm, nats). It vanishes at both ends -- the limit p -> 0 is the one
proved in the text -- and has its maximum 1/e = 0.3679 at p = 1/e (dotted line)."""
import numpy as np
from _figstyle import *

p = np.linspace(0, 1, 1001); f = np.where(p > 0, -p*np.log(np.where(p > 0, p, 1)), 0.0)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(p, f, color=COL[0], lw=1.2)
ax.axvline(1/np.e, color="k", lw=0.6, ls=":")
ax.plot(1/np.e, 1/np.e, "o", color=COL[1], ms=3.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 0.4)
ax.set_xticks([0, 0.2, 1/np.e, 0.6, 0.8, 1]); ax.set_xticklabels(["0", "0.2", "$1/e$", "0.6", "0.8", "1"])
ax.set_xlabel("$p$"); ax.set_ylabel(r"$-p\ln p$ (nats)")
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.19, top=0.95)
fig.savefig(FIGDIR + "entropy.pdf")
print("max of -p ln p: %.4f at p = %.4f (1/e = %.4f)" % (f.max(), p[f.argmax()], 1/np.e))
