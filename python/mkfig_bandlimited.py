"""bandlimited.pdf (label band): band-limited white noise, S_xx(f) = 1 on |f| <= f_BW and 0
elsewhere, with its autocorrelation R_xx(tau) = sin(2 pi f_BW tau)/(pi tau) =
2 f_BW sinc(2 pi f_BW tau) (f_BW = 1). The script also computes the first zero of R_xx
(1/(2 f_BW)), the 1/e lag of rho_xx and its integral time scale, quoted in the caption."""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from _figstyle import *

FBW = 1.0
rho = lambda tau: np.sinc(2*FBW*tau)          # numpy sinc = sin(pi u)/(pi u)
te = brentq(lambda t: rho(t) - np.exp(-1), 1e-6, 0.5/FBW)
ti = quad(lambda t: 1/(2*np.pi*FBW*t), 1e-12, 200/FBW, weight='sin', wvar=2*np.pi*FBW, limit=5000)[0]  # -> 1/(4 f_BW) as the limit is extended
print("first zero 1/(2 f_BW) = %.4f/f_BW; 1/e lag = %.4f/f_BW; integral time scale = %.4f/f_BW" % (0.5, te*FBW, ti*FBW))
tau = np.linspace(-2.75, 2.75, 4001)/FBW
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 2.8))
f = np.linspace(-2, 2, 2001); S = (np.abs(f) <= FBW).astype(float)
a1.plot(f, S, color=COL[0], lw=1.2); a1.set_ylim(-0.1, 1.4); a1.set_xlim(-2, 2)
a1.set_xticks([-FBW, 0, FBW]); a1.set_xticklabels(["$-f_{BW}$", "0", "$f_{BW}$"]); a1.set_yticks([0, 1])
a1.set_xlabel("$f$"); a1.set_ylabel("$S_{xx}(f)$")
a2.plot(tau, 2*FBW*rho(tau), color=COL[0], lw=1.0); a2.axhline(0, color="k", lw=0.4)
a2.set_xlim(-2.75, 2.75); a2.set_ylim(-0.7, 2.4)
tk = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])/FBW
a2.set_xticks(tk); a2.set_xticklabels([r"$-\frac{2}{f_{BW}}$", r"$-\frac{3}{2f_{BW}}$", r"$-\frac{1}{f_{BW}}$", r"$-\frac{1}{2f_{BW}}$", "0",
                                        r"$\frac{1}{2f_{BW}}$", r"$\frac{1}{f_{BW}}$", r"$\frac{3}{2f_{BW}}$", r"$\frac{2}{f_{BW}}$"], fontsize=8)
a2.set_yticks([0, 2*FBW]); a2.set_yticklabels(["0", "$2f_{BW}$"])
a2.set_xlabel(r"$\tau$"); a2.set_ylabel(r"$R_{xx}(\tau)$")
a2.plot([te], [2*FBW*np.exp(-1)], "o", color=COL[1], ms=3); a2.text(te+0.05, 2*FBW*np.exp(-1)+0.1, r"$1/e$", fontsize=7, color=COL[1])
for ax in (a1, a2): ax.grid(True, lw=0.3, alpha=0.5)
fig.tight_layout(h_pad=0.6); fig.savefig(FIGDIR+"bandlimited.pdf")
