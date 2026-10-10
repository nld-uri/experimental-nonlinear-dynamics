"""antial.pdf (label antial): magnitude of an eighth-order Butterworth low-pass filter,
|H(f)| = [1 + (f/f_c)^16]^(-1/2), in dB against f/f_c on a linear axis from 0 to 2 f_c,
as an example of an analog anti-aliasing filter: -3 dB at f_c and, at 2 f_c, the value
printed by the script (about -48 dB; the asymptotic roll-off is 6 x 8 = 48 dB/octave)."""
import numpy as np
from _figstyle import *

ORDER = 8
f = np.linspace(0, 2, 2001)
H = 20*np.log10(1/np.sqrt(1 + f**(2*ORDER)))
print("|H(f_c)| = %.2f dB, |H(2 f_c)| = %.1f dB, asymptotic slope %d dB/octave" % (H[np.argmin(abs(f-1))], H[-1], 6*ORDER))
fig, ax = plt.subplots(figsize=(6, 2.3))
ax.plot(f, H, color=COL[0], lw=1.2)
ax.axvline(1, color=COL[1], lw=0.75, ls=":"); ax.axhline(-3, color=COL[1], lw=0.75, ls=":")
ax.set_xlim(0, 2); ax.set_ylim(-52, 2); ax.set_xticks([0, 0.5, 1, 1.5, 2]); ax.set_xticklabels(["0", "$f_c/2$", "$f_c$", "$3f_c/2$", "$2f_c$"])
ax.set_xlabel("frequency $f$"); ax.set_ylabel("$|H(f)|$ (dB)"); ax.set_yticks([0, -3, -10, -20, -30, -40, -50])
ax.text(1.14, -8, "$f_c$ (cutoff frequency, $-3$ dB)", fontsize=8)
ax.text(1.1, -48, "%d dB/octave asymptotic roll-off" % (6*ORDER), fontsize=8, ha="left")
ax.grid(True, lw=0.3, alpha=0.5)
fig.tight_layout(); fig.savefig(FIGDIR+"antial.pdf")
