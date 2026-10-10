"""specest_leakage.pdf: periodogram of a unit-amplitude sinusoid whose
frequency falls (a) on a DFT bin and (b) midway between bins, with the
rectangular and the Hann window."""
import os                                 # Added to manage directories
import numpy as np
from scipy.signal import get_window
from _sestyle import *

# OVERRIDE FIGDIR: Uses a relative directory so it runs seamlessly on Mac and Linux
FIGDIR = "../figures/" 

N, fs = 256, 256.0                        # 1 Hz bins
t = np.arange(N)/fs; f = np.fft.rfftfreq(N, 1/fs)
def per(x, w):
    return (1/fs)/np.sum(w**2)*np.abs(np.fft.rfft(x*w))**2     # two-sided, x^2/Hz
fig, ax = plt.subplots(1, 2, figsize=(6, 2.5))
for a, f0, lab in zip(ax[:2], [40.0, 40.5], [r"(a) $\hspace{9.3} f_0 = 40$ Hz (on a bin)", r"(b) $\qquad\; f_0 = 40.5$ Hz (between bins)"]):
    x = np.cos(2*np.pi*f0*t)
    for w, c, name in [(np.ones(N), COL[0], "rectangular"), (get_window("hann", N), COL[1], "Hann")]:
        P = per(x, w); a.plot(f, 10*np.log10(P/P.max()+1e-30), color=c, marker=".", ms=2.5, lw=0.7, label=name)
    a.set_xlim(20, 60); a.set_ylim(-80, 5); a.set_xlabel("$f$ (Hz)")
    a.text(0.03, 0.05, lab, transform=a.transAxes, fontsize=7.5)
ax[0].set_ylabel("$\\hat S_{xx}$ relative to peak (dB)"); ax[0].legend(loc="upper right")
fig.tight_layout(); 
os.makedirs(os.path.dirname(FIGDIR + "specest_leakage.pdf"), exist_ok=True)
fig.savefig(FIGDIR+"specest_leakage.pdf")
