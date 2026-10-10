"""specest_ar2psd.pdf: periodogram vs Welch estimate for the AR(2) process of
Chapter 3 (rho = 0.05), with the exact two-sided spectrum overlaid."""
import numpy as np
from _sestyle import *
from psdwelch import psdwelch

# OVERRIDE FIGDIR: Uses a relative directory so it runs seamlessly on Mac and Linux
FIGDIR = "../figures/" 

rng = np.random.default_rng(11)
N, dt, rho = 4096, 1.0, 0.05
x = ar2(rho, N, rng); x -= x.mean()
X = np.fft.rfft(x); f = np.fft.rfftfreq(N, dt)
P = dt/N*np.abs(X)**2                      # two-sided periodogram on f >= 0
fw, G = psdwelch(x, 1/dt, 512, 256, "hann")
S = ar2_psd(rho, f, dt)
fig, ax = plt.subplots(1, 2, figsize=(6, 2.6), sharey=True)
ax[0].semilogy(f[1:], P[1:], color=COL[0], lw=0.6, label="periodogram, $N=4096$")
ax[0].semilogy(f, S, color=COL[1], lw=1.4, label="exact $S_{xx}(f)$")
ax[1].semilogy(fw[1:], G[1:]/2, color=COL[0], lw=1.2, label="Welch, Hann,$L=512$, \n 50% overlap")
ax[1].semilogy(f, S, color=COL[1], lw=1.4, label="exact $S_{xx}(f)$")
for a, t in zip(ax, ["(a)", "(b)"]):
    a.set_xlabel("$f$ (Hz)"); a.set_xlim(0, 0.5); a.legend(loc="upper right")
    a.text(0.02, 0.05, t, transform=a.transAxes)
ax[0].set_ylabel("$\\hat S_{xx}(f)$ (x$^2$/Hz)"); ax[0].set_ylim(1e-3, 1e5)
fig.tight_layout(); fig.savefig(FIGDIR+"specest_ar2psd.pdf")
print("K =", (N-512)//256+1)
