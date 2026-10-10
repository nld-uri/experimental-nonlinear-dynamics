"""specest_windows.pdf: magnitude of the spectral window |W(f)| in dB for the
rectangular, Hann, Hamming, Blackman and flat-top tapers, L = 64."""
import numpy as np
from scipy.signal import get_window
from _sestyle import *
L, nfft = 64, 64*256
bins = np.arange(nfft)/nfft*L; bins[bins > L/2] -= L
o = np.argsort(bins)
fig, ax = plt.subplots(figsize=(6, 2.4))
for name, c, lab in [("boxcar", COL[0], "rectangular"), ("hann", COL[1], "Hann"), ("hamming", COL[2], "Hamming"),
                     ("blackman", COL[3], "Blackman"), ("flattop", COL[4], "flat-top")]:
    w = get_window(name, L); W = np.abs(np.fft.fft(w, nfft)); W /= W.max()
    ax.plot(bins[o], 20*np.log10(W[o]+1e-30), color=c, lw=0.9, label=lab)
ax.set_xlim(-10, 10); ax.set_ylim(-100, 3); ax.set_xlabel("frequency offset from the bin center (bins of $1/T$)")
ax.set_ylabel("$|W(f)|/|W(0)|$ (dB)"); ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout(); fig.savefig(FIGDIR+"specest_windows.pdf")
