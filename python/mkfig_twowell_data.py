"""Fig. twowelldata: the book's measured record at a glance.

(a) 40 forcing periods of the strain-gauge voltage
(b) amplitude distribution over the whole record
(c) Welch power spectral density, showing the drive line and its harmonics
(d) delay reconstruction, tau = 5 samples, with the stroboscopic section

Run from python/: writes ../figures/twowell_data.pdf.
"""
import numpy as np
from scipy.signal import welch
from _figstyle import *
from _twowell import twowell, FS, NPP, section

x = twowell()
n0 = 3000                                   # skip the first 187 forcing periods
seg = x[n0:n0 + 40*NPP]
c = np.arange(len(seg))/NPP

fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.4))

ax[0, 0].plot(c, seg, color=COL[0], lw=0.8)
ax[0, 0].set_xlabel("forcing periods"); ax[0, 0].set_ylabel(r"$x$ (V)")
ax[0, 0].set_xlim(0, 40); panel(ax[0, 0], "a")

# bin edges on the quantization grid: a bin width that is not an integer
# number of A/D steps puts a comb in the histogram
LSB = 10.0/65536
w = 512*LSB
e = np.arange(np.floor(x.min()/w), np.ceil(x.max()/w) + 1)*w
ax[0, 1].hist(x, bins=e, color=COL[0], edgecolor="none")
ax[0, 1].set_xlabel(r"$x$ (V)"); ax[0, 1].set_ylabel("count")
panel(ax[0, 1], "b")

f, P = welch(x - x.mean(), fs=FS, nperseg=8192)
ax[1, 0].semilogy(f, P, color=COL[0], lw=0.7)
fd = FS/NPP
ax[1, 0].plot([fd, 2*fd, 3*fd], [1e-1]*3, "v", color=COL[1], ms=3.5, clip_on=False)
for k, lb in zip((1, 2, 3), (r"$\quad f_{\mathrm{d}}$", r"$2f_{\mathrm{d}}$", r"$3f_{\mathrm{d}}$")):
    ax[1, 0].text(k*fd, 2.8e-1, lb, color=COL[1], fontsize=8, ha="center")
ax[1, 0].set_xlim(0, 40); ax[1, 0].set_ylim(1e-8, 1); ax[1, 0].set_xlabel(r"$f$ (Hz)")
ax[1, 0].set_ylabel(r"$S_{xx}$ (V$^2$/Hz)"); panel(ax[1, 0], "c")

tau = 5
ax[1, 1].plot(x[:-tau], x[tau:], ".", color=COL[0], ms=0.35, alpha=0.25,
              mec="none", rasterized=True)
s = section(x)
ax[1, 1].plot(s[:-1], s[1:], ".", color=COL[1], ms=1.0, mec="none", rasterized=True)
ax[1, 1].set_xlabel(r"$x(t)$ (V)"); ax[1, 1].set_ylabel(r"$x(t+5\Delta t)$ (V)")
panel(ax[1, 1], "d")

fig.tight_layout()
fig.savefig(FIGDIR + "twowell_data.pdf", dpi=600)
print("wrote twowell_data.pdf")
