"""windows.pdf (label windows): the five windows of Table tab:se_windows (rectangular, Hann,
Hamming, Blackman, flat-top; scipy.signal.get_window, L = 64 samples). (a) the window
sequences w_n; (b) the magnitude of their transforms |W(f)| in dB relative to the peak, over
the normalized frequency 0 to 1 in units of f_s/2 (x pi rad/sample), nfft = 64 x 256. Peak
sidelobe levels are printed by the script."""
import numpy as np
from scipy.signal import get_window
from _figstyle import *

L, NF = 64, 64*256
names = [("boxcar", "rectangular"), ("hann", "Hann"), ("hamming", "Hamming"), ("blackman", "Blackman"), ("flattop", "flat-top")]
fig, (a, b) = plt.subplots(1, 2, figsize=(6, 2.6), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.3))
n = np.arange(L); fr = np.arange(NF//2+1)/(NF/2)   # 0..1 (x pi rad/sample)
for k, (nm, lab) in enumerate(names):
    w = get_window(nm, L)
    W = np.abs(np.fft.rfft(w, NF)); W = 20*np.log10(W/W.max() + 1e-12)
    a.plot(n, w, color=COL[k], lw=0.9, label=lab); b.plot(fr, W, color=COL[k], lw=0.6, label=lab)
    main = np.argmax((np.diff(W) > 0) & (W[:-1] < -20))   # first minimum below -20 dB = end of the main lobe
    print("%-12s peak sidelobe %6.1f dB; first zero at %.3f x f_s/2 = %.2f bins" % (lab, W[main:].max(), fr[main], fr[main]*L/2))
a.set_xlabel("sample $n$"); a.set_ylabel("$w_n$"); a.set_xlim(0, L-1); a.set_ylim(-0.1, 1.05); 
b.set_xlabel(r"frequency $f/(f_s/2)$"); b.set_ylabel("$|W(f)|$ (dB)"); b.set_xlim(0, 1); b.set_ylim(-140, 5)
for ax in (a, b): ax.grid(True, lw=0.3, alpha=0.5)
panel(a, "a", x=-0.2); panel(b, "b", x=-0.2)
b.legend(loc="upper center", bbox_to_anchor=(-0.25, -0.22), ncol=5, fontsize=8)
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.3, top=0.92); fig.savefig(FIGDIR+"windows.pdf")
