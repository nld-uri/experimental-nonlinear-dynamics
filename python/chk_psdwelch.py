"""Test psdwelch.py against scipy.signal.welch / csd on a known case."""
import numpy as np
from scipy import signal
from psdwelch import psdwelch
rng = np.random.default_rng(0)
fs, L, D = 200.0, 512, 256
N = L*40
u = rng.standard_normal(N)
b, a, *_ = signal.cont2discrete(([1.0], [1.0, 2*0.05*2*np.pi*5, (2*np.pi*5)**2]), 1/fs, method="zoh")
y = signal.lfilter(b.ravel(), a.ravel(), u) + 1e-6*rng.standard_normal(N)
for win in ["hann", "boxcar", "hamming"]:
    f, S = psdwelch(u, fs, L, D, win, y)
    fr, Pxx = signal.welch(u, fs=fs, window=win, nperseg=L, noverlap=D, detrend="constant")
    _, Pyy = signal.welch(y, fs=fs, window=win, nperseg=L, noverlap=D, detrend="constant")
    _, Pxy = signal.csd(u, y, fs=fs, window=win, nperseg=L, noverlap=D, detrend="constant")
    _, Cxy = signal.coherence(u, y, fs=fs, window=win, nperseg=L, noverlap=D, detrend="constant")
    print(win, "max rel err Gxx:", np.max(np.abs(S["Gxx"][1:]-Pxx[1:])/Pxx[1:]), "Gyy:", np.max(np.abs(S["Gyy"][1:]-Pyy[1:])/Pyy[1:]),
          "Gxy:", np.max(np.abs(S["Gxy"][1:]-Pxy[1:])/np.abs(Pxy[1:])), "coh:", np.max(np.abs(S["coh"][1:]-Cxy[1:])), "f:", np.max(np.abs(f-fr)), "K=", S["K"])
f, G = psdwelch(u, fs, L, D, "hann")
print("Parseval: sum(G)*df =", G.sum()*f[1], " var(u) =", u.var())
fr, H = signal.freqz(b.ravel(), a.ravel(), worN=f, fs=fs)
i = (f > 1) & (f < 12)
print("max |H1/H - 1| in 1-20 Hz:", np.max(np.abs(S["H1"][i]/H[i]-1)), " coh min:", S["coh"][i].min())
