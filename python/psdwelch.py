"""Welch spectral estimates from one or two records (Chapter 6).

    f, G = psdwelch(x, fs, L, D, win)             # one-sided auto-spectrum
    f, S = psdwelch(x, fs, L, D, win, y)          # dict: Gxx, Gyy, Gxy, coh, H1, H2

Segments of L samples, overlapped by D samples, are tapered by win
(a string accepted by scipy.signal.get_window, or an array of length L),
transformed, and their scaled periodograms averaged.  Densities are
one-sided (f >= 0) in units of x^2/Hz, so that sum(G)*f[1] ~ var(x).
Gxy is the one-sided cross-spectrum, coh the ordinary coherence,
H1 = Gxy/Gxx and H2 = Gyy/Gyx the two FRF estimators (x = input).

Copyright by David Chelidze. Written for the second edition.
"""
import numpy as np
from scipy.signal import get_window


def _segments(x, L, D):
    """Return the K x L matrix of overlapped segments (mean removed)."""
    x = np.asarray(x, float).ravel()
    K = (x.size - L)//(L - D) + 1
    idx = np.arange(L)[None, :] + (L - D)*np.arange(K)[:, None]
    seg = x[idx]
    return seg - seg.mean(axis=1, keepdims=True)


def psdwelch(x, fs, L, D=None, win="hann", y=None):
    D = L//2 if D is None else D
    w = get_window(win, L, fftbins=True) if isinstance(win, str) else np.asarray(win, float)
    U = np.sum(w**2)                       # window power; makes E[G] = S for white noise
    nf = L//2 + 1
    f = np.arange(nf)*fs/L
    scale = np.ones(nf)*2/(fs*U)           # one-sided: double the interior bins
    scale[0] /= 2
    if L % 2 == 0:
        scale[-1] /= 2                     # Nyquist bin is not doubled either
    X = np.fft.rfft(_segments(x, L, D)*w, axis=1)
    Gxx = scale*np.mean(np.abs(X)**2, axis=0)
    if y is None:
        return f, Gxx
    Y = np.fft.rfft(_segments(y, L, D)*w, axis=1)
    Gyy = scale*np.mean(np.abs(Y)**2, axis=0)
    Gxy = scale*np.mean(np.conj(X)*Y, axis=0)
    out = dict(Gxx=Gxx, Gyy=Gyy, Gxy=Gxy,
               coh=np.abs(Gxy)**2/(Gxx*Gyy),
               H1=Gxy/Gxx, H2=Gyy/np.conj(Gxy), K=X.shape[0])
    return f, out
