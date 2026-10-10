"""Multitaper spectral estimate and harmonic F-test (Chapter 6).

    f, G = psdmt(x, fs, NW)                     # one-sided auto-spectrum
    f, S = psdmt(x, fs, NW, K, adaptive, nfft)  # dict: G, Gavg, F, mu, nu, lam, K

The whole record is tapered by K discrete prolate spheroidal sequences
(Slepian tapers) of time-bandwidth product NW, half-bandwidth W = NW/(N dt).
Each taper gives an eigenspectrum; they are combined either by plain
averaging (adaptive=False) or by Thomson's adaptive weights (default).
Densities are one-sided (f >= 0) in x^2/Hz, so sum(G)*f[1] ~ var(x).

F is Thomson's harmonic F statistic with 2 and 2K-2 degrees of freedom,
mu the complex line amplitude estimate at each frequency (a line
A cos(2 pi f0 t + phi) has |mu(f0)| = A/2), and nu the equivalent degrees
of freedom of the adaptive estimate at each frequency.

Copyright by David Chelidze. Written for the second edition.
"""
import numpy as np
from scipy.signal.windows import dpss


def psdmt(x, fs, NW=4.0, K=None, adaptive=True, nfft=None, full=False):
    x = np.asarray(x, float).ravel()
    x = x - x.mean()
    N = x.size
    K = int(2*NW) - 1 if K is None else K
    nfft = N if nfft is None else max(int(nfft), N)
    v, lam = dpss(N, NW, Kmax=K, return_ratios=True)    # K x N, unit energy
    nf = nfft//2 + 1
    f = np.arange(nf)*fs/nfft
    dt = 1.0/fs
    y = np.fft.rfft(v*x[None, :], n=nfft, axis=1)        # eigencoefficients, K x nf
    Sk = dt*np.abs(y)**2                                 # eigenspectra (two-sided)
    Savg = Sk.mean(axis=0)
    if adaptive:
        sig2 = x.var()*dt                                # white level, two-sided
        S = Sk[:2].mean(axis=0)
        for _ in range(100):
            b = S[None, :]/(lam[:, None]*S[None, :] + (1 - lam[:, None])*sig2)
            d2 = lam[:, None]*b**2                       # d_k^2
            Snew = np.sum(d2*Sk, axis=0)/np.sum(d2, axis=0)
            if np.max(np.abs(Snew - S)/S) < 1e-6:
                S = Snew
                break
            S = Snew
        nu = 2*np.sum(d2, axis=0)**2/np.sum(d2**2, axis=0)
    else:
        S = Savg
        nu = 2*K*np.ones(nf)
    one = np.ones(nf)*2                                  # one-sided: double interior
    one[0] = 1
    if nfft % 2 == 0:
        one[-1] = 1
    G = one*S
    if not full:
        return f, G
    # harmonic F-test (Thomson 1982): only even tapers have V_k(0) != 0
    V0 = v.sum(axis=1)                                   # K
    mu = (V0[:, None]*y).sum(axis=0)/np.sum(V0**2)       # complex, units of x
    resid = np.sum(np.abs(y - mu[None, :]*V0[:, None])**2, axis=0)
    F = (K - 1)*np.abs(mu)**2*np.sum(V0**2)/resid
    return f, dict(G=G, Gavg=one*Savg, F=F, mu=mu, nu=nu, lam=lam, K=K)


def mtlines(x, fs, NW=4.0, K=None, p=None, pad=8):
    """Detect lines by the harmonic F-test and remove them (Thomson 1982).

    Returns xc, the record with every detected sinusoid subtracted, and an
    array of rows (f0, A, phase, F) for the lines found.  A line is a local
    maximum of F above the 1 - p point of F(2, 2K-2) on a grid zero-padded
    by `pad`; p defaults to 1/N.  With pad = 8 the measured false-line rate
    for white noise (N = 1024) is 1.8 per record, not N p/2 = 0.5, because the
    padded grid searches the maxima of F between the Fourier frequencies.
    The continuum is then estimated by psdmt(xc, ...).
    """
    from scipy.stats import f as fdist
    x = np.asarray(x, float).ravel()
    x = x - x.mean()
    N = x.size
    K = int(2*NW) - 1 if K is None else K
    p = 1.0/N if p is None else p
    f, S = psdmt(x, fs, NW, K, adaptive=False, nfft=pad*N, full=True)
    F = S["F"]
    thr = fdist.ppf(1 - p, 2, 2*K - 2)
    pk = (F > thr) & (F >= np.r_[0, F[:-1]]) & (F >= np.r_[F[1:], 0])
    pk[0] = pk[-1] = False                     # no lines at 0 or Nyquist
    t = np.arange(N)/fs
    xc = x.copy()
    rows = []
    for j in np.flatnonzero(pk):
        mu = S["mu"][j]
        xc -= 2*np.real(mu*np.exp(2j*np.pi*f[j]*t))
        rows.append((f[j], 2*np.abs(mu), np.angle(mu), F[j]))
    return xc, np.array(rows).reshape(-1, 4)
