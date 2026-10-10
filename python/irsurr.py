"""Iteratively refined surrogates.

Python port of irsurr.m (Appendix A.8), after Schreiber and Schmitz,
Surrogate time series, Physica D 142 (2000) 346-382.

Copyright by David Chelidze. Python port for the second edition.
"""
import numpy as np


def irsurr(s, n, rng=None):
    """Return (rb, sb): surrogates matching the histogram, and the PSD.

    s : original scalar time series
    n : number of refinement iterations

    rb matches the amplitude distribution of s exactly; sb matches its
    Fourier magnitudes exactly. Neither matches both, which is the point
    of the method -- report which one you used.
    """
    rng = np.random.default_rng() if rng is None else rng
    s = np.asarray(s, float).ravel()
    N = s.size
    c = np.sort(s)                              # the original magnitudes
    S = np.abs(np.fft.fft(s))                   # the original magnitudes of the FFT
    # start from a phase randomization of the original signal
    rb = np.real(np.fft.ifft(np.fft.fft(s)*np.exp(1j*2*np.pi*rng.random(N))))
    sb = rb.copy()
    for _ in range(n):
        sb = np.real(np.fft.ifft(S*np.exp(1j*np.angle(np.fft.fft(rb)))))
        rb[np.argsort(sb)] = c                  # match the distribution
    return rb, sb
