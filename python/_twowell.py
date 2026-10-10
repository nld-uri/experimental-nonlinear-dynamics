"""The book's measured record: two-well magneto-elastic oscillator, 2002-06-02.

Strain-gauge output of the notched cantilever beam of Chelidze and Liu,
J. Sound Vib. 281 (2004) 887-904, doi:10.1016/j.jsv.2004.02.017.

    x   strain-gauge voltage, V, one column, no header
    N   96,000 samples = 600 s = 6000 forcing periods
    fs  160 Hz
    fd  10 Hz forcing, so 16 samples per forcing period

fs and fd are as recorded at the time. Both are confirmed by the file
itself: the only sharp spectral lines sit at fs/16, fs/8 and 3fs/16, and
every sample value lies on a grid of 10/65536 V, a 16-bit converter over
+/-5 V.
"""
import os
import numpy as np

FS = 160.0                      # Hz, sampling frequency
FD = 10.0                       # Hz, forcing frequency
NPP = 16                        # samples per forcing period, = FS/FD
LSB = 10.0/65536                # V, quantization step recovered from the record
DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "twowell_2002-06-02.csv")


def twowell(n=None, detrend=False):
    """Return the record. n truncates it; detrend removes the mean."""
    x = np.loadtxt(DATA)
    if n is not None:
        x = x[:n]
    return x - x.mean() if detrend else x


def tvec(x, fs=FS):
    """Time vector in seconds for a record sampled at fs."""
    return np.arange(len(x))/fs


def cycles(x, npp=NPP):
    """Time vector in forcing periods -- independent of the value of fs."""
    return np.arange(len(x))/npp


def section(x, npp=NPP, phase=0):
    """Stroboscopic (Poincare) section, one point per forcing period."""
    return x[phase::npp]
