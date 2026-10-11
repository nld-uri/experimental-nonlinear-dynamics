"""The measured tracking matrix of Chapter 17: fatigue in variable operating
conditions.

Data of Chelidze and Liu, Phil. Trans. R. Soc. A 366 (2008) 729-745,
doi:10.1098/rsta.2007.2124, supplied by the author. A stiffened cantilever
beam with a notch, driven at 10 Hz on a shaker, in the field of two
electromagnets whose supply voltages are varied harmonically at a 2:1
frequency ratio. The beam accumulates a fatigue crack at the notch and
fractures at about 12.5 h.

The file holds the finished tracking matrix. The raw strain record behind it
survives separately as `cd.mat` (variable `strn`, 7,080,000 samples at 160 Hz),
whose first 1728 x 2^12 samples are exactly these windows; see
`python/chk_rawmatch.py`. What this file holds:
each row is one phase-space-warping feature vector, computed in the paper
from 2^12 vibration samples in a five-dimensional reconstruction (delay 6
samples at 160 Hz) partitioned into 81 hypercuboids, five features per
region.

    Y  (1728, 405)  tracking matrix, one feature vector per row
    t  (1728,)      time in hours, 0 to 12.281, one record every 25.6 s
    v1, v2 (1728,)  the two electromagnet supply voltages, in the file's
                    own units (mean 5.0, amplitude 4.5)

NOTE ON THE FILE. The two voltage records are stored under the names
`damage1` and `damage2`. They are not damage: they are the operating
conditions, sinusoids at a 2:1 frequency ratio whose Lissajous figure is
the figure-eight of the source paper's figure 9(a). The damage variable is
not in the file; recovering it is the point of the example.
"""
import os
import numpy as np
from scipy.io import loadmat

_HERE = os.path.dirname(__file__)
MAT = os.path.join(_HERE, "..", "data", "psw_fatigue_2005.mat")
NHEALTHY = 1200          # rows used to identify the operating subspace

# The raw strain record. It is 54 MB, too large for the Overleaf tree, so it
# lives beside the project rather than in data/. Put cd.mat (or a renamed copy)
# at one of these paths, or set the environment variable PSW_RAW.
RAW_CANDIDATES = [
    os.environ.get("PSW_RAW", ""),
    os.path.join(_HERE, "..", "data", "cd_raw_strain_2005.mat"),
    os.path.join(_HERE, "..", "..", "cd_raw_strain_2005.mat"),
    os.path.join(_HERE, "..", "..", "cd.mat"),
]


def raw_path():
    """Path to the raw strain record, or raise with the paths tried."""
    for p in RAW_CANDIDATES:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "raw strain record not found; tried:\n  "
        + "\n  ".join(p for p in RAW_CANDIDATES if p)
        + "\nSet PSW_RAW to its location. See data/README.md.")


def raw():
    """The raw strain record: 7,080,000 samples at 160 Hz."""
    from scipy.io import loadmat as _lm
    return _lm(raw_path())["strn"].ravel()


def tracking():
    """Return Y (1728, 405), t (h), v1, v2."""
    d = loadmat(MAT)
    return (d["ss_psw"].T, d["t"].ravel(),
            d["damage1"].ravel(), d["damage2"].ravel())
