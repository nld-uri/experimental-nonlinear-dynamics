"""Shared 'generic continuous signal' for the sketches of Chapter 5 (not a listing).

signal(t): band-limited Gaussian noise, x(t) = sqrt(2/K) sum_k cos(2 pi f_k t + phi_k) with
K = 60 frequencies f_k uniform on (0, f_BW], f_BW = 1 Hz, and phases phi_k uniform on
[0, 2 pi), seed 0; unit variance, no power above f_BW. Evaluated on any time grid, so the
'continuous' curve and its samples come from the same function.
"""
import numpy as np

FBW, K = 1.0, 60
_rng = np.random.default_rng(0)
_fk = _rng.uniform(0, FBW, K)
_ph = _rng.uniform(0, 2*np.pi, K)


def signal(t):
    t = np.asarray(t, float)
    return np.sqrt(2.0/K)*np.cos(2*np.pi*np.outer(t, _fk) + _ph).sum(axis=1)
