"""Shared plotting style for the Chapter 6 figures (not a listing)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
COL = [(0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.929, 0.694, 0.125),
       (0.494, 0.184, 0.556), (0.466, 0.674, 0.188)]
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "mathtext.fontset": "cm",
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "legend.frameon": False,
    "legend.fontsize": 8, "axes.labelsize": 9})
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures") + os.sep

def ar2(rho, N, rng, burn=2000):
    a1 = 2-(np.pi/10)**2-rho; a2 = rho-1
    from scipy.signal import lfilter
    x = lfilter([1.0], [1.0, -a1, -a2], rng.standard_normal(N+burn))
    return x[burn:]

def ar2_psd(rho, f, dt=1.0, s2=1.0):
    a1 = 2-(np.pi/10)**2-rho; a2 = rho-1; w = 2*np.pi*f*dt
    return s2*dt/np.abs(1 - a1*np.exp(-1j*w) - a2*np.exp(-2j*w))**2

def ar2_acf(rho, kmax):
    a1 = 2-(np.pi/10)**2-rho; a2 = rho-1
    r = np.zeros(kmax+1); r[0] = 1; r[1] = a1/(1-a2)
    for k in range(2, kmax+1): r[k] = a1*r[k-1] + a2*r[k-2]
    return r
