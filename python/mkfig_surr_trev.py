"""Figure surr_trev.pdf (Chapter 16): surrogate tests on the AR(1)-cubed record
s_n = x_n^3, x_n = 0.99 x_{n-1} + eta_n, with statistics that are NOT preserved
by the surrogates.  (a) time-reversal asymmetry vs 99 FT surrogates;
(b) excess kurtosis vs 99 FT surrogates; (c) time-reversal asymmetry vs 99 IAAFT
surrogates (histogram exactly preserved).  Prints the numbers quoted in the text."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from irsurr import irsurr
COL = [(0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.929, 0.694, 0.125), (0.494, 0.184, 0.556)]
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
                     "xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 0.6})
rng = np.random.default_rng(2016)
N, a = 4096, 0.99
x = np.empty(N + 2000); x[0] = 0.0
eta = rng.standard_normal(N + 2000)
for n in range(1, N + 2000):
    x[n] = a*x[n-1] + eta[n]
x = x[2000:]; x -= x.mean(); s = x**3

def trev(y):                       # time-reversal asymmetry, normalised
    d = y[1:] - y[:-1]
    return np.mean(d**3)/np.mean(d**2)**1.5

def ft_surr(y):
    Y = np.fft.rfft(y); ph = rng.random(Y.size)*2*np.pi; ph[0] = 0
    if y.size % 2 == 0: ph[-1] = 0
    return np.fft.irfft(np.abs(Y)*np.exp(1j*ph), n=y.size)

B = 99
ft = np.array([ft_surr(s) for _ in range(B)])
ia = np.array([irsurr(s, 60, rng)[0] for _ in range(B)])
stats = {
 "trev_ft":  (trev(s), np.array([trev(u) for u in ft])),
 "kurt_ft":  (kurtosis(s), np.array([kurtosis(u) for u in ft])),
 "trev_ia":  (trev(s), np.array([trev(u) for u in ia])),
 "kurt_ia":  (kurtosis(s), np.array([kurtosis(u) for u in ia])),
}
for k, (d, sur) in stats.items():
    rank = np.sum(sur < d) + 1
    print(f"{k}: data {d:.4f}; surrogates min {sur.min():.4f} max {sur.max():.4f} "
          f"mean {sur.mean():.4f} sd {sur.std():.4f}; rank of data {rank}/{B+1}")
# spectral check of IAAFT convergence
P = lambda y: np.abs(np.fft.rfft(y))**2
print("IAAFT relative PSD mismatch (rms over bins, log10):",
      np.sqrt(np.mean((np.log10(P(ia[0])[1:]) - np.log10(P(s)[1:]))**2)))
# repeated-trial false rejection rate of trev/FT (two-sided, alpha=0.02 with 99 surr -> reject if rank 1 or 100)
rej = 0; T = 100
for t in range(T):
    xx = np.empty(N + 2000); xx[0] = 0; ee = rng.standard_normal(N + 2000)
    for n in range(1, N + 2000): xx[n] = a*xx[n-1] + ee[n]
    ss = (xx[2000:] - xx[2000:].mean())**3
    d = trev(ss); sur = np.array([trev(ft_surr(ss)) for _ in range(B)])
    r = np.sum(sur < d) + 1
    rej += (r == 1) or (r == B + 1)
print(f"trev/FT two-sided rejection rate over {T} independent records: {rej/T:.2f} (nominal 0.02)")

fig, ax = plt.subplots(1, 3, figsize=(6, 2.0))
panels = [("trev_ft", r"$T_{\rm rev}$, FT surrogates", "(a)"),
          ("kurt_ft", r"excess kurtosis $\gamma_2$, FT surrogates", "(b)"),
          ("trev_ia", r"$T_{\rm rev}$, IAAFT surrogates", "(c)")]
for axi, (k, lab, tag) in zip(ax, panels):
    d, sur = stats[k]
    axi.hist(sur, bins=15, color=COL[0], alpha=0.75, edgecolor="none", label="99 surrogates")
    axi.axvline(d, color=COL[1], lw=1.5, label="data")
    axi.set_xlabel(lab); axi.set_yticks([])
    axi.text(0.02, 0.92, tag, transform=axi.transAxes)
ax[0].set_ylabel("count"); ax[0].legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig("../figures/surr_trev.pdf")
print("wrote figures/surr_trev.pdf")
