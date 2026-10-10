"""specest_acfband.pdf: sample autocorrelation of white noise and of the AR(2)
process (rho = 0.05), N = 1000, with the +-2/sqrt(N) white-noise band and,
for the AR(2), the Bartlett large-lag band."""
import numpy as np
from _sestyle import *
rng = np.random.default_rng(5)
N, kmax = 1000, 120
def acf(x, kmax):
    x = x - x.mean(); F = np.fft.fft(x, 2*N)
    c = np.fft.ifft(np.abs(F)**2).real[:kmax+1]/N
    return c/c[0]
w = rng.standard_normal(N); x = ar2(0.05, N, rng)
k = np.arange(kmax+1); r = ar2_acf(0.05, 3000)
bart = np.sqrt(np.sum(np.concatenate([r[1:][::-1], r])**2)/N)
fig, ax = plt.subplots(1, 2, figsize=(6, 2.5))
for a, s, lab in zip(ax, [w, x], ["(a) white noise", "(b) AR(2), $\\rho=0.05$"]):
    rh = acf(s, kmax)
    a.axhspan(-2/np.sqrt(N), 2/np.sqrt(N), color=COL[0], alpha=0.18, lw=0, label="$\\pm 2/\\sqrt{N}$")
    a.plot(k, rh, color=COL[0], lw=0.9, label="$\\hat\\rho_k$")
    a.axhline(0, color="k", lw=0.4); a.set_xlabel("lag $k$"); a.set_xlim(0, kmax)
    a.text(0.03, 0.9, lab, transform=a.transAxes)
ax[0].set_ylim(-0.15, 0.15); ax[0].set_ylabel("$\\hat\\rho_{xx}(k)$")
ax[1].plot(k, r[:kmax+1], color=COL[1], lw=0.9, ls="--", label="exact $\\rho_k$")
ax[1].axhline(2*bart, color=COL[3], lw=0.8, ls=":"); ax[1].axhline(-2*bart, color=COL[3], lw=0.8, ls=":", label="$\\pm 2\\sqrt{\\sum_j \\rho_j^2/N}$")
ax[1].set_ylim(-1.05, 1.75)
ax[0].legend(loc="upper right"); ax[1].legend(loc="upper right")
fig.tight_layout(); fig.savefig(FIGDIR+"specest_acfband.pdf")
print("Bartlett band:", 2*bart, " white band:", 2/np.sqrt(N), " sum rho^2:", np.sum(np.concatenate([r[1:][::-1], r])**2))
