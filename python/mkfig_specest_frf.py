"""specest_frf.pdf: coherence and the H1/H2 estimates for the SDOF oscillator
(f_n = 5 Hz, zeta = 0.05) driven by white noise, sampled at 100 Hz, with
(left) noise added to the measured output and (right) noise added to the
measured input. The reference curve is the exact response of the
zero-order-hold discretization used to simulate the system."""
import numpy as np
from scipy import signal
from _sestyle import *
from psdwelch import psdwelch
rng = np.random.default_rng(21)
fn, zeta, fs, m = 5.0, 0.05, 100.0, 1.0
wn = 2*np.pi*fn; k = m*wn**2; c = 2*zeta*wn*m
b, a, *_ = signal.cont2discrete(([1.0], [m, c, k]), 1/fs, method="zoh"); b, a = b.ravel(), a.ravel()
L, nd = 2048, 200
u = rng.standard_normal(L*nd); yc = signal.lfilter(b, a, u)
fr, Hd = signal.freqz(b, a, worN=L//2+1, whole=False, fs=fs)
fig, ax = plt.subplots(2, 2, figsize=(6, 4.4), sharex=True)
cases = [("output noise", u, yc + 0.5*yc.std()*rng.standard_normal(u.size)),
         ("input noise", u + 0.5*rng.standard_normal(u.size), yc)]
for j, (lab, x, y) in enumerate(cases):
    f, S = psdwelch(x, fs, L, L//2, "hann", y)
    ax[0, j].plot(f, S["coh"], color=COL[0], lw=0.9)
    ax[0, j].set_ylim(0, 1.05); ax[0, j].set_title(lab, fontsize=9)
    ax[1, j].semilogy(f, np.abs(Hd)*k, color="k", lw=1.6, alpha=0.35, label="exact $|H|$")
    ax[1, j].semilogy(f, np.abs(S["H1"])*k, color=COL[0], lw=0.9, label="$|\\hat H_1|$")
    ax[1, j].semilogy(f, np.abs(S["H2"])*k, color=COL[1], lw=0.9, ls="--", label="$|\\hat H_2|$")
    ax[1, j].set_xlabel("$f$ (Hz)"); ax[1, j].set_ylim(3e-2, 30)
    for i5 in [np.argmin(np.abs(f-5.0)), np.argmin(np.abs(f-2.0)), np.argmin(np.abs(f-12.0))]:
        print(lab, f"f={f[i5]:.2f}: coh={S['coh'][i5]:.3f} |H1|/|H|={np.abs(S['H1'][i5])/np.abs(Hd[i5]):.3f} |H2|/|H|={np.abs(S['H2'][i5])/np.abs(Hd[i5]):.3f}")
ax[0, 0].set_ylabel("$\\hat\\gamma^2_{xy}(f)$"); ax[1, 0].set_ylabel("$k\\,|\\hat H(f)|$")
ax[1, 0].legend(loc="upper right"); ax[1, 1].legend(loc="upper right")
for a in ax.ravel(): a.set_xlim(0, 15)
for a, t in zip(ax.ravel(), ["(a)", "(b)", "(c)", "(d)"]): a.text(0.03, 0.08, t, transform=a.transAxes)
fig.tight_layout(); fig.savefig(FIGDIR+"specest_frf.pdf"); print("K =", S["K"])
