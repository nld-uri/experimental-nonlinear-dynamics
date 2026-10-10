"""autocorrs.pdf (label rxx): autocorrelation coefficient rho_xx(tau), 1/(N-m)
normalization, lags 0..100, from records of N = 40000 samples: the AR(2) process
x_{n+1} = (2 - (pi/10)^2 - rho) x_n + (rho-1) x_{n-1} + eta_n for rho = 0.02, 0.05, 0.2,
0.5 (unit white eta, x_1 = x_2 = 0, seed 0); the x record of the Duffing oscillator
x'' + 0.2 x' - x + x^3 = 0.33 cos t from (-0.6973, 0.0729), dt = 0.1; and the x record of
the Lorenz system (10, 28, 8/3) from (5, -5, 20), dt = 0.02. All as in Problem 3.2."""
import numpy as np, time
from _figstyle import *
from _ch03data import ar2, duffing_x, lorenz_x, acf

N, MMAX = 40000, 100
rng = np.random.default_rng(0)
t0 = time.time()
curves = []
for k, rho in enumerate((0.02, 0.05, 0.2, 0.5)):
    r = acf(ar2(rho, N, rng), MMAX); curves.append(("AR(%g)" % rho, r, COL[k], "odsv"[k]))
    print("AR(%g): rho_1 = %.3f (exact %.3f), first zero crossing at lag %d" % (rho, r[1], (2-(np.pi/10)**2-rho)/(2-rho), np.argmax(r < 0)))
rd = acf(duffing_x(N), MMAX); curves.append(("Duffing", rd, COL[4], "^"))
rl = acf(lorenz_x(N), MMAX); curves.append(("Lorenz", rl, COL[5], "<"))
print("Duffing: first zero at lag %d, rho(100) = %.3f; Lorenz: first zero at lag %d, rho(100) = %.3f; 1/e lags: Duffing %d, Lorenz %d" %
      (np.argmax(rd < 0) if (rd < 0).any() else -1, rd[-1], np.argmax(rl < 0) if (rl < 0).any() else -1, rl[-1], np.argmax(rd < np.exp(-1)), np.argmax(rl < np.exp(-1))))
print("run time %.1f s" % (time.time()-t0))
fig, ax = plt.subplots(figsize=(6, 3.4))
m = np.arange(MMAX+1)
for lab, r, c, mk in curves:
    ax.plot(m, r, "-", color=c, lw=0.8, marker=mk, ms=2.5, markevery=2, label=lab)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(0, MMAX); ax.set_ylim(-1, 1); ax.grid(True, lw=0.3, alpha=0.5)
ax.set_xlabel(r"delay $\tau$ (samples)"); ax.set_ylabel(r"$\rho_{xx}(\tau)$"); ax.legend(loc="upper right", ncol=2)
fig.tight_layout(); fig.savefig(FIGDIR+"autocorrs.pdf")
