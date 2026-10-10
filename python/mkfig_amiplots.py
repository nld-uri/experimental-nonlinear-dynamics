"""amiplots.pdf (label ami): average mutual information I_xx(tau), lags 0..100, from
records of N = 40,000 samples, plug-in estimator with n_b = 32 equal-width bins
(python/ami.py, the routine of Appendix A): the AR(2) processes
x_{n+1} = (2 - (pi/10)^2 - rho) x_n + (rho-1) x_{n-1} + eta_n, rho = 0.02, 0.05, 0.2, 0.5
(unit white eta, x_1 = x_2 = 0, seed 0 -- the same records as Fig. rxx of Chapter corr);
the x record of the book's Duffing oscillator x'' + 0.25 x' - x + x^3 = 0.3 cos t from
(-0.67, -0.37), dt = 0.1, 200 time units of transient discarded; and the x record of the
Lorenz system (10, 28, 8/3) from (5, -5, 20), dt = 0.02, 20 time units discarded.
(a) all six curves; (b) the AR(2) estimates (markers) against the exact Gaussian value
-1/2 log2(1 - rho_k^2) (thin lines) with rho_k from the Yule-Walker recursion.
The script also prints the first minima and the Duffing result with the Problem 3.2
parameters (0.2, 0.33) for comparison."""
import numpy as np, time
from _figstyle import *
from _ch03data import ar2, W0
from _ch08data import duffing, lorenz
from ami import ami

N, KMAX, NB = 40000, 100, 32
rng = np.random.default_rng(0)
t0 = time.time()

def firstmin(I):
    for k in range(1, len(I) - 1):
        if I[k] < I[k-1] and I[k] <= I[k+1]: return k
    return None

def rho_exact(rho, kmax):
    a1, a2 = 2 - W0**2 - rho, rho - 1
    r = np.ones(kmax + 1); r[1] = a1/(1 - a2)
    for k in range(2, kmax + 1): r[k] = a1*r[k-1] + a2*r[k-2]
    return r

curves, exact = [], []
for k, rho in enumerate((0.02, 0.05, 0.2, 0.5)):
    x = ar2(rho, N, rng)
    I = ami(x, KMAX, NB); rk = rho_exact(rho, KMAX)
    Iex = -0.5*np.log2(np.maximum(1 - rk**2, 1e-300))
    curves.append(("AR(%g)" % rho, I, COL[k], "odsv"[k])); exact.append(Iex)
    zc = np.argmax(rk < 0)
    print("AR(%g): I(0) = H = %.2f bit; first min at lag %d (exact rho_k zero crossing at lag %d); "
          "I(1) = %.3f vs exact %.3f; max |I - Iex| for k>=5: %.3f; I(100) = %.3f (exact %.3f)"
          % (rho, I[0], firstmin(I), zc, I[1], Iex[1], np.abs(I[5:] - Iex[5:]).max(), I[-1], Iex[-1]))
xd = duffing(N, 0.1, ttrans=200.0)[0]
Id = ami(xd, KMAX, NB); curves.append(("Duffing", Id, COL[4], "^"))
xl = lorenz(N, 0.02, ttrans=20.0)[0]
Il = ami(xl, KMAX, NB); curves.append(("Lorenz", Il, COL[5], "<"))
print("Duffing (0.25, 0.3): H = %.2f, first min at lag %d, I(100) = %.3f" % (Id[0], firstmin(Id), Id[-1]))
print("Lorenz: H = %.2f, first min at lag %d, I(100) = %.3f" % (Il[0], firstmin(Il), Il[-1]))
xd2 = duffing(N, 0.1, x0=(-0.6973, 0.0729), gamma=0.2, F=0.33)[0]
Id2 = ami(xd2, KMAX, NB)
print("Duffing (0.2, 0.33, Problem 3.2 record, no transient removed): H = %.2f, first min at lag %d" % (Id2[0], firstmin(Id2)))
print("run time %.1f s" % (time.time() - t0))

fig, axs = plt.subplots(1, 2, figsize=(6, 2.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
m = np.arange(KMAX + 1)
ax = axs[0]
for lab, I, c, mk in curves:
    ax.plot(m, I, "-", color=c, lw=0.8, marker=mk, ms=2.5, markevery=3, label=lab)
ax.set_xlim(0, KMAX); ax.set_ylim(0, 5.2); ax.grid(True, lw=0.3, alpha=0.5)
ax.set_xlabel(r"delay $\tau$ (samples)"); ax.set_ylabel(r"$I_{xx}(\tau)$ (bits)"); ax.legend(loc="upper right", ncol=2)
panel(ax, "a", x=-0.13)
ax = axs[1]
for (lab, I, c, mk), Iex in zip(curves[:4], exact):
    ax.plot(m[:41], Iex[:41], "-", color=c, lw=0.6)
    ax.plot(m[:41], I[:41], ls="none", color=c, marker=mk, ms=2.5, label=lab)
ax.set_xlim(0, 40); ax.set_ylim(0, 1.6); ax.grid(True, lw=0.3, alpha=0.5)
ax.set_xlabel(r"delay $\tau$ (samples)"); ax.set_ylabel(r"$I_{xx}(\tau)$ (bits)")
ax.text(0.97, 0.95, r"$-\frac{1}{2}\log_2(1-\rho_\tau^2)$", transform=ax.transAxes, ha="right", va="top", fontsize=8)
panel(ax, "b", x=-0.2)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.92, wspace=0.35)
fig.savefig(FIGDIR + "amiplots.pdf")
