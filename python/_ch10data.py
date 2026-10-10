"""Shared records and nearest-neighbor results for the Chapter 10 figures (not a listing).

Records (N = 20 000 each, seed 0, normalized inside fnn so sigma_x = 1):
  Lorenz x   (10, 28, 8/3) from (5,-5,20), dt = 0.02, 50 t.u. transient discarded
  Duffing x  x'' + 0.25 x' - x + x^3 = 0.3 cos t (Section sec:runningdata), dt = 0.1,
             200 t.u. transient (the _ch02data record)
  Henon x    (a, b) = (1.4, 0.3), 1000 transient iterates discarded
  Gaussian and uniform white noise
  AR(rho)    the AR(2) process of Chapter 3, rho = 0.05 and 0.5, 200 samples discarded
Delays are the first AMI minimum (Lorenz 8, Duffing 14, AR 5; noise 1); the Theiler
window is w = 3 tau. Noisy records: x + p sigma_x eta, eta ~ N(0,1), p in per cent.

fnn results are cached in the system temp directory, keyed by every parameter, so the
six figure scripts share one computation (about 8 minutes in total on two cores).
"""
import os, tempfile, hashlib
import numpy as np
from scipy.integrate import solve_ivp
from _ch02data import duffing_record
from _ch03data import ar2
import fnn as F

N = 20000
DT = {"Lorenz": 0.02, "Duffing": 0.1}
TAU = {"Lorenz": 8, "Duffing": 14, "Henon": 1, "Gaussian": 1, "Uniform": 1,
       "AR(0.05)": 5, "AR(0.5)": 5}
WFAC = 3                                   # Theiler window w = WFAC * tau
RECORDS = ["Lorenz", "Duffing", "Gaussian", "Uniform", "AR(0.05)", "AR(0.5)"]
CACHE = os.path.join(tempfile.gettempdir(), "ch10cache")


def lorenz_x(n=N, dt=0.02, ttr=50.0, x0=(5.0, -5.0, 20.0)):
    def f(t, y):
        return [10*(y[1]-y[0]), y[0]*(28-y[2]) - y[1], y[0]*y[1] - 8/3*y[2]]
    t = ttr + dt*np.arange(n)
    return solve_ivp(f, (0, t[-1]), x0, t_eval=t, rtol=1e-9, atol=1e-12).y[0]


def henon_x(n=N, ntr=1000):
    x, y, out = 0.1, 0.1, np.empty(n + ntr)
    for i in range(n + ntr):
        x, y = 1.0 - 1.4*x*x + y, 0.3*x
        out[i] = x
    return out[ntr:]


def record(name, noise=0.0, seed=0):
    """Named record, optionally with additive Gaussian noise of noise*100 % of sigma_x."""
    rng = np.random.default_rng(seed)
    if name == "Lorenz":
        x = lorenz_x()
    elif name == "Duffing":
        x = duffing_record(N)[1]
    elif name == "Henon":
        x = henon_x()
    elif name == "Gaussian":
        x = rng.standard_normal(N)
    elif name == "Uniform":
        x = rng.random(N)
    elif name.startswith("AR("):
        x = ar2(float(name[3:-1]), N + 200, rng)[200:]
    else:
        raise ValueError(name)
    if noise > 0:
        x = x + noise*x.std()*np.random.default_rng(seed + 1).standard_normal(N)
    return x


def tau_w(name):
    tau = TAU[name]
    return tau, WFAC*tau


def nn(name, noise=0.0, dmax=20, k=1, l=0, tau=None, w=None):
    """Cached fnn.fnn for a record: (eps, dlt, dsur) lists over d = 1..dmax."""
    tau0, w0 = tau_w(name)
    tau = tau0 if tau is None else tau
    w = w0 if w is None else w
    key = "%s_%g_%d_%d_%d_%d_%d" % (name, noise, dmax, k, l, tau, w)
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, hashlib.md5(key.encode()).hexdigest() + ".npz")
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=True)
        return list(z["eps"]), list(z["dlt"]), list(z["dsur"])
    eps, dlt, dsur = F.fnn(record(name, noise), tau, dmax, w, k, l)
    np.savez(fn, eps=np.array(eps, dtype=object), dlt=np.array(dlt, dtype=object),
             dsur=np.array(dsur, dtype=object))
    return eps, dlt, dsur


# ---- plotting helpers shared by the six figure scripts ------------------------------
def family(ax, curves, sur=None, col=(0, 0.447, 0.741), scol=(0.85, 0.325, 0.098)):
    """Plot a family of fractions against d (rows of `curves`, one per threshold) with
    x markers, and the surrogate family with circles."""
    d = np.arange(1, curves.shape[1] + 1)
    for c in curves:
        ax.plot(d, c, "-x", color=col, lw=0.6, ms=3, mew=0.6)
    if sur is not None:
        for c in sur:
            ax.plot(d, c, "-o", color=scol, lw=0.6, ms=2.5, mfc="none", mew=0.6)
    ax.set_xlim(0, d[-1] + 1); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("$d$")


def sweep(eps, dlt, which, rs, ss):
    """Array of fractions (len(rs) or len(ss) rows) over d for a swept threshold."""
    rows = []
    for r in np.atleast_1d(rs):
        for s in np.atleast_1d(ss):
            rows.append(F.fnnfrac(eps, dlt, r=r, s=s)[which])
    return np.array(rows)
