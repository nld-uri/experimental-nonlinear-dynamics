"""Shared data and statistics for the Chapter 16 (surrogate) figures.

One module so that mkfig_e15_1.py, mkfig_e15_3.py, mkfig_e15_5.py,
mkfig_e15_7.py, mkfig_e15_8.py and mkfig_e15_9.py all work on exactly the
same records and the same surrogate conventions as mkfig_surr_trev.py.

Records
-------
ar1_cubed()  : s_n = x_n^3 with x_n = 0.99 x_{n-1} + eta_n, N = 4096,
               eta_n ~ N(0,1), 2000 transient points discarded, mean of x
               removed before cubing (Section 16.1.1).  Same seed and same
               construction as python/mkfig_surr_trev.py.
lorenz_x()   : x(t) of the Lorenz system (sigma=10, rho=28, beta=8/3),
               sampled at dt = 0.03, N = 4096 after transient.

Surrogates
----------
shuffle_surr : random permutation (null hypothesis one)
ft_surr      : phase randomization (null hypothesis two), identical to the
               generator in mkfig_surr_trev.py
irsurr       : iteratively refined (null hypothesis three), from irsurr.py

Statistics
----------
lag1         : ordinary (non-circular) sample autocorrelation at lag one
lag1_circ    : circular autocorrelation at lag one, the quantity the FT
               surrogates preserve exactly
prederr      : normalized RMS one-step-ahead local-linear prediction error,
               out of sample (first half models, second half predicted)

Copyright by David Chelidze.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

N_DEFAULT = 4096
A_AR = 0.99
SEED = 2016            # the seed of mkfig_surr_trev.py


def ar1_cubed(N=N_DEFAULT, a=A_AR, seed=SEED, burn=2000):
    """Return (x, s) with s = x**3; x is the mean-removed AR(1) record."""
    rng = np.random.default_rng(seed)
    x = np.empty(N + burn)
    x[0] = 0.0
    eta = rng.standard_normal(N + burn)
    for n in range(1, N + burn):
        x[n] = a*x[n-1] + eta[n]
    x = x[burn:]
    x = x - x.mean()
    return x, x**3


def lorenz_x(N=N_DEFAULT, dt=0.03, sigma=10.0, rho=28.0, beta=8.0/3.0,
             burn=200.0, seed=7):
    """Return the scalar x(t) record of the Lorenz system, mean removed."""
    def f(t, u):
        x, y, z = u
        return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]
    rng = np.random.default_rng(seed)
    u0 = np.array([1.0, 1.0, 20.0]) + 0.01*rng.standard_normal(3)
    sol0 = solve_ivp(f, (0.0, burn), u0, rtol=1e-10, atol=1e-12,
                     dense_output=True)
    t = np.arange(N)*dt
    sol = solve_ivp(f, (0.0, t[-1] + dt), sol0.y[:, -1], t_eval=t,
                    rtol=1e-10, atol=1e-12)
    x = sol.y[0]
    return x - x.mean()


# ----------------------------------------------------------------- surrogates
def shuffle_surr(y, rng):
    """Unconstrained randomization: a random permutation of the record."""
    return rng.permutation(y)


def ft_surr(y, rng):
    """Phase-randomized (FT) surrogate; conjugate symmetry enforced."""
    Y = np.fft.rfft(y)
    ph = rng.random(Y.size)*2*np.pi
    ph[0] = 0.0
    if y.size % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(Y)*np.exp(1j*ph), n=y.size)


# ---------------------------------------------------------------- statistics
def lag1(y):
    """Ordinary sample autocorrelation at lag one."""
    y = np.asarray(y, float) - np.mean(y)
    return float(np.sum(y[:-1]*y[1:])/np.sum(y*y))


def lag1_circ(y):
    """Circular autocorrelation at lag one (what phase randomization keeps)."""
    y = np.asarray(y, float) - np.mean(y)
    return float((np.sum(y[:-1]*y[1:]) + y[-1]*y[0])/np.sum(y*y))


def embed(y, d, tau):
    """Forward delay vectors, shape (M, d)."""
    y = np.asarray(y, float).ravel()
    M = y.size - (d - 1)*tau
    return np.stack([y[j*tau:j*tau + M] for j in range(d)], axis=1)


def prederr(y, d=3, tau=1, k=8, tw=20):
    """Normalized RMS one-step-ahead local-linear prediction error.

    Every delay vector is predicted one step ahead from a linear model fitted
    to its k nearest neighbours, with neighbours closer than tw in time
    excluded (Theiler window), so the prediction is out of sample in the sense
    that matters for a correlated record.  The RMS error is divided by the
    standard deviation of the record, so it is dimensionless and comparable
    between the data and its surrogates.
    """
    y = np.asarray(y, float).ravel()
    Y = embed(y, d, tau)
    M = Y.shape[0] - 1                       # last vector has no successor
    tree = cKDTree(Y[:M])
    _, idx = tree.query(Y[:M], k=k + 2*tw + 1)
    err = np.empty(M)
    for i in range(M):
        j = idx[i][np.abs(idx[i] - i) > tw][:k]
        Xn = np.hstack([Y[j], np.ones((j.size, 1))])            # (k, d+1)
        z = np.append(Y[i], 1.0)
        coef, *_ = np.linalg.lstsq(Xn, Y[j + 1, -1], rcond=None)
        err[i] = z @ coef - Y[i + 1, -1]
    return float(np.sqrt(np.mean(err**2))/np.std(y))


# ------------------------------------------------- the Lorenz example record
LOR_NOISE = 0.05        # measurement noise, fraction of the record's RMS
LOR_D, LOR_TAU = 3, 6   # embedding: d = 3, tau = 6 samples (first AMI minimum)


def lorenz_record(N=N_DEFAULT, dt=0.03, noise=LOR_NOISE, seed=101):
    """Return s_n = x_n + eta_n, the noisy Lorenz observation of Section 16.1.2."""
    x = lorenz_x(N=N, dt=dt)
    rng = np.random.default_rng(seed)
    return x + noise*x.std()*rng.standard_normal(x.size)
