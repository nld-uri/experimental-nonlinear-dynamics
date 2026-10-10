"""Shared synthetic records for the Chapter 3 figures (and the ergodicity figure of
Chapter 2) -- not a listing.

ar2(rho, N, rng): the book's AR(2) benchmark
    x_{n+1} = (2 - (pi/10)^2 - rho) x_n + (rho - 1) x_{n-1} + eta_n,  eta_n ~ N(0,1),
started from x_1 = x_2 = 0 as Problem 3.2 prescribes (no burn-in; the transient is a few
tens of samples out of 40,000).
duffing_x(N, dt): x record of x'' + 0.2 x' - x + x^3 = 0.33 cos t from (x, x') =
    (-0.6973, 0.0729), sampling time 0.1 (the parameters of Fig. rxx and Problem 3.2).
lorenz_x(N, dt): x record of the Lorenz system (10, 28, 8/3) from (5, -5, 20), dt = 0.02.
acf(x, mmax): autocorrelation coefficient with the 1/(N-m) normalization.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import lfilter

W0 = 2*np.pi/20   # rad/sample; (pi/10)^2 = W0^2


def ar2(rho, N, rng):
    a1, a2 = 2 - W0**2 - rho, rho - 1
    eta = rng.standard_normal(N)
    x = lfilter([1.0], [1.0, -a1, -a2], eta)   # x_1 = x_2 = 0 initial state
    return x


def duffing_x(N=40000, dt=0.1, x0=(-0.6973, 0.0729), gamma=0.2, F=0.33):
    def f(t, y):
        return [y[1], -gamma*y[1] + y[0] - y[0]**3 + F*np.cos(t)]
    t = dt*np.arange(N)
    return solve_ivp(f, (0, t[-1]), x0, t_eval=t, rtol=1e-9, atol=1e-12).y[0]


def lorenz_x(N=40000, dt=0.02, x0=(5.0, -5.0, 20.0)):
    def f(t, y):
        return [10*(y[1]-y[0]), y[0]*(28-y[2]) - y[1], y[0]*y[1] - 8/3*y[2]]
    t = dt*np.arange(N)
    return solve_ivp(f, (0, t[-1]), x0, t_eval=t, rtol=1e-9, atol=1e-12).y[0]


def acf(x, mmax):
    x = np.asarray(x, float) - np.mean(x); N = x.size
    c = np.array([np.dot(x[:N-m], x[m:])/(N-m) for m in range(mmax+1)])
    return c/c[0]
