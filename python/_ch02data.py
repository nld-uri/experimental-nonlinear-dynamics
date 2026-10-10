"""Shared synthetic records for the Chapter 2 figures (not a listing).

The 'generic signal' of the probability figures is the x record of the book's
two-well Duffing oscillator (Section sec:runningdata), x'' + 0.25 x' - x + x^3 =
0.3 cos t, integrated with solve_ivp (RK45, rtol 1e-9) from (x, x') = (-0.67079530230,
-0.36933661878) (the initial condition of Problem 1.x), sampled at dt = 0.1 after a
transient of 200 time units. Its density is bimodal, which is what makes p(x) worth
drawing next to x(t).
"""
import numpy as np
from scipy.integrate import solve_ivp

GAMMA, F0 = 0.25, 0.3
X0 = (-0.67079530230, -0.36933661878)
DT = 0.1


def duffing_record(n=40000, dt=DT, t_trans=200.0, x0=X0):
    """x_n = x(t_trans + n dt), n = 0..n-1, of the forced two-well Duffing oscillator."""
    def f(t, y):
        return [y[1], -GAMMA*y[1] + y[0] - y[0]**3 + F0*np.cos(t)]
    t = t_trans + dt*np.arange(n)
    sol = solve_ivp(f, (0.0, t[-1]), x0, t_eval=t, rtol=1e-9, atol=1e-12, method="RK45")
    return t - t_trans, sol.y[0]


def density(x, nbins=80):
    """Histogram estimate of p(x) (unit area) and its bin centers."""
    h, e = np.histogram(x, bins=nbins, density=True)
    return 0.5*(e[1:] + e[:-1]), h, e
