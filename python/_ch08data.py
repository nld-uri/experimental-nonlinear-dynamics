"""Shared simulated records for the figures of Chapters 7-9 (not a listing).

lorenz(N, dt, x0, ttrans): full state (3 x N) of the Lorenz system (10, 28, 8/3),
    default start (5, -5, 20), dt = 0.02, first ttrans time units discarded.
duffing(N, dt, x0, gamma, F, ttrans): full state (2 x N) of the book's running two-well
    Duffing oscillator x'' + gamma x' - x + x^3 = F cos t with gamma = 0.25, F = 0.3
    (Section sec:runningdata), default start (-0.67, -0.37), transient discarded.
duffing_strobe(nper, n, gamma, F, phi, ntrans): stroboscopic section of the same
    oscillator, (x, x') once per forcing period T = 2 pi at forcing phase phi, by
    fixed-step RK4 with n steps per period (dt = 2 pi / n), ntrans periods discarded.
The AR(2) benchmark is imported from _ch03data (ar2) so that Chapters 3 and 9 use
the same records.
"""
import numpy as np
from scipy.integrate import solve_ivp


def lorenz(N=40000, dt=0.02, x0=(5.0, -5.0, 20.0), ttrans=0.0):
    def f(t, y):
        return [10*(y[1]-y[0]), y[0]*(28-y[2]) - y[1], y[0]*y[1] - 8/3*y[2]]
    t = ttrans + dt*np.arange(N)
    return solve_ivp(f, (0, t[-1]), x0, t_eval=t, rtol=1e-9, atol=1e-12).y


def duffing(N=40000, dt=0.1, x0=(-0.67, -0.37), gamma=0.25, F=0.3, ttrans=0.0):
    def f(t, y):
        return [y[1], -gamma*y[1] + y[0] - y[0]**3 + F*np.cos(t)]
    t = ttrans + dt*np.arange(N)
    return solve_ivp(f, (0, t[-1]), x0, t_eval=t, rtol=1e-9, atol=1e-12).y


def duffing_strobe(nper=5000, n=200, gamma=0.25, F=0.3, phi=0.0, ntrans=50,
                   x0=(-0.67, -0.37)):
    T = 2*np.pi; dt = T/n
    def f(t, x, v):
        return v, -gamma*v + x - x**3 + F*np.cos(t)
    x, v = float(x0[0]), float(x0[1]); t = 0.0
    out = np.empty((2, nper))
    # advance to phase phi first
    def step(t, x, v):
        k1x, k1v = f(t, x, v); k2x, k2v = f(t+dt/2, x+dt/2*k1x, v+dt/2*k1v)
        k3x, k3v = f(t+dt/2, x+dt/2*k2x, v+dt/2*k2v); k4x, k4v = f(t+dt, x+dt*k3x, v+dt*k3v)
        return x + dt/6*(k1x+2*k2x+2*k3x+k4x), v + dt/6*(k1v+2*k2v+2*k3v+k4v)
    nphi = int(round(phi/T*n))
    for _ in range(nphi):
        x, v = step(t, x, v); t += dt
    for k in range(ntrans + nper):
        if k >= ntrans:
            out[:, k-ntrans] = x, v
        for _ in range(n):
            x, v = step(t, x, v); t += dt
    return out
