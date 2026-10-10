"""bundle_exp.py -- the strand-bundle study of Chapter 14 (Table TB).

Lorenz x channel, 20,000 points at dt = 0.01 after a 2000-sample
transient, standardized, with 20 % and 40 % Gaussian noise added.
Filter: sodfilt with tau = 12, d = 6, eleven-point strands, q = 3, tent
weighting, three iterations, bundle size r = 1, 2, 4, 8. The bundle
weight is w(delta) = a^4/(delta^4 + a^4), eq. (loclinw) of Chapter 15,
with a the distance to the farthest strand of each bundle.

Three quantities per record, each through a single window taken from
the clean record's scaling region so that records are comparable:

    SNR       eq. (eq:nr_snr) against the clean record
    D2        slope of ln C(eps) over ln eps in [-3.05, -1.58]
              (c2r, d = 3, tau = 12, whole record)
    lambda_1  slope of Delta_n over steps 236-275, divided by dt
              (lyapunov, d = 3, tau = 12, N = 4000 fiducial points, r = 1)

The last block repeats r = 4 with the stack-and-project reading of the
bundle (sodbundle.py, bundle="subspace"), which the chapter mentions as
recovering considerably less. Runtime is about ten minutes.

    python3 bundle_exp.py
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from sodfilt import sodfilt
from sodbundle import sodfilt_bundle
from c2r import c2r
from lyapunov import lyapunov

L, M, D, Q, ITER = 11, 12, 6, 3, 3
RS = (1, 2, 4, 8)


def lorenz(npts=20000, dt=0.01):
    f = lambda t, s: [10*(s[1]-s[0]), s[0]*(28-s[2])-s[1], s[0]*s[1]-8/3*s[2]]
    T = (npts + 2000)*dt
    S = solve_ivp(f, (0, T), [5, -5, 20], t_eval=np.arange(0, T, dt),
                  rtol=1e-10, atol=1e-12).y
    x = S[0, 2000:2000+npts]
    return (x - x.mean())/x.std()


def snr(c, e):
    return 20*np.log10(c.std()/(e - c).std())


def d2(x):
    lnr, lnc = c2r(x, 3, 12, r=60, rng=np.random.default_rng(0))
    k = (lnr >= -3.05) & (lnr <= -1.58) & np.isfinite(lnc)
    return np.polyfit(lnr[k], lnc[k], 1)[0]


def lam1(x, dt=0.01):
    ns, lnd = lyapunov(x, 3, 12, 4000, 1, 300, rng=np.random.default_rng(0))
    k = (ns >= 236) & (ns <= 275)
    return np.polyfit(ns[k], lnd[k], 1)[0]/dt


def filt(xn, r, mode=None):
    xf = xn.copy()
    for _ in range(ITER):
        if mode is None:
            xf = sodfilt(xf, L, M, D, Q, w=1, r=r)
        else:
            xf = sodfilt_bundle(xf, L, M, D, Q, w=1, r=r, bundle=mode)
    return xf


if __name__ == "__main__":
    xc = lorenz()
    print(f"Lorenz, {xc.size} points, l={L}, m={M}, d={D}, q={Q}, tent "
          f"weight, {ITER} iterations")
    print(f"clean record:  D2 = {d2(xc):.3f}   lambda_1 = {lam1(xc):.3f}\n")
    print(f"{'':28s} {'SNR (dB)':>9s} {'D2':>7s} {'lambda_1':>9s} {'time':>7s}")
    rng = np.random.default_rng(7)
    noisy = {}
    for frac in (0.2, 0.4):
        xn = xc + frac*rng.standard_normal(xc.size)
        noisy[frac] = xn
        print(f"{int(frac*100):d} % noise, unfiltered      "
              f"{snr(xc, xn):9.2f} {d2(xn):7.3f} {lam1(xn):9.3f}")
        for r in RS:
            t0 = time.time()
            xf = filt(xn, r)
            t = time.time() - t0                  # the filter alone
            print(f"    r = {r:<22d}{snr(xc, xf):9.2f} {d2(xf):7.3f} "
                  f"{lam1(xf):9.3f} {t:6.0f}s", flush=True)
        print()
    print("stack-and-project reading of the bundle (sodbundle, subspace):")
    for frac, xn in noisy.items():
        xf = filt(xn, 4, mode="subspace")
        print(f"{int(frac*100):d} % noise, r = 4          "
              f"{snr(xc, xf):9.2f} {d2(xf):7.3f} {lam1(xf):9.3f}", flush=True)
