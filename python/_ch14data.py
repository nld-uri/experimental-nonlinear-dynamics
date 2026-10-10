"""Shared computation for the Chapter 14 figures and for Tables TL and TD (not a listing).

One script computes the whole evaluation of Section sec:nr_eval, so that every
figure of the chapter and both signal-to-noise tables come from the same records.

Records (N = 20 000 samples each, standardized to zero mean and unit variance;
the first edition used 60 000, reduced here to keep the run to about half an hour
on two cores -- the captions and table headers say so):

  Lorenz    x of (10, 28, 8/3) from (x,y,z) = (5,-5,20), dt = 0.01,
            2000-sample transient discarded.  tau = 12 (first AMI minimum).
  Duffing   x of x'' + 0.25 x' - x + x^3 = 0.3 cos t, sampled 30 times per
            forcing period (dt = 2 pi/30), 200 time units of transient
            discarded.  tau = 7 (first AMI minimum).

Noise: additive Gaussian, standard deviation p = 0.05, 0.10, 0.20, 0.40, 0.80
of the signal's, one fixed seed per (record, level).

Filters, both for ITER = 10 iterations at d = 6, q = 3:
  POD   podfilt(x, 64, tau, 6, 3)                   64-point neighbour clouds
  SOD   sodfilt(x, 11, tau, 6, 3, w=1, r=9)         bundles of nine 11-point
                                                    strands, tent weighting

Everything is cached in NPZ files under CACHE (system temp directory), keyed by
every parameter, so the figure scripts share one computation.

Run the whole evaluation on two cores with

    python3 _ch14data.py all

or a single job with

    python3 _ch14data.py job <record> <noise> <POD|SOD>

Copyright by David Chelidze. Written for the second edition.
"""
import os
import sys
import tempfile
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from podfilt import podfilt
from sodfilt import sodfilt

N = 20000
NOISE = (0.05, 0.10, 0.20, 0.40, 0.80)
D, Q, ITER = 6, 3, 10
RPOD = 64                       # neighbours in the local projective cloud
LSOD, RSOD, WSOD = 11, 9, 1     # 11-point strands, bundle of 9, tent weights
TAU = {"Lorenz": 12, "Duffing": 7}
DT = {"Lorenz": 0.01, "Duffing": 2*np.pi/30}
SEED = {"Lorenz": 100, "Duffing": 200}
CACHE = os.path.join(tempfile.gettempdir(), "ch14cache")


# ------------------------------------------------------------------ records
def lorenz_x(n=N, dt=0.01, ntr=2000, x0=(5.0, -5.0, 20.0)):
    def f(t, s):
        return [10*(s[1] - s[0]), s[0]*(28 - s[2]) - s[1], s[0]*s[1] - 8/3*s[2]]
    t = dt*np.arange(n + ntr)
    y = solve_ivp(f, (0.0, t[-1]), x0, t_eval=t, rtol=1e-10, atol=1e-12).y[0]
    return y[ntr:]


def duffing_x(n=N, dt=2*np.pi/30, t_trans=200.0,
              x0=(-0.67079530230, -0.36933661878)):
    def f(t, y):
        return [y[1], -0.25*y[1] + y[0] - y[0]**3 + 0.3*np.cos(t)]
    t = t_trans + dt*np.arange(n)
    return solve_ivp(f, (0.0, t[-1]), x0, t_eval=t,
                     rtol=1e-10, atol=1e-12).y[0]


def clean(name):
    """Standardized clean record."""
    x = lorenz_x() if name == "Lorenz" else duffing_x()
    return (x - x.mean())/x.std()


def noisy(name, p):
    """Clean record plus Gaussian noise of standard deviation p (sigma_x = 1)."""
    rng = np.random.default_rng(SEED[name] + int(round(1000*p)))
    return clean(name) + p*rng.standard_normal(N)


def snr(xc, x):
    """Signal-to-noise ratio in dB, eq. (eq:nr_snr)."""
    return 20*np.log10(xc.std()/(x - xc).std())


# ------------------------------------------------------------------ filtering
def _fn(name, p, meth):
    return os.path.join(CACHE, "%s_%03d_%s_%d_%d_%d_%d.npz"
                        % (name, round(100*p), meth, D, Q, ITER, N))


def run(name, p, meth, force=False):
    """Filtered records after each of ITER iterations, shape (ITER, N).

    Cached; call with force=True to recompute.
    """
    os.makedirs(CACHE, exist_ok=True)
    fn = _fn(name, p, meth)
    if os.path.exists(fn) and not force:
        return np.load(fn)["xf"]
    tau = TAU[name]
    x = noisy(name, p)
    out = np.empty((ITER, N))
    for i in range(ITER):
        if meth == "POD":
            x = podfilt(x, RPOD, tau, D, Q)
        else:
            x = sodfilt(x, LSOD, tau, D, Q, w=WSOD, r=RSOD)
        out[i] = x
    np.savez_compressed(fn, xf=out)
    return out


def snr_curve(name, p, meth):
    """SNR of the noisy record and after each iteration: array of ITER+1 values."""
    xc = clean(name)
    return np.concatenate([[snr(xc, noisy(name, p))],
                           [snr(xc, r) for r in run(name, p, meth)]])


def final(name, p, meth):
    """Filtered record after ITER iterations."""
    return run(name, p, meth)[-1]


# ------------------------------------------------------- embedding statistics
def fnn_curve(name, p, dmax=10, thr=10.0, atol=2.0, which="kennel"):
    """False-nearest-neighbour fraction (per cent) for d = 1..dmax.

    fnn.fnn with the plain nearest neighbour and Theiler window w = 3 tau,
    then fnn.fnnfrac. which = "kennel" is Kennel's composite criterion
    (ratio threshold thr AND the attractor-size threshold atol), which is the
    one that shows a noise floor; "orig" is the ratio criterion alone, which
    vanishes at large d whatever the noise level. Cached.
    """
    import fnn as F
    os.makedirs(CACHE, exist_ok=True)
    tau = TAU[name]
    fn = os.path.join(CACHE, "fnn_%s_%03d_%d_%g_%g_%s_%d.npz"
                      % (name, round(100*p), dmax, thr, atol, which, N))
    if os.path.exists(fn):
        return np.load(fn)["f"]
    x = clean(name) if p == 0 else noisy(name, p)
    eps, dlt, _ = F.fnn(x, tau, dmax, 3*tau, 1, 0)
    f = 100*F.fnnfrac(eps, dlt, r=thr, a=atol)[which]
    np.savez(fn, f=f)
    return f


# ------------------------------------------------------------------ driver
if __name__ == "__main__":
    import time
    if len(sys.argv) > 1 and sys.argv[1] == "job":
        name, p, meth = sys.argv[2], float(sys.argv[3]), sys.argv[4]
        t0 = time.time()
        run(name, p, meth)
        print("%-8s %4.2f %s  %6.0f s" % (name, p, meth, time.time() - t0),
              flush=True)
    else:
        import subprocess
        jobs = [(nm, p, mt) for nm in ("Lorenz", "Duffing")
                for p in NOISE for mt in ("SOD", "POD")]
        me = os.path.abspath(__file__)
        args = ["%s job %s %g %s\n" % (me, nm, p, mt) for nm, p, mt in jobs]
        t0 = time.time()
        subprocess.run(["xargs", "-P", "2", "-L", "1", sys.executable],
                       input="".join(args), text=True, check=True)
        print("total %.0f s" % (time.time() - t0))
