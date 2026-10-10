"""Shared records and cached correlation sums for the Chapter 11 figures (not a listing).

Records (N = 20 000 each, the Chapter 10 records, Table 10.1):
  Lorenz x   (10, 28, 8/3) from (5, -5, 20), dt = 0.02, 50 t.u. transient discarded
  Duffing x  x'' + 0.25 x' - x + x^3 = 0.3 cos t (Section sec:runningdata), dt = 0.1,
             200 t.u. transient (the _ch02data record)
Delays are the first AMI minimum (Lorenz tau = 8, Duffing tau = 14) and the Theiler
window is w = 3 tau, as in Chapter 10. c2r normalizes every record to unit variance,
so ln eps is in units of the standard deviation of the record. Noisy records:
x + p sigma_x eta, eta ~ N(0,1), seed 1.

c2r and pwdim results are cached in the system temp directory, keyed by every
parameter, so the figure scripts share one computation.
"""
import os, tempfile, hashlib
import numpy as np
from _ch08data import lorenz
from _ch02data import duffing_record
from c2r import c2r
from pwdim import pwdim

N = 20000
DT = {"Lorenz": 0.02, "Duffing": 0.1}
TAU = {"Lorenz": 8, "Duffing": 14}
WFAC = 3                                   # Theiler window w = WFAC * tau
DEMB = {"Lorenz": 3, "Duffing": 4}         # embedding dimension used for D2 (Ch. 10)
LNEPS = np.linspace(-8.0, 2.0, 101)        # ln of the ball sizes handed to c2r
CACHE = os.path.join(tempfile.gettempdir(), "ch11cache")


def record(name, noise=0.0, seed=1):
    if name == "Lorenz":
        x = lorenz(N, dt=DT[name], ttrans=50.0)[0]
    elif name == "Duffing":
        x = duffing_record(N)[1]
    else:
        raise ValueError(name)
    if noise > 0:
        x = x + noise*x.std()*np.random.default_rng(seed).standard_normal(N)
    return x


def tau_w(name):
    tau = TAU[name]
    return tau, WFAC*tau


def _cached(key, fn):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, hashlib.md5(key.encode()).hexdigest() + ".npz")
    if os.path.exists(f):
        z = np.load(f)
        return [z["a%d" % i] for i in range(len(z.files))]
    out = fn()
    np.savez(f, **{"a%d" % i: a for i, a in enumerate(out)})
    return list(out)


def corrsum(name, d, noise=0.0, tau=None, w=None, lneps=LNEPS):
    """Cached (lnr, lnc) from c2r for one record and embedding dimension."""
    tau0, w0 = tau_w(name)
    tau = tau0 if tau is None else tau
    w = w0 if w is None else w
    key = "c2r_%s_%d_%g_%d_%d_%g_%g_%d" % (name, d, noise, tau, w, lneps[0], lneps[-1],
                                          lneps.size)
    return _cached(key, lambda: c2r(record(name, noise), d, tau, r=np.exp(lneps), p=w))


def pointwise(name, d, k=500, tau=None, w=None, Nq=4096):
    """Cached (lnD, lnN) from pwdim (fixed distance in column 0, fixed mass in column 1),
    on the record normalized to unit variance so its scales match c2r."""
    tau0, w0 = tau_w(name)
    tau = tau0 if tau is None else tau
    w = w0 if w is None else w
    key = "pwdim_%s_%d_%d_%d_%d_%d" % (name, d, k, tau, w, Nq)
    x = record(name)
    x = (x - x.mean())/x.std()
    return _cached(key, lambda: pwdim(x, tau, d, k=k, N=Nq, w=w,
                                      rng=np.random.default_rng(0)))


def local_slope(lnr, lnc):
    """d ln C / d ln eps by central differences on the finite part of the curve."""
    ok = np.isfinite(lnc)
    r, c = lnr[ok], lnc[ok]
    s = np.gradient(c, r)
    return r, s


def fit_range(lnr, lnc, lo, hi):
    """Least-squares slope of lnc on lnr over lo <= lnr <= hi, its standard error,
    and the mean and standard deviation of the local slopes in the range."""
    ok = np.isfinite(lnc) & (lnr >= lo) & (lnr <= hi)
    r, c = lnr[ok], lnc[ok]
    A = np.column_stack([r, np.ones_like(r)])
    coef, res, _, _ = np.linalg.lstsq(A, c, rcond=None)
    n = r.size
    resid = c - A @ coef
    s2 = resid @ resid/(n - 2)
    se = np.sqrt(s2/np.sum((r - r.mean())**2))
    rs, sl = local_slope(lnr, lnc)
    m = (rs >= lo) & (rs <= hi)
    return coef[0], se, sl[m].mean(), sl[m].std()


def takens(name, d, eps0, noise=0.0, tau=None, w=None, seed=0, nq=3000):
    """Takens' estimator eq. (eq:takensD): all pair distances below eps0 between nq
    random reference points (Theiler window w) and the whole record, normalized to
    unit variance. Returns (D_T, formal error D_T/sqrt(M), M)."""
    from scipy.spatial import cKDTree
    tau0, w0 = tau_w(name)
    tau = tau0 if tau is None else tau
    w = w0 if w is None else w
    x = record(name, noise)
    x = (x - x.mean())/x.std()
    n = x.size - (d - 1)*tau
    y = np.array([x[np.arange(n) + i*tau] for i in range(d)]).T
    tree = cKDTree(y)
    rng = np.random.default_rng(seed)
    ref = rng.permutation(n)[:nq]
    logs = []
    for i in ref:
        idx = tree.query_ball_point(y[i], eps0)
        idx = np.array([j for j in idx if abs(j - i) > w])
        if idx.size:
            r = np.sqrt(np.sum((y[idx] - y[i])**2, axis=1))
            r = r[r > 0]
            logs.append(np.log(eps0/r))
    logs = np.concatenate(logs)
    M = logs.size
    D = M/logs.sum()
    return D, D/np.sqrt(M), M
