"""False nearest neighbors, composite fraction, and Cao's statistics.

Python twin of fnn.m (Appendix A). Every statistic of Chapter `embdim`
from one nearest-neighbor search per embedding dimension:

  fnn(x, tau, dmax, w, k, l)  -> eps, dlt, dsur   (per-pair separations)
  fnnfrac(eps, dlt, r, s, a)  -> the five fractions f_nn, f_hat, f_tld,
                                 f_bar and R_nn against d
  cao(x, tau, dmax, w)        -> Cao's E1(d), E2(d)

The record is normalized to zero mean and unit variance, so sigma_x = 1
and every distance is in units of sigma_x. Delay vectors are forward,
x_i = [x_i, x_{i+tau}, ..., x_{i+(d-1)tau}], and only the N - d tau
points whose (d+1)-th coordinate x_{i+d tau} exists are used.

Copyright by David Chelidze. Python version.
"""
import numpy as np
from scipy.spatial import cKDTree


def embed(x, d, tau, n):
    """n forward delay vectors of dimension d as rows."""
    return np.array([x[np.arange(n) + q*tau] for q in range(d)]).T


def nnsearch(Y, w, k, p=2.0):
    """k nearest neighbors of every row of Y outside the Theiler window w,
    eq. (rnns). Returns index array J (n x k) and distance array D."""
    n = Y.shape[0]
    tree = cKDTree(Y)
    i = np.arange(n)
    K = min(n, k + 2*w + 1)
    D, J = tree.query(Y, k=K, p=p, workers=-1)
    Dk, Jk = np.empty((n, k)), np.empty((n, k), int)
    todo = i
    while todo.size:
        ok = np.abs(J - todo[:, None]) > w
        cnt = np.cumsum(ok, axis=1)
        done = cnt[:, -1] >= k
        sel = ok[done] & (cnt[done] <= k)      # exactly k per finished row
        Dk[todo[done]] = D[done][sel].reshape(-1, k)
        Jk[todo[done]] = J[done][sel].reshape(-1, k)
        todo = todo[~done]
        if todo.size:                          # too many close-in-time points
            K = min(n, 2*K)
            D, J = tree.query(Y[todo], k=K, p=p, workers=-1)
    return Jk, Dk


def fnn(x, tau, dmax, w=0, k=1, l=0, rng=None):
    """Nearest-neighbor separations for d = 1..dmax.

    x    : scalar record;  tau : delay;  dmax : largest dimension
    w    : Theiler window in samples, eq. (rnns); 0 keeps all pairs
    k, l : number of candidate neighbors and strand half-length of the
           composite-fraction selection, eq. (CFrac); k=1, l=0 is the
           plain nearest neighbor
    Returns three lists indexed by d-1: eps[d-1] the d-dimensional
    separations eps_i, dlt[d-1] the appended separations delta_i, and
    dsur[d-1] the surrogate delta_i with the neighbor's appended
    coordinate drawn at random from the record (Section sec:ed_records).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x, float).ravel()
    x = (x - x.mean())/x.std()
    N = x.size
    eps, dlt, dsur = [], [], []
    for d in range(1, dmax + 1):
        n = N - d*tau
        Y = embed(x, d, tau, n)
        J, D = nnsearch(Y, w, k)
        if k > 1 and l > 0:                     # strand selection, eq. (CFrac)
            i = np.arange(n)
            S = np.zeros((n, k))
            for q in range(-l, l + 1):
                iq = np.clip(i + q, 0, n - 1)
                jq = np.clip(J + q, 0, n - 1)
                S += np.linalg.norm(Y[jq] - Y[iq][:, None, :], axis=2)
            m = np.argmin(S/np.maximum(D, 1e-300), axis=1)
            j, e = J[i, m], D[i, m]
        else:
            j, e = J[:, 0], D[:, 0]
        i = np.arange(n)
        eps.append(e)
        dlt.append(np.abs(x[i + d*tau] - x[j + d*tau]))
        dsur.append(np.abs(x[i + d*tau] - x[rng.integers(0, N, n)]))
    return eps, dlt, dsur


def fnnfrac(eps, dlt, r=10.0, s=0.5, a=2.0):
    """The fractions of Chapter `embdim` for thresholds r, s, a (sigma_x = 1).

    Returns a dict of arrays over d: 'orig' f_nn (fnnFrac), 'kennel' f_hat
    (OrKennel), 'tisean' f_tld (tisean), 'rel' f_bar (K_Rel) and 'comp'
    R_nn (new). The TISEAN fraction is zero where it has no admissible pair.
    """
    out = {q: np.zeros(len(eps)) for q in ("orig", "kennel", "tisean", "rel", "comp")}
    for m, (e, dl) in enumerate(zip(eps, dlt)):
        ratio = dl > r*e
        out["orig"][m] = ratio.mean()
        out["kennel"][m] = (ratio | (dl**2 + e**2 > a**2)).mean()
        adm = r*e < 1.0
        out["tisean"][m] = (ratio & adm).sum()/max(adm.sum(), 1)
        out["rel"][m] = (dl > s).mean()
        out["comp"][m] = (ratio | (dl > s)).mean()
    return out


def cao(x, tau, dmax, w=0):
    """Cao's E1(d) and E2(d), eqs. (eq:ed_caoE1)-(eq:ed_caoE2), d = 1..dmax,
    maximum norm, nearest neighbor outside the Theiler window w."""
    x = np.asarray(x, float).ravel()
    x = (x - x.mean())/x.std()
    N = x.size
    E, Es = np.zeros(dmax + 1), np.zeros(dmax + 1)
    for d in range(1, dmax + 2):
        n = N - d*tau
        Y = embed(x, d, tau, n)
        J, D = nnsearch(Y, w, 1, p=np.inf)
        j, e = J[:, 0], D[:, 0]
        i = np.arange(n)
        dl = np.abs(x[i + d*tau] - x[j + d*tau])
        ok = e > 0
        E[d - 1] = np.mean(np.maximum(e[ok], dl[ok])/e[ok])
        Es[d - 1] = dl.mean()
    return E[1:]/E[:-1], Es[1:]/Es[:-1]
