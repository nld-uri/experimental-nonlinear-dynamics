"""Which step of the IAAFT iteration places the badly predictable samples?

Section sec:surr_robust reports that a robust prediction-error statistic
rejects a true null against IAAFT surrogates, because the surrogates' local
linear prediction errors are far more heavy-tailed than the data's. The text
records that both branches of `irsurr` carry the excess kurtosis, so the final
rank-ordering is not by itself the cause, and leaves the mechanism open.

This script separates the two steps of the iteration

    sb = ifft(|S| exp(i arg fft(rb)))     # spectral step, spectrum exact
    rb[argsort(sb)] = sort(data)          # rank step, histogram exact

by measuring the per-sample prediction error of each intermediate, and by
asking WHERE in the amplitude distribution the bad samples sit.

Run from python/. About two minutes.
"""
import numpy as np
from scipy.spatial import cKDTree
import _ch16data as C

D, TAU, K, TW = 3, 1, 8, 20


def perr(y, d=D, tau=TAU, k=K, tw=TW):
    """Per-sample one-step prediction error, normalized by sd(y)."""
    y = np.asarray(y, float).ravel()
    Y = C.embed(y, d, tau)
    M = Y.shape[0] - 1
    tree = cKDTree(Y[:M])
    _, idx = tree.query(Y[:M], k=k + 2*tw + 1)
    e = np.empty(M)
    for i in range(M):
        j = idx[i][np.abs(idx[i] - i) > tw][:k]
        Xn = np.hstack([Y[j], np.ones((j.size, 1))])
        coef, *_ = np.linalg.lstsq(Xn, Y[j + 1, -1], rcond=None)
        e[i] = np.append(Y[i], 1.0) @ coef - Y[i + 1, -1]
    return e/np.std(y)


def kurt(e):
    e = e - e.mean()
    return float(np.mean(e**4)/np.mean(e**2)**2)


def iaaft_trace(s, n, rng):
    """Yield (label, series) after each step of the iteration."""
    s = np.asarray(s, float).ravel()
    N = s.size
    c = np.sort(s)
    S = np.abs(np.fft.fft(s))
    rb = np.real(np.fft.ifft(np.fft.fft(s)*np.exp(1j*2*np.pi*rng.random(N))))
    yield "0 phase-randomized start", rb.copy()
    for it in range(n):
        sb = np.real(np.fft.ifft(S*np.exp(1j*np.angle(np.fft.fft(rb)))))
        rb = rb.copy()
        rb[np.argsort(sb)] = c
        if it in (0, 1, 4, n - 1):
            yield "%d after spectral step" % (it + 1), sb.copy()
            yield "%d after rank step" % (it + 1), rb.copy()


def by_amplitude(y, e, nb=5):
    """Mean squared prediction error by quintile of |y|, to locate the damage."""
    y = np.asarray(y, float).ravel()[:len(e)]
    q = np.argsort(np.argsort(np.abs(y)))*nb//len(y)
    return [float(np.mean(e[q == b]**2)) for b in range(nb)]


def main():
    rng = np.random.default_rng(7)
    _, y = C.ar1_cubed()                   # s = x^3; the null HOLDS for s
    print("Record: AR(1) cubed, N = %d. The null of a static transform of a "
          "linear Gaussian process is TRUE here.\n" % len(y))
    e = perr(y)
    print("%-28s rms %.4f  kurtosis %6.2f" % ("data", np.sqrt(np.mean(e**2)), kurt(e)))
    print("  mean squared error by |y| quintile:",
          " ".join("%.4f" % v for v in by_amplitude(y, e)))
    print()
    for lab, z in iaaft_trace(y, 20, rng):
        e = perr(z)
        print("%-28s rms %.4f  kurtosis %6.2f" % (lab, np.sqrt(np.mean(e**2)), kurt(e)))
        print("  mean squared error by |y| quintile:",
              " ".join("%.4f" % v for v in by_amplitude(z, e)))
    print()
    ft = C.ft_surr(y, rng)
    e = perr(ft)
    print("%-28s rms %.4f  kurtosis %6.2f" % ("FT surrogate", np.sqrt(np.mean(e**2)), kurt(e)))
    print("  mean squared error by |y| quintile:",
          " ".join("%.4f" % v for v in by_amplitude(ft, e)))


if __name__ == "__main__":
    main()
