"""psw_farfield.pdf -- the far-field test on the measured record
(Chapter 17, figure pswfar).

(a) healthy-stage SOVs for three feature sets: the mixed reference points of
    Section sec:mt_alg, a shell of points at eight bounding-box half-widths
    where eq:farfield certainly applies, and the 20 moment features
(b) the measured voltage portrait and the one recovered from the far-field
    shell alone
(c) the normalized damage SOC from each of the three

The numbers are computed by chk_farfield_exp.py, which this script imports and
re-runs; about four minutes.
"""
import os
import numpy as np
from _figstyle import *
from _psw2005 import tracking, raw, raw_path, NHEALTHY
import chk_farfield_exp as X
from scipy.io import loadmat

SMALL = 8


def main():
    Y, t, v1, v2 = tracking()
    x = X.raw()
    W, M = X.windows(x)
    pool, mid, half = X.box(W, M)
    rng = np.random.default_rng(X.SEED)
    ref = mid + (2*rng.random((X.L, X.DIM)) - 1)*half*X.ENLARGE
    u = rng.normal(size=(X.L, X.DIM))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    reff = mid + u*half*X.FAR

    cache = "/tmp/psw_ff_cache.npz"           # the three feature matrices
    if os.path.exists(cache):
        z = np.load(cache)
        Fa, Ff, Fm = z["Fa"], z["Ff"], z["Fm"]
    else:
        Fa, Ff, Fm = (X.chardist(W, M, ref), X.chardist(W, M, reff),
                      X.moments(W, M))
        np.savez_compressed(cache, Fa=Fa, Ff=Ff, Fm=Fm)
    sets = [("all", Fa, COL[0]), ("far field", Ff, COL[1]),
            ("moments", Fm, COL[3])]
    res = {n: X.stages(F, v1, v2, t) for n, F, _ in sets}

    fig, ax = plt.subplots(1, 3, figsize=(6.0, 2.1))
    for (n, F, c) in sets:
        _, Psi, lam, _ = X.sod_rows(F[:NHEALTHY])
        k = min(20, len(lam))
        ax[0].semilogy(np.arange(1, k + 1), lam[:k], "o", ms=2.5, mfc="none",
                       color=c, label=n)
    ax[0].set_xlabel("SOV index"); ax[0].set_ylabel(r"$\lambda_p$")
    ax[0].set_xticks([1, 5, 10, 15, 20])
    ax[0].set_ylim(top=6e4)          # headroom for the legend
    ax[0].legend(fontsize=SMALL - 1, frameon=False, handletextpad=0.2,
                 borderaxespad=0.2, loc="upper right", ncol=1,
                 labelspacing=0.15)

    # (b) voltages recovered from the far-field shell alone
    F = sets[1][1]
    n0 = NHEALTHY
    Fc = F - F[:n0].mean(0)
    _, Psi, _, _ = X.sod_rows(F[:n0])
    soc = Fc @ Psi
    A = np.c_[soc[:n0, :2], np.ones(n0)]
    G = np.linalg.lstsq(A, np.c_[v1, v2][:n0], rcond=None)[0]
    vh = A @ G
    ax[1].plot(v1[:n0], v2[:n0], color="0.45", lw=1.2)
    ax[1].plot(vh[:, 0], vh[:, 1], color=COL[1], lw=0.3)
    ax[1].set_xlabel(r"$v_1$"); ax[1].set_ylabel(r"$v_2$")
    ax[1].set_aspect("equal")

    for (n, _, c) in sets:
        ax[2].plot(t, res[n]["q"], color=c, lw=0.5)
    ax[2].set_xlabel(r"$t$ (h)"); ax[2].set_ylabel(r"$\bar q_1$, normalized")

    for k, a in enumerate(ax):
        a.xaxis.label.set_size(SMALL); a.yaxis.label.set_size(SMALL)
        a.tick_params(labelsize=SMALL)
        panel(a, "abc"[k], x=-0.22, y=1.02); a.texts[-1].set_fontsize(SMALL)
    fig.tight_layout(pad=0.3, w_pad=1.6)
    fig.savefig(FIGDIR + "psw_farfield.pdf")
    plt.close(fig)
    print("wrote psw_farfield.pdf")


if __name__ == "__main__":
    main()
