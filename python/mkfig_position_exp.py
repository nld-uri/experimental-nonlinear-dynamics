"""position_exp.pdf -- panel (b) of Chapter 17's characteristic-position figure,
rebuilt from the measured record.

The characteristic position of Section sec:mt_chardist is the mean unit vector
from a reference point to the trajectory, m columns per reference point instead
of one. This script computes it on the 1728 windows of the raw strain record of
Section 17.4 and puts it through the same two stages: the operating subspace
from the healthy stage, then the damage in the complement.

Sixty reference points at m = 5 give a 1728 x 300 tracking matrix, the same
width as the characteristic-distance matrix of Table 17.1, so the two are
compared at equal L*m rather than equal L.

Run from python/. About a minute.
"""
import numpy as np
from scipy.linalg import subspace_angles
from _figstyle import *
from _psw2005 import tracking, raw, NHEALTHY
import chk_farfield_exp as X

SMALL = 8
LP = 60                       # reference points; LP*DIM = 300 columns


def charpos(W, M, ref):
    """Mean unit vector from each reference point to the window's trajectory."""
    Y = np.empty((X.NW, ref.shape[0]*X.DIM))
    for l in range(X.NW):
        P = X.embed(W, l, M)
        dv = P[:, None, :] - ref[None, :, :]
        Y[l] = (dv/np.sqrt((dv**2).sum(-1))[..., None]).mean(0).ravel()
    return Y


def main():
    Yt, t, v1, v2 = tracking()
    W, M = X.windows(raw())
    pool, mid, half = X.box(W, M)
    rng = np.random.default_rng(X.SEED)
    ref = mid + (2*rng.random((LP, X.DIM)) - 1)*half*X.ENLARGE
    F = charpos(W, M, ref)

    d = X.stages(F, v1, v2, t)
    print("characteristic position, L = %d reference points -> %d columns"
          % (LP, F.shape[1]))
    print("  healthy-stage SOVs 1-3: %s" % np.round(d["sov"], 1))
    print("  R^2 on v1, v2         : %.4f, %.4f" % tuple(d["R2"]))
    print("  principal angles      : %.4f, %.4f rad" % tuple(d["ang"]))
    print("  damage SOVs           : %.1f vs %.1f" % tuple(d["sov2"]))
    print("  Paris r               : %.4f" % d["r"])
    Fm = X.moments(W, M)
    A = np.c_[(Fm - Fm.mean(0))/Fm.std(0), np.ones(X.NW)]
    Fc = F - F.mean(0)
    res = Fc - A @ np.linalg.lstsq(A, Fc, rcond=None)[0]
    print("  R^2 on the 20 moment features: %.5f"
          % (1 - res.var(0).sum()/Fc.var(0).sum()))

    n0 = NHEALTHY
    Fc = F - F[:n0].mean(0)
    _, Psi, _, _ = X.sod_rows(F[:n0])
    soc = Fc @ Psi
    A2 = np.c_[soc[:n0, :2], np.ones(n0)]
    G = np.linalg.lstsq(A2, np.c_[v1, v2][:n0], rcond=None)[0]
    vh = A2 @ G

    # One panel, drawn at the width it is included at, so nothing is rescaled.
    # The damage coordinate is not repeated here; it is Fig. pswdamage(b).
    fig, ax = plt.subplots(figsize=(2.9, 2.2))
    ax.plot(v1[:n0], v2[:n0], color="0.45", lw=1.2)
    ax.plot(vh[:, 0], vh[:, 1], color=COL[0], lw=0.3)
    ax.set_xlabel(r"$v_1$"); ax.set_ylabel(r"$v_2$")
    ax.set_aspect("equal")
    ax.xaxis.label.set_size(SMALL); ax.yaxis.label.set_size(SMALL)
    ax.tick_params(labelsize=SMALL)
    panel(ax, "b", x=-0.22)
    ax.texts[-1].set_fontsize(SMALL)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGDIR + "position_exp.pdf")
    plt.close(fig)
    print("wrote position_exp.pdf")


if __name__ == "__main__":
    main()
