"""psw_track.pdf and psw_damage.pdf -- the measured SOD tracking example of
Chapter 17 (figures pswtrack and pswdamage).

Two stages on the same 1728 x 405 tracking matrix:

  psw_track   the operating subspace, identified from the healthy stage
              (a) SOVs, (b) the two smoothest SOCs against time,
              (c) their phase portrait beside the known voltage portrait

  psw_damage  the damage variable, in the complement of that subspace
              (a) SOVs after projection, (b) the normalized damage SOC,
              (c) its reciprocal against time with the Paris-law fit

Run from python/: writes ../figures/. Every number printed here is quoted in
Chapter 17.
"""
import numpy as np
from _figstyle import *
from _psw2005 import tracking, NHEALTHY
from gsvd import sod
from scipy.linalg import subspace_angles

SMALL = 8
NOP = 2                     # operating-condition modes to project out


def sod_rows(M):
    """SOD of a tracking matrix whose ROWS are the feature vectors."""
    S = M.T - M.T.mean(axis=1, keepdims=True)
    return sod(S)


def main():
    Y, t, v1, v2 = tracking()
    n0 = NHEALTHY
    Yc = Y - Y[:n0].mean(axis=0)

    # stage 1 -- the operating subspace, from the healthy stage only
    Phi, Psi, lam, _ = sod_rows(Y[:n0])
    soc = Yc @ Psi
    print("SOVs, healthy stage 1-8:", np.round(lam[:8], 2))
    print("  ratio SOV2/SOV3 = %.1f" % (lam[1]/lam[2]))

    # the paper's least-squares voltage subspace, for the angle check
    V = np.c_[v1 - v1[:n0].mean(), v2 - v2[:n0].mean()][:n0]
    B = np.linalg.lstsq(Yc[:n0], V, rcond=None)[0]
    P = Yc[:n0] @ B
    ang = subspace_angles(np.linalg.qr(P)[0], np.linalg.qr(soc[:n0, :2])[0])
    print("  subspace angles(voltage plane, first two SOCs) = %s rad"
          % np.round(ang, 4))

    fig, ax = plt.subplots(1, 3, figsize=(6.0, 2.1))
    ax[0].semilogy(np.arange(1, 21), lam[:20], "o", ms=3, color=COL[0],
                   mfc="none")
    ax[0].set_xlabel("SOV index"); ax[0].set_ylabel(r"$\lambda_p$")
    ax[0].set_xticks([1, 5, 10, 15, 20])
    ax[1].plot(t[:n0], soc[:n0, 0], color=COL[0], lw=0.5, label=r"$q_1$")
    ax[1].plot(t[:n0], soc[:n0, 1], color=COL[1], lw=0.5, label=r"$q_2$")
    ax[1].set_xlabel(r"$t$ (h)")
    ax[1].set_ylabel("SOC", labelpad=1)
    ax[1].legend(fontsize=SMALL, handlelength=1.2, ncol=2, frameon=False,
                 columnspacing=0.6, loc="upper right", handletextpad=0.4,
                 bbox_to_anchor=(1.02, 1.20))
    # The SOCs recover the SUBSPACE, not the individual voltages, so the
    # comparison is made after the least-squares affine map from (q1, q2) to
    # (v1, v2) fitted on the healthy stage. R^2 is the honest figure of merit.
    G = np.linalg.lstsq(np.c_[soc[:n0, :2], np.ones(n0)],
                        np.c_[v1[:n0], v2[:n0]], rcond=None)[0]
    vh = np.c_[soc[:n0, :2], np.ones(n0)] @ G
    R2 = 1 - ((np.c_[v1[:n0], v2[:n0]] - vh).var(0)
              / np.c_[v1[:n0], v2[:n0]].var(0))
    print("  R^2 of (v1, v2) on the two SOCs: %.4f, %.4f" % tuple(R2))
    # no legend here: two curves in a two-inch panel, named in the caption
    ax[2].plot(v1[:n0], v2[:n0], color=COL[2], lw=1.1)
    ax[2].plot(vh[:, 0], vh[:, 1], color=COL[0], lw=0.3)
    ax[2].set_xlabel(r"$v_1$"); ax[2].set_ylabel(r"$v_2$")
    ax[2].set_aspect("equal")
    for k, a in enumerate(ax):
        a.xaxis.label.set_size(SMALL); a.yaxis.label.set_size(SMALL)
        a.tick_params(labelsize=SMALL)
        panel(a, "abc"[k], x=-0.22, y=1.02); a.texts[-1].set_fontsize(SMALL)
    fig.tight_layout(pad=0.3, w_pad=1.6)
    fig.savefig(FIGDIR + "psw_track.pdf")
    plt.close(fig)

    # stage 2 -- the damage variable, orthogonal to the operating subspace
    Ybar = Yc @ Psi[:, NOP:]
    Phi2, Psi2, lam2, _ = sod_rows(Ybar)
    dam = (Ybar - Ybar[:n0].mean(axis=0)) @ Psi2[:, 0]
    if np.polyfit(t, dam, 1)[0] < 0:
        dam = -dam
    dn = (dam - dam.min())/(dam.max() - dam.min())
    print("SOVs after projection 1-8:", np.round(lam2[:8], 2))
    print("  ratio SOV1/SOV2 = %.1f" % (lam2[0]/lam2[1]))
    m = t >= 10.0
    p = np.polyfit(t[m], 1/dn[m], 1)
    r = np.corrcoef(t[m], 1/dn[m])[0, 1]
    print("  1/q1 over 10-12.28 h: slope %.2f /h, r = %.4f" % (p[0], r))

    fig, ax = plt.subplots(1, 3, figsize=(6.0, 2.1))
    ax[0].semilogy(np.arange(1, 21), lam2[:20], "o", ms=3, color=COL[0],
                   mfc="none")
    ax[0].set_xlabel("SOV index"); ax[0].set_ylabel(r"$\bar\lambda_p$")
    ax[0].set_xticks([1, 5, 10, 15, 20])
    ax[1].plot(t, dn, color=COL[0], lw=0.5)
    ax[1].axvline(t[n0], color="0.6", lw=0.6, ls="--")
    ax[1].text(t[n0] - 0.3, 0.72, "healthy stage", fontsize=SMALL - 1,
               rotation=90, ha="right", va="center", color="0.4")
    ax[1].set_xlabel(r"$t$ (h)"); ax[1].set_ylabel(r"$\bar q_1$, normalized")
    ax[2].plot(t[m], 1/dn[m], color=COL[0], lw=0.5)
    ax[2].plot(t[m], np.polyval(p, t[m]), color=COL[1], lw=0.8)
    ax[2].set_xlabel(r"$t$ (h)"); ax[2].set_ylabel(r"$1/\bar q_1$")
    for k, a in enumerate(ax):
        a.xaxis.label.set_size(SMALL); a.yaxis.label.set_size(SMALL)
        a.tick_params(labelsize=SMALL)
        panel(a, "abc"[k], x=-0.22, y=1.02); a.texts[-1].set_fontsize(SMALL)
    fig.tight_layout(pad=0.3, w_pad=1.6)
    fig.savefig(FIGDIR + "psw_damage.pdf")
    plt.close(fig)
    return lam, lam2, ang, p, r


if __name__ == "__main__":
    main()
