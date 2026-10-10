"""POD_Curv.pdf and POD_Curv_Corr.pdf -- the curvature bias of the local
projection and its correction (Chapter 14, figure POD_Curv).

Both panels are computed, not drawn. The clean set is 20 000 points uniformly
spaced on the unit circle; the neighbourhood of a base point is every point
within Euclidean distance EPS_DRAW of it, displaced by Gaussian noise of
standard deviation SIG for the drawing only (noise shifts no centre of mass to
first order, and the numbers below are the noise-free ones).

  left  (POD_Curv.pdf)      the cloud, its centre of mass c_k = ybar_n, and the
                            leading POM of the cloud through c_k -- a chord of
                            the arc, not a tangent, displaced inward by about
                            kappa eps^2/6.
  right (POD_Curv_Corr.pdf) the centres of mass c_k^(j) of the neighbours,
                            their mean c'_k = ybarbar_n, displaced inward twice
                            as far, and the corrected centre 2 c_k - c'_k,
                            back on the arc.

EPS_DRAW = 0.6 is used for the drawing so that the displacement is visible;
the text quotes eps = 0.3, for which this script prints |c_k| = 0.98496 (the
predicted 1 - eps^2/6 = 0.985), |c'_k| = 0.97014 and |2 c_k - c'_k| = 0.99977.

    python3 mkfig_POD_Curv.py
"""
import numpy as np
from _figstyle import plt, COL, FIGDIR

# 1. Enable LaTeX rendering
plt.rcParams['text.usetex'] = True

# 2. Add the 'bm' package to the LaTeX preamble
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

N = 20000
EPS_DRAW = 0.6
EPS_TEXT = 0.3
SIG = 0.012                      # drawing noise only
I0 = N//4                        # base point at the top of the circle


def circle(n=N):
    t = 2*np.pi*np.arange(n)/n
    return np.vstack([np.cos(t), np.sin(t)])


def com(Y, i, eps):
    """Centre of mass of the points of Y within eps of column i."""
    d = np.linalg.norm(Y - Y[:, i:i + 1], axis=0)
    return Y[:, d <= eps].mean(axis=1), np.where(d <= eps)[0]


def study(eps, n=N):
    """|c_k|, |c'_k| and |2 c_k - c'_k| for a noise-free circle."""
    Y = circle(n)
    c, nb = com(Y, I0, eps)
    cp = np.mean([com(Y, j, eps)[0] for j in nb[::max(1, nb.size//200)]],
                 axis=0)
    return np.linalg.norm(c), np.linalg.norm(cp), np.linalg.norm(2*c - cp)


def frame(ax, half, ylo, yhi):
    t = np.pi/2 + np.linspace(-1.5*half, 1.5*half, 400)
    ax.plot(np.cos(t), np.sin(t), color=COL[0], lw=1.3, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05*np.sin(1.5*half), 1.05*np.sin(1.5*half))
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


if __name__ == "__main__":
    print(" eps      |c_k|     |c'_k|   |2c_k-c'_k|")
    for e in (0.1, 0.2, EPS_TEXT, 0.4, EPS_DRAW):
        a, b, c = study(e)
        print("%4.2f   %9.5f %9.5f %9.5f   (1-eps^2/6 = %.5f)"
              % (e, a, b, c, 1 - e*e/6))

    Y = circle()
    c, nb = com(Y, I0, EPS_DRAW)
    sub = nb[::max(1, nb.size//200)]
    cj = np.array([com(Y, j, EPS_DRAW)[0] for j in sub]).T
    cp = cj.mean(axis=1)
    cc = 2*c - cp
    half = np.arcsin(EPS_DRAW/2)
    rng = np.random.default_rng(0)
    P = Y[:, nb[::30]] + SIG*rng.standard_normal((2, nb[::30].size))

    # ---- left: the cloud, its centre of mass, and the POD line -------------
    fig, ax = plt.subplots(figsize=(2.4, 1.45))
    frame(ax, half, 0.885, 1.035)
    ax.plot(P[0], P[1], ".", color="0.45", ms=2.0, zorder=2)
    u = np.linalg.svd(Y[:, nb] - c[:, None], full_matrices=False)[0][:, 0]
    s = np.linspace(-0.30, 0.30, 2)
    ax.plot(c[0] + s*u[0], c[1] + s*u[1], color=COL[1], lw=1.1, zorder=3)
    ax.plot([0], [1], "o", color=COL[0], ms=4, zorder=4)
    ax.plot([c[0]], [c[1]], "o", color=COL[1], ms=4, zorder=4)
    ax.annotate(r"$\tilde{\bm{y}}_n$", (0, 1), xytext=(0, 7.5),
                textcoords="offset points", color=COL[0])
    ax.annotate(r"$\bar{\bm{y}}_n=\bm{c}_k$", (c[0], c[1]),
                xytext=(-2, -10), textcoords="offset points", color=COL[1],
                ha="center")
    ax.annotate("", xy=(c[0], c[1]), xytext=(0, 1),
                arrowprops=dict(arrowstyle="->", color="0.2", lw=0.7))
    fig.tight_layout(pad=0.05)
    fig.savefig(FIGDIR + "POD_Curv.pdf")
    plt.close(fig)

    # ---- right: the correction --------------------------------------------
    fig, ax = plt.subplots(figsize=(2.7, 1.55))
    frame(ax, half, 0.855, 1.045)
    ax.plot(cj[0], cj[1], "--", color=COL[1], ms=2.0, zorder=2,
            label=r"$\mathbf{c}_k^{(j)}$")
    ax.plot([0], [1], "o", color=COL[0], ms=4, zorder=4)
    ax.plot([c[0]], [c[1]], "o", color=COL[1], ms=4, zorder=4)
    ax.plot([cp[0]], [cp[1]], "s", color=COL[3], ms=4, zorder=4)
    ax.plot([cc[0]], [cc[1]], "^", color=COL[4], ms=4, zorder=4)
    ax.annotate("", xy=(cp[0], cp[1]), xytext=(0, 1),
                arrowprops=dict(arrowstyle="->", color="0.2", lw=0.7))
    ax.annotate(r"$\tilde{\bm{y}}_n$", (0, 1), xytext=(6, 5),
                textcoords="offset points", color=COL[0])
    ax.annotate(r"$\bar{\bm{y}}_n$", (c[0], c[1]), xytext=(6, 4),
                textcoords="offset points", color=COL[1])
    ax.annotate(r"$\bar{\bar{\bm{y}}}_n$", (cp[0], cp[1]),
                xytext=(6, -2), textcoords="offset points", color=COL[3])
    ax.annotate(r"$2\bm{c}_k-\bm{c}_k^{\prime}$", (cc[0], cc[1]),
                xytext=(-37, 5), textcoords="offset points", color=COL[4])
    fig.tight_layout(pad=0.05)
    fig.savefig(FIGDIR + "POD_Curv_Corr.pdf")
