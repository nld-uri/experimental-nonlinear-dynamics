"""stability_diagram.pdf (label stability_diagram): classification of the linear
two-dimensional flow x' = A x in the (tau, Delta) plane, tau = trace A, Delta = det A,
with the parabola tau^2 = 4 Delta. Every phase portrait is computed: trajectories
x(t) = expm(A t) x0 from initial conditions on a circle of radius 0.55 (12 for nodes, stars,
saddle and degenerate nodes, 4 for the spirals; three circles of radius 0.3, 0.6, 0.9 for the
center; 16 points on the box edges for the lines),
integrated forward and backward in time and clipped to the box [-1, 1]^2; the arrow on each
trajectory points in the direction of increasing time; the straight eigendirections are
drawn in col2 where they are real. The matrices used (eigenvalues in brackets):
stable/unstable spiral [-0.25 +- i], [0.25 +- i]; center [+- i]; stable/unstable node
[-1, -3] / [1, 3] with eigenvectors (1, 0.5) and (1, -1); stable/unstable star -I / I;
stable/unstable degenerate node [[-1,1],[0,-1]] / [[1,1],[0,1]]; saddle [-1, 1] with the
same eigenvectors as the nodes; stable/unstable lines [[-1,1],[0,0]] / [[1,-1],[0,0]]."""
import numpy as np
from scipy.linalg import expm
from _figstyle import *

R = np.array([[1.0, 1.0], [0.5, -1.0]])            # eigenvector matrix for nodes/saddle
def sim(l1, l2): return R @ np.diag([l1, l2]) @ np.linalg.inv(R)
CASES = [   # (name, A, tau-Delta position of the inset, label offset dx, dy, label ha)
    ("stable\nspirals",   np.array([[-0.25, 1], [-1, -0.25]]), (-1.05, 1.95), (0, 0.55), "center"),
    ("unstable\nspirals", np.array([[0.25, 1], [-1, 0.25]]),   (1.05, 1.95), (0, 0.55), "center"),
    ("centers",           np.array([[0, 1], [-1, 0]]),         (0.0, 0.95), (0, 0.55), "center"),
    ("stable\nnodes",     sim(-1, -3),                          (-2.85, 0.55), (0, 0.55), "center"),
    ("unstable\nnodes",   sim(1, 3),                            (2.85, 0.55), (0, 0.55), "center"),
    ("stable\nstars",     -np.eye(2),                           (-1.55, 0.6), (0, 0.55), "center"),
    ("unstable\nstars",   np.eye(2),                            (1.55, 0.6), (0, 0.55), "center"),
    ("stable\ndegenerate\nnodes", np.array([[-1, 1], [0, -1]]), (-2.6, 1.9), (-0.55, 0.0), "right"),
    ("unstable\ndegenerate\nnodes", np.array([[1, 1], [0, 1]]), (2.6, 1.9), (0.55, 0.0), "left"),
    ("saddles",           sim(-1, 1),                           (0.0, -0.85), (0.0, -0.6), "center"),
    ("stable lines",      np.array([[-1, 1], [0, 0]]),          (-1.5, -0.7), (0, -0.55), "center"),
    ("unstable lines",    np.array([[1, -1], [0, 0]]),          (1.5, -0.7), (0, -0.55), "center"),
]

def portrait(axi, A):
    ev, V = np.linalg.eig(A)
    lines = abs(ev.imag).max() < 1e-12 and abs(ev[0]*ev[1]) < 1e-12
    complex_ = abs(ev.imag).max() > 1e-12
    k = 16 if lines else (4 if complex_ else 12)
    th = np.linspace(0, 2*np.pi, k, endpoint=False) + 0.1
    X0 = np.vstack([np.cos(th), np.sin(th)]) * 0.55
    if complex_ and abs(ev.real).max() < 1e-12:      # center: three closed orbits
        k = 3; X0 = np.vstack([[0.3, 0.6, 0.9], [0, 0, 0]])
    if lines:   # start on the box edges so the straight trajectories fill the box
        s = np.linspace(-0.9, 0.9, k); X0 = np.vstack([0.95*(-1)**np.arange(k), s])
    ts = np.linspace(-6, 6, 1201)
    Ms = [expm(A*t) for t in ts]
    for j in range(k):
        Y = np.array([M @ X0[:, j] for M in Ms])
        inside = (abs(Y) <= 1.0).all(axis=1)
        # keep the contiguous run containing t = 0
        i0 = len(ts)//2
        a, b = i0, i0
        while a > 0 and inside[a-1]: a -= 1
        while b < len(ts)-1 and inside[b+1]: b += 1
        Y = Y[a:b+1]
        if len(Y) < 3: continue
        axi.plot(Y[:, 0], Y[:, 1], color=COL[0], lw=0.6)
        m = len(Y)//2
        axi.annotate("", xy=Y[min(m+2, len(Y)-1)], xytext=Y[m],
                     arrowprops=dict(arrowstyle="-|>", color=COL[0], lw=0.6, mutation_scale=5))
    if abs(ev.imag).max() < 1e-12:      # real eigendirections
        for q in range(2):
            v = V[:, q].real; v = v/np.abs(v).max()
            axi.plot([-v[0], v[0]], [-v[1], v[1]], color=COL[1], lw=0.7)
    axi.axhline(0, color="0.5", lw=0.4); axi.axvline(0, color="0.5", lw=0.4)
    axi.set_xlim(-1, 1); axi.set_ylim(-1, 1); axi.set_xticks([]); axi.set_yticks([])
    axi.patch.set_alpha(0)
    for s in axi.spines.values(): s.set_visible(False)

W, H = 6.0, 4.3
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
ax.set_xlim(-4.0, 4.0); ax.set_ylim(-1.62, 2.65)
for s in ("top", "right", "bottom", "left"): ax.spines[s].set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.annotate("", xy=(3.7, 0), xytext=(-3.7, 0), arrowprops=dict(arrowstyle="->", lw=0.7))
ax.annotate("", xy=(0, 2.6), xytext=(0, -1.4), arrowprops=dict(arrowstyle="->", lw=0.7))
ax.text(3.6, 0.08, r"$\tau$", fontsize=11); ax.text(0.08, 2.5, r"$\Delta$", fontsize=11)
tt = np.linspace(-3.2, 3.2, 300); ax.plot(tt, tt**2/4, color=COL[0], lw=1.4)
ax.text(2.2, 2.35, r"$\tau^2 = 4\Delta$", color=COL[0], fontsize=10, ha="right")
ax.text(-0.15, -0.28, "0", fontsize=8, ha="right")
sz = 0.135   # inset width as fraction of figure width
for name, A, (tx, dy), (lx, ly), ha in CASES:
    cx, cy = ax.transData.transform((tx, dy)); cx /= W*fig.dpi; cy /= H*fig.dpi
    axi = fig.add_axes([cx - sz/2, cy - sz*W/H/2, sz, sz*W/H])
    portrait(axi, A)
    ax.text(tx + lx, dy + ly, name, fontsize=8, ha=ha, va="center", linespacing=0.95)
fig.savefig(FIGDIR + "stability_diagram.pdf")
