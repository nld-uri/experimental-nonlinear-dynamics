"""pitchfork.pdf (label pitchfork): the pitchfork bifurcation of the unforced Duffing
oscillator x'' + gamma x' - beta x + x^3 = 0.
(a) equilibria x_e against beta on [-1, 1]: x_e = 0 (stable for beta < 0, col1 solid;
    unstable for beta > 0, col2 dashed) and x_e = +-sqrt(beta) (stable, col1).
(b), (c) the potential V(x) = -beta x^2/2 + x^4/4 at beta = -1 and beta = +1, equilibria
    marked (col1 stable, col2 unstable).
(d), (e) computed phase portraits (x, x') at the same two beta with the book's damping
    gamma = 0.25: trajectories from 4 initial conditions on the circle of radius 1.6 (d) and
    from 12 initial conditions on the box edge |x| = 1.8 or |x'| = 1.8 (e), solve_ivp RK45,
    rtol 1e-9, t in [0, 30]; the col2 curves in (e) are the two branches of the saddle's
    unstable manifold, started 1e-4 from the origin along the unstable eigenvector."""
import numpy as np
from scipy.integrate import solve_ivp
from _figstyle import *

GAMMA = 0.25
def rhs(beta):
    return lambda t, y: [y[1], -GAMMA*y[1] + beta*y[0] - y[0]**3]

def traj(beta, y0, tend=30.0):
    s = solve_ivp(rhs(beta), (0, tend), y0, rtol=1e-9, atol=1e-12, dense_output=True)
    return s.sol(np.linspace(0, tend, 4001))

fig = plt.figure(figsize=(6, 5.2))
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 0.9, 1.5], hspace=0.55, wspace=0.3)
axa = fig.add_subplot(gs[0, :])
b = np.linspace(0, 1, 200)
axa.plot([-1, 0], [0, 0], color=COL[0], lw=1.4)
axa.plot([0, 1], [0, 0], "--", color=COL[1], lw=1.4)
axa.plot(b, np.sqrt(b), color=COL[0], lw=1.4); axa.plot(b, -np.sqrt(b), color=COL[0], lw=1.4)
axa.text(-0.55, 0.1, "$x_e = 0$", ha="center"); axa.text(0.55, 0.08, "$x_e = 0$", ha="center", color=COL[1])
axa.text(0.45, 0.85, r"$x_e = \sqrt{\beta}$"); axa.text(0.05, -1.05, r"$x_e = -\sqrt{\beta}$", va="bottom")
axa.axvline(-1, color="0.6", lw=0.5, ls=":"); axa.axvline(1, color="0.6", lw=0.5, ls=":")
axa.set_xlim(-1.05, 1.05); axa.set_ylim(-1.3, 1.2)
axa.set_xlabel(r"$\beta$"); axa.set_ylabel("equilibrium $x_e$"); panel(axa, "a", x=-0.06)
axa.text(-0.98, -1.25, "(b), (d)", fontsize=8, color="0.4", ha="left", va="bottom")
axa.text(0.98, -1.25, "(c), (e)", fontsize=8, color="0.4", ha="right", va="bottom")

x = np.linspace(-1.6, 1.6, 400)
for k, (beta, p) in enumerate([(-1.0, "b"), (1.0, "c")]):
    ax = fig.add_subplot(gs[1, k])
    V = -beta*x**2/2 + x**4/4
    ax.plot(x, V, color="k", lw=1.0)
    if beta < 0: ax.plot(0, 0, "o", color=COL[0], ms=4, zorder=3)
    else:
        ax.plot([-1, 1], [-0.25, -0.25], "o", color=COL[0], ms=4, zorder=3); ax.plot(0, 0, "o", color=COL[1], ms=4, zorder=3)
    sketch_axes(ax, "$x$", "$V(x)$", yzero=False); ax.axhline(0, color="k", lw=0.6)
    ax.spines["left"].set_position(("data", 0)); ax.yaxis.set_label_coords(0.53, 0.98)
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-0.4, 0.8 if beta < 0 else 0.5)
    ax.text(0.98, 0.95, r"$\beta = %g$" % beta, transform=ax.transAxes, ha="right", va="top", fontsize=9)
    panel(ax, p, x=-0.02, y=0.9)

L = 1.8
for k, (beta, p) in enumerate([(-1.0, "d"), (1.0, "e")]):
    ax = fig.add_subplot(gs[2, k])
    if beta < 0:
        th = np.linspace(0, 2*np.pi, 4, endpoint=False) + 0.3
        ics = [(1.6*np.cos(a), 1.6*np.sin(a)) for a in th]
    else:
        s = np.linspace(-1.5, 1.5, 3)
        ics = [(v, L) for v in s] + [(v, -L) for v in s] + [(L, v) for v in s] + [(-L, v) for v in s]
    for y0 in ics:
        Y = traj(beta, y0)
        ax.plot(Y[0], Y[1], color=COL[0], lw=0.6)
        ax.plot(y0[0], y0[1], ".", color=COL[0], ms=3)
    if beta > 0:
        lam = (-GAMMA + np.sqrt(GAMMA**2 + 4*beta))/2
        for sgn in (1, -1):
            Y = traj(beta, (sgn*1e-4, sgn*1e-4*lam), 60.0)
            ax.plot(Y[0], Y[1], color=COL[1], lw=0.9)
        ax.plot([-1, 1], [0, 0], "o", color=COL[0], ms=4, zorder=3); ax.plot(0, 0, "x", color=COL[1], ms=5, zorder=3)
    else:
        ax.plot(0, 0, "o", color=COL[0], ms=4, zorder=3)
    ax.set_xlim(-L-0.1, L+0.1); ax.set_ylim(-L-0.1, L+0.1); ax.set_aspect("equal")
    ax.set_xlabel("$x$"); ax.set_ylabel(r"$\dot x$")
    ax.text(0.97, 0.96, r"$\beta = %g$" % beta, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(fc="w", ec="none", pad=1.5))
    panel(ax, p, x=-0.2)
fig.subplots_adjust(left=0.1, right=0.97, bottom=0.08, top=0.96)
fig.savefig(FIGDIR + "pitchfork.pdf")
