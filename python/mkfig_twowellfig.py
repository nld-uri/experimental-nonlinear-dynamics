"""twowellfig.pdf (label soluts): the unforced, damped two-well oscillator
x'' + gamma x' + x (x^2 - 1) = 0 with gamma = 0.25 (the damping of the book's Duffing
record). (a) potential V(x) = -x^2/2 + x^4/4; (b) basins of attraction in the (x, x')
plane on [-2.5, 2.5]^2: a 301 x 301 grid of initial conditions integrated by fixed-step
RK4 (dt = 0.02) to t = 100 and classified by the sign of x(100); the basin B_1 of the
equilibrium x = -1 is shaded, and the two sample trajectories of (c) are drawn in it;
(c) x(t) for the initial conditions (x, x') = (2.2, 0) (col2) and (0, -1) (col1), solve_ivp
RK45 with rtol 1e-9, t in [0, 60]."""
import numpy as np
from scipy.integrate import solve_ivp
from _figstyle import *

GAMMA = 0.25
IC = [((0.0, -1.0), COL[0]), ((2.2, 0.0), COL[1])]

def rhs(t, y):
    return [y[1], -GAMMA*y[1] - y[0]*(y[0]**2 - 1)]

# (b) basins by vectorized RK4
L, NG, DT, TEND = 2.5, 301, 0.02, 100.0
g = np.linspace(-L, L, NG); X, V = np.meshgrid(g, g)
def f(x, v): return v, -GAMMA*v - x*(x**2 - 1)
x, v = X.copy(), V.copy()
for _ in range(int(TEND/DT)):
    k1x, k1v = f(x, v); k2x, k2v = f(x+0.5*DT*k1x, v+0.5*DT*k1v)
    k3x, k3v = f(x+0.5*DT*k2x, v+0.5*DT*k2v); k4x, k4v = f(x+DT*k3x, v+DT*k3v)
    x = x + DT/6*(k1x+2*k2x+2*k3x+k4x); v = v + DT/6*(k1v+2*k2v+2*k3v+k4v)
B1 = x < 0
print("fraction of the grid in B_1: %.3f; unresolved |x|>0.5 from +-1: %d" % (B1.mean(), np.sum(abs(abs(x)-1) > 0.5)))

fig = plt.figure(figsize=(6, 3.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[1, 2.6], hspace=0.35, wspace=0.3)
axV = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[1, 0]); axT = fig.add_subplot(gs[:, 1])
xx = np.linspace(-1.6, 1.6, 400)
axV.plot(xx, -xx**2/2 + xx**4/4, color="k", lw=1.0)
axV.plot([-1, 1], [-0.25, -0.25], "o", color=COL[0], ms=3)
sketch_axes(axV, "$x$", "$V(x)$"); axV.spines["left"].set_position(("data", 0)); axV.yaxis.set_label_coords(0.53, 0.98)
axV.set_xlim(-1.7, 1.7); axV.set_ylim(-0.35, 0.45); panel(axV, "a", x=-0.05)
axB.contourf(g, g, B1.astype(float), levels=[0.5, 1.5], colors=[(0.75, 0.87, 0.95)])
axB.contour(g, g, B1.astype(float), levels=[0.5], colors="k", linewidths=0.5)
for (x0, c) in IC:
    s = solve_ivp(rhs, (0, 60), x0, rtol=1e-9, atol=1e-12, dense_output=True)
    tt = np.linspace(0, 60, 6001); y = s.sol(tt)
    axB.plot(y[0], y[1], color=c, lw=0.7); axB.plot(x0[0], x0[1], "o", color=c, ms=3)
axB.plot([-1, 1], [0, 0], "o", color="k", ms=3, mfc="w"); axB.plot([0], [0], "x", color="k", ms=4)
axB.text(0.3, 1.9, r"${\cal B}_1$", fontsize=9)
axB.set_xlim(-L, L); axB.set_ylim(-L, L); axB.set_aspect("equal")
axB.set_xlabel("$x$"); axB.set_ylabel(r"$\dot x$"); panel(axB, "b", x=-0.2)
for (x0, c) in IC:
    s = solve_ivp(rhs, (0, 60), x0, rtol=1e-9, atol=1e-12, dense_output=True)
    tt = np.linspace(0, 60, 6001); y = s.sol(tt)
    axT.plot(tt, y[0], color=c, lw=0.8, label=r"$(x_0,\dot x_0) = (%g, %g)$" % x0)
    print("IC", x0, "min x %.3f at t %.2f; |x+1| < 0.05 after t = %.1f" % (y[0].min(), tt[y[0].argmin()], tt[np.max(np.where(abs(y[0]+1) > 0.05))+1]))
axT.axhline(-1, color="k", lw=0.5, ls=":")
axT.set_xlim(0, 60); axT.set_ylim(-2.1, 2.5); axT.set_xlabel("time, $t$"); axT.set_ylabel("$x(t)$")
axT.legend(loc="upper right", fontsize=7)
axT.annotate("not stationary", xy=(12, -0.55), xytext=(17, 0.6), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
axT.annotate("approximately stationary", xy=(50, -0.97), xytext=(24, -1.75), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
panel(axT, "c", x=-0.1)
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.93)
fig.savefig(FIGDIR+"twowellfig.pdf")
