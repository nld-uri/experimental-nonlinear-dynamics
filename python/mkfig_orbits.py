"""closedO.pdf, quasiperiodic.pdf, chaotic.pdf -- the three orbit types of
Section 'Dimensionality of Continuous Attractors'.

closedO      : two closed curves drawn parametrically (a schematic, as before):
               a period-1 loop and a period-2 loop that closes after two turns.
quasiperiodic: x(t) = A1 sin t + A2 sin(sqrt 7 t), A1 = 1, A2 = 0.4, 0 <= t <= T.
               (a) the signal; (b) the orbit on the 2-torus with angles
               theta1 = t, theta2 = sqrt 7 t (major radius A1, tube radius A2);
               the first two turns are highlighted.
chaotic      : the running two-well Duffing oscillator x'' + 0.25 x' - x + x^3 =
               0.3 cos t, drawn in the cylindrical phase space (x, x', phi = t mod
               2 pi): NPER forcing periods of the trajectory, and the stroboscopic
               Poincare section at phi = 0 (NSEC points) on the half-plane phi = 0.
"""
from _figstyle import *
from _ch08data import duffing, duffing_strobe

# ---- closedO ------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(4.2, 1.7))
t = np.linspace(0, 2*np.pi, 400)
x, y = np.cos(t), 0.6*np.sin(t)
axs[0].plot(x, y, color=COL[0])
axs[0].annotate("", xy=(x[210], y[210]), xytext=(x[200], y[200]),
                arrowprops=dict(arrowstyle="-|>", color=COL[0], lw=0.8))
axs[0].text(-1.0, 0.75, "$P$-$1$", fontsize=9)
t = np.linspace(0, 4*np.pi, 800)
r = 1 + 0.28*np.cos(t/2)
x, y = r*np.cos(t), 0.6*r*np.sin(t)
axs[1].plot(x, y, color=COL[0])
for k in (100, 500):
    axs[1].annotate("", xy=(x[k + 10], y[k + 10]), xytext=(x[k], y[k]),
                    arrowprops=dict(arrowstyle="-|>", color=COL[0], lw=0.8))
axs[1].text(-1.3, 0.85, "$P$-$2$", fontsize=9)
for a in axs:
    a.set_aspect("equal"); a.axis("off"); a.set_xlim(-1.4, 1.4); a.set_ylim(-0.95, 0.95)
fig.tight_layout(pad=0.2)
fig.savefig(FIGDIR + "closedO.pdf")
print("wrote closedO.pdf")

# ---- quasiperiodic -------------------------------------------------------------
A1, A2, T = 1.0, 0.4, 250.0
t = np.arange(0, T, 0.01)
x = A1*np.sin(t) + A2*np.sin(np.sqrt(7)*t)
fig = plt.figure(figsize=(6.0, 2.4))
ax = fig.add_axes([0.10, 0.2, 0.40, 0.72])
m = t <= 40
ax.plot(t[m], x[m], color=COL[0])
ax.plot(t[t <= 4*np.pi], x[t <= 4*np.pi], color=COL[1], lw=1.6)
ax.set_xlabel("$t$"); ax.set_ylabel("$x(t)$"); ax.set_xlim(0, 40); ax.set_ylim(-1.5, 1.5)
panel(ax, "a")
ax3 = fig.add_axes([0.52, -0.25, 0.52, 1.5], projection="3d")
th1, th2 = t, np.sqrt(7)*t
X = (A1 + A2*np.cos(th2))*np.cos(th1)
Y = (A1 + A2*np.cos(th2))*np.sin(th1)
Z = A2*np.sin(th2)
ax3.plot(X, Y, Z, color=COL[0], lw=0.25)
m = t <= 4*np.pi
ax3.plot(X[m], Y[m], Z[m], color=COL[1], lw=1.4)
ax3.set_box_aspect((1, 1, 0.4))
ax3.set_xlim(-1.3, 1.3); ax3.set_ylim(-1.3, 1.3); ax3.set_zlim(-0.5, 0.5)
ax3.view_init(elev=40, azim=-60)
ax3.set_axis_off()
fig.text(0.55, 0.9, "(b)", fontsize=9)
fig.savefig(FIGDIR + "quasiperiodic.pdf")
print("wrote quasiperiodic.pdf  A1 =", A1, "A2 =", A2, "T =", T)

# ---- chaotic -------------------------------------------------------------------
NPER, NSEC, DT = 30, 4000, 2*np.pi/200
Y = duffing(N=int(NPER*2*np.pi/DT) + 1, dt=DT, ttrans=200.0)
tt = 200.0 + DT*np.arange(Y.shape[1])
phi = np.mod(tt, 2*np.pi)
R0 = 3.0                                  # offset so the cylinder does not self-cross
X = (Y[0] + R0)*np.cos(phi); Yc = (Y[0] + R0)*np.sin(phi); Z = Y[1]
sec = duffing_strobe(nper=NSEC, n=200, ntrans=50)
fig = plt.figure(figsize=(6.0, 2.6))
ax3 = fig.add_axes([-0.22, -0.4, 0.95, 1.8], projection="3d")
# break the line where phi wraps so no chord is drawn across the cylinder
brk = np.where(np.diff(phi) < 0)[0] + 1
for seg in np.split(np.arange(phi.size), brk):
    ax3.plot(X[seg], Yc[seg], Z[seg], color=COL[0], lw=0.4, alpha=0.85)
# the half-plane phi = 0 and the section points on it
xs = np.linspace(-1.8, 1.8, 2); zs = np.linspace(-1.2, 1.2, 2)
XS, ZS = np.meshgrid(xs, zs)
ax3.plot_surface(XS + R0, np.zeros_like(XS), ZS, color="0.75", alpha=0.35, shade=False)
ax3.scatter(sec[0] + R0, np.zeros(NSEC), sec[1], s=0.8, color=COL[1], depthshade=False, zorder=10)
ax3.set_box_aspect((1, 1, 0.35))
ax3.set_xlim(-4.5, 4.5); ax3.set_ylim(-4.5, 4.5); ax3.set_zlim(-1.3, 1.3)
ax3.view_init(elev=30, azim=-40)
ax3.set_axis_off()
fig.text(0.03, 0.9, "(a)", fontsize=9)
ax = fig.add_axes([0.62, 0.2, 0.36, 0.72])
ax.plot(sec[0], sec[1], ".", ms=1.0, color=COL[1], rasterized=False)
ax.set_xlabel("$x$"); ax.set_ylabel(r"$\dot x$")
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.6, 0.9)
panel(ax, "b", x=-0.2)
fig.savefig(FIGDIR + "chaotic.pdf")
print("wrote chaotic.pdf  periods =", NPER, "section points =", NSEC,
      "x range %.2f..%.2f  v range %.2f..%.2f" % (sec[0].min(), sec[0].max(), sec[1].min(), sec[1].max()))
