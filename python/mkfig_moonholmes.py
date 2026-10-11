"""moonholmes.pdf -- the Moon-Holmes magneto-elastic oscillator of Section 1.2
(Fig. 'OneTwo'): (a) side elevation of the apparatus, redrawn from the
original schematic; (b) the double-well potential near the free end.

(a) A function generator and power amplifier drive an electromagnetic shaker,
    whose armature moves a rigid frame horizontally (base motion). A steel
    cantilever hangs from the top of the frame; two permanent magnets on the
    frame's base buckle its free end into one of two equilibria. A strain
    gauge near the clamp gives the measured signal. x runs down the beam and
    u(x,t) is its transverse deflection.
(b) V(u_L) = -u_L^2/2 + u_L^4/4 in scaled units, the standard two-well form
    of the Duffing model of this apparatus (Holmes 1979); wells at u_L = +-1.
"""
from _figstyle import *
from _scene3d import use_cm_fonts
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, Circle

use_cm_fonts()
INK = "0.12"
ALU, STEEL, BODY = "0.86", "0.55", "0.70"
MAG_N, MAG_S = COL[6], COL[0]                 # magnet poles: dark red / blue

fig = plt.figure(figsize=(6.0, 2.75))
ax = fig.add_axes([0.0, 0.0, 0.60, 1.0])
bx = fig.add_axes([0.67, 0.17, 0.31, 0.72])


def rect(x0, y0, w, h, fc, lw=0.8, z=2, **k):
    ax.add_patch(Rectangle((x0, y0), w, h, fc=fc, ec=INK, lw=lw, zorder=z, **k))


def box(x0, y0, w, h, text):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.04", fc="white",
                                ec=INK, lw=0.8, zorder=3))
    ax.text(x0 + w / 2, y0 + h / 2, text, fontsize=7.5, ha="center", va="center",
            linespacing=1.05, zorder=4)


def arrow(p0, p1, both=False, lw=0.8, ms=7):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(
        arrowstyle="<|-|>" if both else "-|>", color=INK, lw=lw, mutation_scale=ms,
        shrinkA=0, shrinkB=0), zorder=5)


def lab(text, xy, xytext, ha="left"):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=7.5, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color="0.3", lw=0.5, shrinkA=1, shrinkB=1),
                zorder=6)


# ---- signal chain ----------------------------------------------------------
box(-0.15, 0.55, 1.15, 0.55, "function\ngenerator")
box(-0.15, 1.55, 1.15, 0.55, "power\namplifier")
arrow((0.425, 1.13), (0.425, 1.51))
ax.plot([0.425, 0.425, 1.45], [2.14, 2.55, 2.55], color=INK, lw=0.8, zorder=2)
arrow((1.45, 2.55), (1.62, 2.55))

# ---- shaker: body on a trunnion, armature to the frame ------------------------
rect(1.65, 1.75, 1.45, 1.55, BODY, lw=0.9)
for yy in np.linspace(1.95, 3.10, 6):                     # cooling fins
    ax.plot([1.72, 3.03], [yy, yy], color="0.55", lw=0.5, zorder=3)
ax.add_patch(Polygon([(1.85, 1.75), (2.90, 1.75), (3.05, 1.0), (1.70, 1.0)],
                     closed=True, fc="0.80", ec=INK, lw=0.8, zorder=1))   # stand
ax.add_patch(Circle((2.375, 1.75), 0.10, fc="white", ec=INK, lw=0.7, zorder=3))
rect(3.10, 2.42, 0.42, 0.26, STEEL)                       # armature
ax.plot([1.45, 3.75], [0.98, 0.98], color=INK, lw=1.2, zorder=1)        # floor
lab("shaker", (2.2, 3.30), (1.85, 3.80))

# ---- frame (moves with the armature) -------------------------------------------
rect(3.52, 0.95, 0.20, 3.55, ALU, lw=0.9)                  # upright
rect(3.52, 4.30, 1.60, 0.20, ALU, lw=0.9)                  # top arm
rect(3.72, 0.95, 2.05, 0.20, ALU, lw=0.9)                  # base
rect(4.62, 3.98, 0.34, 0.32, "0.70", lw=0.8)               # clamp

# ---- beam: cantilever, deflected toward the right magnet ------------------------
x_c, y_top, y_tip, dlt = 4.79, 3.98, 1.62, 0.26
yy = np.linspace(y_top, y_tip, 60)
s = (y_top - yy) / (y_top - y_tip)
xb = x_c + dlt * s**2 * (3 - s) / 2
ax.fill_betweenx(yy, xb - 0.035, xb + 0.035, color=STEEL, ec=INK, lw=0.7, zorder=3)
ax.add_patch(Rectangle((x_c - 0.03, 3.45), 0.10, 0.28, fc=COL[1], ec=INK, lw=0.5, zorder=4))
lab("strain gauge", (x_c + 0.07, 3.60), (5.25, 3.85))
lab("beam", (xb[30] + 0.04, yy[30]), (5.40, 2.85))
# u(x,t): deflection from the undeflected line at one station
k = 42
ax.plot([x_c, x_c], [y_top, 1.35], color="0.45", lw=0.6, ls=(0, (3, 2)), zorder=2)
arrow((x_c, yy[k]), (xb[k] - 0.035, yy[k]), lw=0.6, ms=5)
ax.text(x_c - 0.05, yy[k], r"$u(x,t)$", fontsize=8, ha="right", va="center")
arrow((x_c - 0.55, y_top - 0.05), (x_c - 0.55, y_top - 0.55), lw=0.6, ms=5)
ax.text(x_c - 0.62, y_top - 0.30, r"$x$", fontsize=8, ha="right", va="center")

# ---- permanent magnets on the base under the free end ----------------------------
for xm in (x_c - 0.42, x_c + 0.42):
    rect(xm - 0.13, 1.15, 0.26, 0.13, MAG_S, lw=0.6, z=3)
    rect(xm - 0.13, 1.28, 0.26, 0.13, MAG_N, lw=0.6, z=3)
lab("permanent magnets", (x_c + 0.55, 1.33), (5.30, 1.85))

# ---- base motion ----------------------------------------------------------------
arrow((3.80, 0.62), (5.50, 0.62), both=True, lw=0.9, ms=8)
ax.text(4.65, 0.50, "base motion", fontsize=7.5, ha="center", va="top")

ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-0.25, 6.6); ax.set_ylim(0.15, 4.65)
ax.text(-0.2, 4.55, "(a)", fontsize=9, va="top")

# ---- (b) double-well potential -----------------------------------------------------
u = np.linspace(-1.55, 1.55, 400)
bx.plot(u, -u**2 / 2 + u**4 / 4, color=COL[0], lw=1.4)
bx.axhline(0, color="0.6", lw=0.5, zorder=0)
for xw in (-1, 1):
    bx.plot([xw], [-0.25], "o", ms=3.5, color=COL[1], zorder=3)
bx.set_xticks([-1, 0, 1]); bx.set_xticklabels([r"$-1$", r"$0$", r"$1$"])
bx.set_yticks([])
bx.set_xlabel(r"$u(L,t) = u_L$", fontsize=8.5)
bx.set_ylabel(r"$V(u_L)$", fontsize=8.5)
bx.set_xlim(-1.6, 1.6); bx.set_ylim(-0.38, 0.62)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
bx.text(-1.95, 0.70, "(b)", fontsize=9, va="top")

fig.savefig(FIGDIR + "moonholmes.pdf")
print("wrote moonholmes.pdf")
