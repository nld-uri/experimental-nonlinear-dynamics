"""mass-spring.pdf -- the nine-mass planar spring network of Section 1.2
(Fig. 'One'), redrawn from D. Chelidze's figure with the same topology.

Masses m1..m9 move in the plane; springs join the pairs listed in SPRINGS and
tie the masses listed in GROUND to fixed supports; a cam (motor driver) forces
m6 through a spring; two optical sensors read the horizontal and vertical
displacement of m5, giving the observation (u, v).
"""
from _figstyle import *
from _scene3d import use_cm_fonts
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Polygon, Ellipse
from matplotlib.patches import FancyArrowPatch

use_cm_fonts()

# mass positions (from the original drawing, scaled to ~unit spacing)
P = {1: (0.6, 3.3), 2: (1.5, 5.0), 3: (4.0, 5.85), 4: (8.6, 6.65), 5: (11.1, 6.65),
     6: (6.5, 5.4), 7: (10.3, 5.0), 8: (4.85, 4.15), 9: (8.6, 3.3)}
SPRINGS = [(1, 2), (1, 8), (2, 3), (2, 6), (3, 4), (4, 5), (4, 6), (5, 7),
           (6, 7), (6, 8), (7, 9), (8, 9)]
GROUND = {1: (1.3, 0.5), 2: (2.7, 0.1), 3: (4.0, 1.4), 8: (5.25, 1.4),
          4: (7.3, 1.0), 9: (8.6, 0.5), 7: (9.4, 1.4), 5: (11.5, 2.3)}
MOTOR = (6.2, 0.55)

INK = "0.12"
MASS = np.array(COL[0])          # masses (blue)
MEAS = np.array(COL[1])          # the measured mass m5 (orange)


def spring(ax, a, b, n=4, amp=0.20, coil=1.1, lw=0.95, color=INK):
    """Zigzag spring from a to b: n teeth over a coil of fixed length `coil`
    centred on the span, straight leads to either end."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a; L = np.hypot(*d); t = d / L; nrm = np.array([-t[1], t[0]])
    lead = max(0.5 * (L - min(coil, 0.7 * L)), 0.15)
    s = np.linspace(lead, L - lead, 2 * n + 1)
    off = np.zeros_like(s); off[1:-1:2] = amp; off[2:-1:2] = -amp
    pts = [a] + [a + si * t + oi * nrm for si, oi in zip(s, off)] + [b]
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, solid_joinstyle="miter", zorder=2)


def ground(ax, p, w=0.42):
    x, y = p
    ax.plot([x - w / 2, x + w / 2], [y, y], color=INK, lw=1.6, zorder=2)
    for k in np.linspace(-w / 2, w / 2 - 0.08, 5):
        ax.plot([x + k, x + k + 0.10], [y - 0.02, y - 0.14], color=INK, lw=0.6, zorder=2)


def mass(ax, p, color, r=0.17):
    """Shaded sphere: base disc, darker rim, small highlight."""
    ax.add_patch(Circle(p, r, fc=color * 0.85, ec=color * 0.45, lw=0.8, zorder=4))
    ax.add_patch(Circle((p[0] - 0.04, p[1] + 0.04), r * 0.72,
                        fc=np.clip(color * 0.85 + 0.12, 0, 1), ec="none", zorder=4))
    ax.add_patch(Circle((p[0] - 0.06, p[1] + 0.07), r * 0.28, fc="white", ec="none",
                        alpha=0.85, zorder=5))


fig, ax = plt.subplots(figsize=(5.4, 3.5))

for i, j in SPRINGS:
    spring(ax, P[i], P[j])
for i, g in GROUND.items():
    spring(ax, P[i], (g[0], g[1]), amp=0.17)
    ground(ax, g)

# motor driver: eccentric cam on a pedestal, rotating at Omega(t), forcing m6
cx, cy = MOTOR
spring(ax, P[6], (cx, cy + 0.42), amp=0.17)
ax.add_patch(Ellipse((cx, cy + 0.12), 0.46, 0.62, fc="0.78", ec=INK, lw=0.9, zorder=3))
ax.add_patch(Circle((cx, cy + 0.30), 0.06, fc="white", ec=INK, lw=0.8, zorder=4))   # pin
ax.add_patch(Circle((cx, cy + 0.02), 0.09, fc="white", ec=INK, lw=0.8, zorder=4))   # axle
ax.plot([cx - 0.28, cx + 0.28], [cy - 0.22, cy - 0.22], color=INK, lw=1.6, zorder=3)
for k in np.linspace(-0.26, 0.18, 5):
    ax.plot([cx + k, cx + k + 0.10], [cy - 0.24, cy - 0.36], color=INK, lw=0.6)
ax.add_patch(FancyArrowPatch((cx + 0.30, cy + 0.45), (cx + 0.52, cy - 0.05),
                             connectionstyle="arc3,rad=-0.5", arrowstyle="-|>",
                             mutation_scale=8, lw=0.8, color=INK, zorder=4))
ax.text(cx + 0.62, cy + 0.05, r"$\Omega(t)$", fontsize=9, va="center")
ax.text(cx, cy - 0.55, "motor driver", fontsize=8, ha="center", va="top")

# optical sensors on m5: one reads the vertical, one the horizontal motion
x5, y5 = P[5]
ax.add_patch(Rectangle((x5 - 0.07, y5 + 0.32), 0.14, 0.75, fc="0.55", ec=INK, lw=0.6, zorder=3))
ax.add_patch(Rectangle((x5 + 0.32, y5 - 0.07), 0.75, 0.14, fc="0.55", ec=INK, lw=0.6, zorder=3))
for p0, p1 in (((x5, y5 + 0.32), (x5, y5 + 0.19)), ((x5 + 0.32, y5), (x5 + 0.19, y5))):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=COL[6],
                lw=0.7, mutation_scale=6), zorder=3)
bx, by = 11.75, 7.55
ax.add_patch(FancyBboxPatch((bx, by), 1.25, 0.75, boxstyle="round,pad=0.03",
                            fc="white", ec=INK, lw=0.9, zorder=3))
ax.text(bx + 0.625, by + 0.375, "optical\nsensors", fontsize=8, ha="center", va="center",
        linespacing=1.05)
ax.plot([x5, x5, bx], [y5 + 1.07, by + 0.375, by + 0.375], color=INK, lw=0.6)
ax.plot([x5 + 1.07, bx + 0.625, bx + 0.625], [y5, y5, by], color=INK, lw=0.6)
ax.annotate("", xy=(bx + 0.625, by + 1.15), xytext=(bx + 0.625, by + 0.78),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=7))
ax.text(bx + 0.625, by + 1.25, r"$(u, v)$", fontsize=9, ha="center", va="bottom")

for i, p in P.items():
    mass(ax, np.array(p), MEAS if i == 5 else MASS)
    dx, dy = {1: (-0.30, 0.22), 5: (-0.42, 0.18), 7: (0.38, -0.12), 9: (0.0, 0.32)}.get(i, (0.0, 0.32))
    ax.text(p[0] + dx, p[1] + dy, f"$m_{i}$", fontsize=9, ha="center", va="bottom")

ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(0.0, 13.2); ax.set_ylim(-0.2, 9.1)
fig.tight_layout(pad=0.1)
fig.savefig(FIGDIR + "mass-spring.pdf")
print("wrote mass-spring.pdf")
