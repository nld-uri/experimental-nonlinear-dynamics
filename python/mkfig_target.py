"""target.pdf -- shooting an arrow at a target (Section 1.2, Fig. 'target').
Replaces a clip-art silhouette.

(a) Side view: an archer (line drawing) and the trajectories of successive
    shots toward the target. (b) The target face: successive hits n = 1..4 and
    the targeting errors e_n, the vectors from the bull's-eye to each hit --
    the sequence {e_1, e_2, e_3, ...} whose dynamics the text asks about.
The hit positions are illustrative, not data.
"""
from _figstyle import *
from _scene3d import use_cm_fonts
from matplotlib.patches import Circle, Arc, Polygon, Rectangle

use_cm_fonts()
INK = "0.12"
HITS = np.array([(0.62, 0.38), (-0.42, -0.30), (0.25, -0.22), (-0.10, 0.12)])

fig = plt.figure(figsize=(6.0, 2.3))
ax = fig.add_axes([0.0, 0.0, 0.66, 1.0])
bx = fig.add_axes([0.70, 0.04, 0.29, 0.92])

# ---------------------------------------------------------------- (a) archer
# Clay-mannequin archer rendered by render_archer_blender.py (orthographic, so
# the image is placed in these axes by its world extent, in metres); the bow,
# arrow and trajectories are drawn here at the rendered grip and nock points.
import json
from PIL import Image
arch = Image.open(FIGDIR + "archer_render.png").convert("RGBA")
AA = json.load(open(FIGDIR + "archer_anchors.json"))
# one flat tone: the render supplies only the silhouette (its alpha channel)
alpha = np.asarray(arch)[:, :, 3].astype(float) / 255.0
flat = np.zeros(alpha.shape + (4,)); flat[..., :3] = 0.25; flat[..., 3] = alpha
ax.imshow(flat, extent=AA["extent"], interpolation="lanczos", zorder=3)
grip = np.array(AA["grip"]); draw = np.array(AA["nock"])

# recurve bow: limbs bend back toward the archer, tips recurve forward
s = np.linspace(-1, 1, 81)
bow_x = grip[0] + 0.05 - 0.20 * s**2 + 0.30 * np.clip(np.abs(s) - 0.85, 0, None)   # symmetric recurve tips
bow_y = draw[1] + 0.62 * s                               # centred on the arrow
w = 0.012 + 0.016 * (1 - np.abs(s))                       # thicker at the riser
ax.fill_betweenx(bow_y, bow_x - w, bow_x + w, color=COL[6], lw=0, zorder=5)
tips = [(bow_x[0], bow_y[0]), (bow_x[-1], bow_y[-1])]
ax.plot([tips[0][0], draw[0], tips[1][0]], [tips[0][1], draw[1], tips[1][1]],
        color="0.35", lw=0.6, zorder=2)
ax.annotate("", xy=(grip[0] + 0.32, draw[1]), xytext=(draw[0] - 0.02, draw[1]),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=7), zorder=6)
ax.plot([-0.5, 4.9], [0.0, 0.0], color="0.55", lw=0.6, zorder=0)            # ground

# target on its stand
tx, ty, tr = 4.55, 1.30, 0.42
ax.add_patch(Polygon([(tx - 0.05, ty - tr), (tx + 0.05, ty - tr), (tx + 0.05, ty + tr),
                      (tx - 0.05, ty + tr)], closed=True, fc="0.85", ec=INK, lw=0.8, zorder=3))
for f in (1.0, 0.66, 0.33):
    ax.plot([tx - 0.055, tx - 0.055], [ty - f * tr, ty + f * tr], color=COL[1] if f < 0.4 else
            "0.5", lw=1.4 if f < 0.4 else 0.9, zorder=4)
ax.plot([tx, tx - 0.25], [ty - tr, 0.0], color=INK, lw=0.9)
ax.plot([tx, tx + 0.25], [ty - tr, 0.0], color=INK, lw=0.9)

# trajectories of successive shots: shallow arcs ending at the hit heights
x0, y0 = grip[0] + 0.32, draw[1]
for n, (hx, hy) in enumerate(HITS, start=1):
    yh = ty + hy * tr
    xs = np.linspace(x0, tx - 0.06, 80)
    s = (xs - x0) / (xs[-1] - x0)
    ys = y0 + (yh - y0) * s + 0.32 * s * (1 - s)
    ax.plot(xs, ys, color=COL[0], lw=0.7, ls=(0, (4, 2.5)), alpha=0.35 + 0.15 * n, zorder=2)
ax.text(2.6, 2.02, r"shots $n = 1, 2, 3, \ldots$", fontsize=8, ha="center", color=COL[0])

ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-0.82, 5.0); ax.set_ylim(-0.15, 2.3)
ax.text(-0.80, 2.25, "(a)", fontsize=9, va="top")

# ---------------------------------------------------------------- (b) face
ring = ["0.97", "0.88", "0.97", "0.88"]
for k, r in enumerate((1.0, 0.75, 0.5, 0.25)):
    bx.add_patch(Circle((0, 0), r, fc=ring[k], ec="0.5", lw=0.6, zorder=1))
bx.add_patch(Circle((0, 0), 0.10, fc=COL[1], ec="0.3", lw=0.6, zorder=2))
for n, (hx, hy) in enumerate(HITS, start=1):
    bx.annotate("", xy=(hx, hy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=COL[0], lw=0.9, mutation_scale=7,
                                shrinkA=0, shrinkB=2), zorder=3)
    bx.add_patch(Circle((hx, hy), 0.045, fc=INK, ec="white", lw=0.5, zorder=4))
    off = np.array([hx, hy]) / np.hypot(hx, hy) * 0.15
    bx.text(hx + off[0], hy + off[1], f"$e_{n}$", fontsize=9, ha="center", va="center",
            zorder=5)
bx.set_aspect("equal"); bx.axis("off")
bx.set_xlim(-1.08, 1.08); bx.set_ylim(-1.08, 1.08)
bx.text(-1.05, 1.05, "(b)", fontsize=9, va="top")

fig.savefig(FIGDIR + "target.pdf", dpi=400)   # keeps the archer render sharp
print("wrote target.pdf")
