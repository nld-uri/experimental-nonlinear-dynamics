"""Experiment2ElMag.pdf -- the two-magnet fatigue rig of Section 17.4
(Fig. 'expt'): the Blender render of render_rig_blender.py, cropped, set on
white, and labelled in Computer Modern.

Inputs (written by render_rig_blender.py, which needs Blender's Python
module): ../figures/rig_render.png (RGBA) and ../figures/rig_anchors.json
(label anchors in normalized image coordinates, origin bottom-left).
This script needs only matplotlib and Pillow, so the labels can be edited
without re-rendering.
"""
import json
from PIL import Image
from _figstyle import *
from _scene3d import use_cm_fonts

use_cm_fonts()

im = Image.open(FIGDIR + "rig_render.png").convert("RGBA")
A = json.load(open(FIGDIR + "rig_anchors.json"))
W0, H0 = im.size

# crop to the visible content (alpha above a small threshold) plus a margin
alpha = np.asarray(im.split()[3])
ys, xs = np.nonzero(alpha > 60)
m = int(0.015 * W0)
x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, W0)
y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, H0)
crop = im.crop((x0, y0, x1, y1))
bg = Image.new("RGBA", crop.size, "white"); bg.alpha_composite(crop)
img = np.asarray(bg.convert("RGB"))
w, h = crop.size


def px(key):
    """Anchor in cropped-image pixel coordinates (imshow, origin upper)."""
    ax_, ay_ = A[key]
    return ax_ * W0 - x0, (1 - ay_) * H0 - y0


FW = 4.8                                   # printed width, inches
IMW = 0.60                                 # image share of the width
fig = plt.figure(figsize=(FW, FW * IMW * h / w))
ax = fig.add_axes([0.21, 0.0, IMW, 1.0])
ax.imshow(img, interpolation="lanczos")
ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")

kw = dict(fontsize=8, va="center", zorder=5, annotation_clip=False,
          arrowprops=dict(arrowstyle="-", color="0.30", lw=0.5, shrinkA=1, shrinkB=1))


def lab(text, key, tx, ty, ha="left"):
    """Label at fractional position (tx, ty) of the image box."""
    ax.annotate(text, xy=px(key), xytext=(tx * w, ty * h), ha=ha, **kw)


lab("clamp", "clamp", 0.50, 0.04)
lab("notch", "notch", -0.06, 0.20, ha="right")
lab("stiffeners", "stiffeners", -0.06, 0.40, ha="right")
lab("electromagnets", "magnet", -0.06, 0.70, ha="right")
lab("laser vibrometers", "head2", 0.66, 0.06)
lab(r"supply, $v_1(t)$, $v_2(t)$", "supply", 0.72, 0.62)
for ch, note in (("CH2", "frame"), ("CH1", "beam")):
    x, y = px(ch)
    ax.text(x + 0.012 * w, y, f"{ch} ({note})", fontsize=7.5, ha="left", va="center",
            clip_on=False)

xa, ya = px("drive_a"); xb, yb = px("drive_b")
ax.annotate("", xy=(xb, yb), xytext=(xa, ya), annotation_clip=False,
            arrowprops=dict(arrowstyle="<|-|>", color="0.15", lw=0.9, mutation_scale=8,
                            shrinkA=0, shrinkB=0))
ax.text(xa - 0.02 * w, ya, "harmonic\ndrive, 10 Hz", fontsize=8, ha="right",
        va="center", clip_on=False, linespacing=1.1)

fig.savefig(FIGDIR + "Experiment2ElMag.pdf", dpi=300)
print("wrote Experiment2ElMag.pdf", img.shape)
