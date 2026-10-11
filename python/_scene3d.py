"""Minimal 3D schematic renderer for the book's drawn figures (not a listing).

A pinhole camera and a painter's-algorithm scene: primitives are collected with
their depth and drawn far to near in matplotlib. Used by mkfig_humans.py
(Fig. 8.3) and mkfig_rig.py (Fig. 17.8). Coordinates: z up, metres or any
consistent unit. Faces are shaded by a fixed light; back faces are culled.
"""
import shutil, subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle


def use_cm_fonts():
    """LaTeX text when TeX with cm-super is available, else matplotlib's own
    Computer Modern (cmr10 + cm mathtext) -- the same face without TeX."""
    ok = False
    if shutil.which("kpsewhich"):
        r = subprocess.run(["kpsewhich", "type1ec.sty"], capture_output=True, text=True)
        ok = bool(r.stdout.strip())
    if ok:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["Computer Modern Roman"]})
    else:
        plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                             "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                             "axes.unicode_minus": False,
                             "axes.formatter.use_mathtext": True})


class Camera:
    """Pinhole camera looking at `target` from azimuth `az` (degrees, measured
    from the -y axis toward +x) and elevation `el`, at distance `dist`."""
    def __init__(self, target, az, el, dist, f=3.2):
        a, e = np.deg2rad(az), np.deg2rad(el)
        self.C = np.asarray(target, float) + dist * np.array(
            [np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
        fwd = np.asarray(target, float) - self.C
        self.fw = fwd / np.linalg.norm(fwd)
        r = np.cross(self.fw, [0, 0, 1.0]); self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.fw)
        self.F = f

    def depth(self, p):
        return (np.atleast_2d(p) - self.C) @ self.fw

    def __call__(self, p):
        d = np.atleast_2d(p) - self.C
        zc = d @ self.fw
        return np.c_[self.F * (d @ self.r) / zc, self.F * (d @ self.u) / zc]


LIGHT = np.array([0.4, -0.7, 0.6]) / np.linalg.norm([0.4, -0.7, 0.6])


def shade(color, normal, lo=0.62):
    n = np.asarray(normal, float); n = n / np.linalg.norm(n)
    return np.clip(np.asarray(color) * (lo + (1 - lo) * max(0.0, n @ LIGHT)), 0, 1)


class Scene:
    """Collects primitives with a depth, then draws them far to near."""
    def __init__(self, cam, lw_ref=3.6):
        self.cam, self.items, self.lw_ref = cam, [], lw_ref

    def visible(self, normal, point):
        return np.dot(normal, self.cam.C - np.asarray(point)) > 0

    def line(self, pts, color, lw, ls="-", bias=0.0, scale=True):
        pts = np.asarray(pts, float)
        for a, b in zip(pts[:-1], pts[1:]):
            self.items.append((self.cam.depth((a + b) / 2)[0] + bias, "line",
                               (np.vstack([a, b]), color, lw, ls, scale)))

    def ball(self, p, r, fc, ec="white", bias=-0.02):
        self.items.append((self.cam.depth(p)[0] + bias, "ball", (p, r, fc, ec)))

    def head(self, p, r, color, lw):
        self.items.append((self.cam.depth(p)[0], "head", (p, r, color, lw)))

    def poly(self, pts, fc, ec="0.45", lw=0.5, bias=0.0):
        pts = np.asarray(pts, float)
        self.items.append((self.cam.depth(pts.mean(0))[0] + bias, "poly",
                           (pts, fc, ec, lw)))

    def quad(self, pts, color, normal, ec="0.45", lw=0.5, bias=0.0):
        """Planar face with outward `normal`: culled if facing away, shaded."""
        pts = np.asarray(pts, float)
        if self.visible(normal, pts.mean(0)):
            self.poly(pts, shade(color, normal), ec, lw, bias)

    def box(self, lo, hi, color, bias=0.0, ec="0.45", lw=0.5):
        """Axis-aligned box."""
        x0, y0, z0 = lo; x1, y1, z1 = hi
        V = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
        faces = {(-1, 0, 0): [0, 1, 3, 2], (1, 0, 0): [4, 6, 7, 5],
                 (0, -1, 0): [0, 4, 5, 1], (0, 1, 0): [2, 3, 7, 6],
                 (0, 0, -1): [0, 2, 6, 4], (0, 0, 1): [1, 5, 7, 3]}
        for n, idx in faces.items():
            self.quad(V[idx], color, n, ec, lw, bias)

    def cylinder_z(self, base, r, h, color, n=40, ec="0.35", top=None):
        """Upright cylinder on `base` (centre of the bottom face)."""
        b = np.asarray(base, float)
        t = np.linspace(0, 2 * np.pi, n + 1)
        for a0, a1 in zip(t[:-1], t[1:]):
            am = 0.5 * (a0 + a1)
            nrm = np.array([np.cos(am), np.sin(am), 0.0])
            P = [b + [r * np.cos(a0), r * np.sin(a0), 0], b + [r * np.cos(a1), r * np.sin(a1), 0],
                 b + [r * np.cos(a1), r * np.sin(a1), h], b + [r * np.cos(a0), r * np.sin(a0), h]]
            if self.visible(nrm, np.mean(P, 0)):
                self.poly(P, shade(color, nrm), ec="none", lw=0)
        ring = np.c_[b[0] + r * np.cos(t), b[1] + r * np.sin(t), np.full(t.size, b[2] + h)]
        self.poly(ring, shade(top if top is not None else color, [0, 0, 1]), ec=ec, lw=0.5,
                  bias=-1e-3)

    def disc_y(self, centre, radius, fc, n=60):
        """Disc in a plane y = const."""
        t = np.linspace(0, 2 * np.pi, n)
        P = np.c_[centre[0] + radius * np.cos(t), np.full(n, centre[1]),
                  centre[2] + radius * np.sin(t)]
        self.poly(P, fc, ec="0.4", lw=0.6)

    def draw(self, ax):
        cam = self.cam
        for k, (d, kind, a) in enumerate(sorted(self.items, key=lambda t: -t[0])):
            z = k + 1
            if kind == "line":
                q = cam(a[0])
                lw = a[2] * self.lw_ref / d if a[4] else a[2]
                ax.plot(q[:, 0], q[:, 1], color=a[1], lw=lw, ls=a[3],
                        solid_capstyle="round", dash_capstyle="butt", zorder=z)
            elif kind == "poly":
                ax.add_patch(Polygon(cam(a[0]), closed=True, fc=a[1], ec=a[2],
                                     lw=a[3], zorder=z))
            else:
                p, r = a[0], a[1]
                q, rs = cam(p)[0], cam.F * r / d
                if kind == "ball":
                    ax.add_patch(Circle(q, rs, fc=a[2], ec=a[3], lw=0.5, zorder=z))
                else:
                    ax.add_patch(Circle(q, rs, fc="white", ec=a[2],
                                        lw=a[3] * self.lw_ref / d, zorder=z))


def ik(root, tip, l1, l2, bend=+1):
    """Middle joint of a two-link chain, solved in the x-z plane (knee,
    elbow); bend=+1 puts the joint ahead (+x) of the root-tip line."""
    root, tip = np.asarray(root, float), np.asarray(tip, float)
    dx, dz = tip[0] - root[0], tip[2] - root[2]
    L = min(np.hypot(dx, dz), l1 + l2 - 1e-6)
    a = np.arccos(np.clip((l1**2 + L**2 - l2**2) / (2 * l1 * L), -1, 1))
    ang = np.arctan2(dz, dx) + bend * a
    return np.array([root[0] + l1 * np.cos(ang), 0.5 * (root[1] + tip[1]),
                     root[2] + l1 * np.sin(ang)])


def label(ax, cam, text, p3, xytext, ha="left", fontsize=8):
    ax.annotate(text, xy=cam(p3)[0], xytext=xytext, ha=ha, va="center",
                fontsize=fontsize, zorder=10_000,
                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.5,
                                shrinkA=1, shrinkB=2))
