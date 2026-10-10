"""fnn.pdf (label fnn): the idea of false nearest neighbors on a computed closed curve.
The curve is r(s) = (cos s, sin 2s, 0.7 sin s + 0.2 cos 2s + 0.4), s in [0, 2 pi) (col1), whose
projection onto the (x, y) plane (col2, drawn at z = -1.4) is a figure-eight that crosses
itself at the origin: the points A = r(pi/2) and B = r(3 pi/2), 1.4 apart in three
dimensions, project onto the same point A' = B' = (0, 0). They are false neighbors of the
projection. C = r(s_C) and D = r(s_C + 0.15), s_C = 0.6, are true neighbors: close on the
curve, close in the projection (C', D'). The thin dotted lines are the projection rays.
The grey dashed curves are the projections onto the two vertical walls."""
import numpy as np
from _figstyle import *

s = np.linspace(0, 2*np.pi, 800)
def r(s): return np.array([np.cos(s), np.sin(2*s), 0.7*np.sin(s) + 0.2*np.cos(2*s) + 0.4])
R = r(s); Z0 = -1.4
sC = 0.6; pts = {"A": r(np.pi/2), "B": r(3*np.pi/2), "C": r(sC), "D": r(sC + 0.15)}
print("|A-B| = %.3f, |A'-B'| = %.1e; |C-D| = %.3f, |C'-D'| = %.3f" % (
    np.linalg.norm(pts["A"]-pts["B"]), np.linalg.norm(pts["A"][:2]-pts["B"][:2]),
    np.linalg.norm(pts["C"]-pts["D"]), np.linalg.norm(pts["C"][:2]-pts["D"][:2])))

fig = plt.figure(figsize=(4.2, 3.6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(R[0], R[1], R[2], color=COL[0], lw=1.2)
ax.plot(R[0], R[1], Z0*np.ones_like(s), color=COL[1], lw=1.2)
ax.plot(R[0], 1.3*np.ones_like(s), R[2], color="0.5", lw=0.7, ls="--")
ax.plot(-1.3*np.ones_like(s), R[1], R[2], color="0.5", lw=0.7, ls="--")
for name, p in pts.items():
    ax.plot([p[0]], [p[1]], [p[2]], "o", color=COL[0], ms=3.5, mfc="w", mew=1.0)
    ax.plot([p[0]], [p[1]], [Z0], "o", color=COL[1], ms=3.5, mfc="w", mew=1.0)
    ax.plot([p[0], p[0]], [p[1], p[1]], [Z0, p[2]], ":", color="k", lw=0.5)
off = {"A": (-0.12, 0, 0.12), "B": (0.12, 0, -0.18), "C": (-0.15, 0, 0.1), "D": (0.1, 0, 0.1)}
for name, p in pts.items():
    o = off[name]; ax.text(p[0]+o[0], p[1]+o[1], p[2]+o[2], "$%s$" % name, fontsize=9)
ax.text(0.15, 0.0, Z0 + 0.03, "$A' = B'$", fontsize=9, color=COL[1])
ax.text(pts["C"][0]-0.1, pts["C"][1], Z0 - 0.3, "$C'$", fontsize=9, color=COL[1])
ax.text(pts["D"][0]+0.12, pts["D"][1], Z0 - 0.3, "$D'$", fontsize=9, color=COL[1])
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(Z0, 1.3)
ax.set_xlabel("$x$", labelpad=-6); ax.set_ylabel("$y$", labelpad=-6); ax.set_zlabel("$z$", labelpad=-6)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.view_init(elev=25, azim=-38)
ax.set_box_aspect((1, 1, 0.9), zoom=1.15)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
fig.savefig(FIGDIR + "fnn.pdf")
