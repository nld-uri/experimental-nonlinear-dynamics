"""Peanocurve.pdf -- the first three iterations of Peano's space-filling curve
(Chapter 11, Fig. 'Peano'). Replaces an unscripted figure of unknown origin.

Construction (Peano 1890): the iteration-k curve visits the 3^k x 3^k cell
centres of the unit square. It is the 3 x 3 serpentine of nine copies of the
iteration-(k-1) curve, traversed column by column (up, down, up); the copy in
block (column c, row r) is reflected in x when r is odd and in y when c is odd,
which makes consecutive copies join end to start. The check at the bottom
asserts that every step moves exactly one cell, horizontally or vertically,
and that every cell is visited once.
"""
from _figstyle import *


def peano(k):
    """Cell-centre path of the iteration-k Peano curve, integer coordinates."""
    if k == 0:
        return np.zeros((1, 2), dtype=int)
    p = peano(k - 1)
    n = 3 ** (k - 1)
    out = []
    for c in range(3):
        rows = range(3) if c % 2 == 0 else range(2, -1, -1)
        for r in rows:
            q = p.copy()
            if r % 2:
                q[:, 0] = n - 1 - q[:, 0]
            if c % 2:
                q[:, 1] = n - 1 - q[:, 1]
            out.append(q + np.array([c * n, r * n]))
    return np.vstack(out)


fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.0))
for k, ax in zip((1, 2, 3), axs):
    p = peano(k)
    n = 3 ** k
    steps = np.abs(np.diff(p, axis=0)).sum(axis=1)
    assert np.all(steps == 1), "non-adjacent step"
    assert len({tuple(v) for v in p}) == n * n, "cell visited twice"
    xy = (p + 0.5) / n
    ax.plot(xy[:, 0], xy[:, 1], color=COL[0], lw=(1.2, 0.9, 0.6)[k - 1],
            solid_joinstyle="miter")
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="0.75", lw=0.4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
fig.tight_layout(pad=0.2, w_pad=1.5)
fig.savefig(FIGDIR + "Peanocurve.pdf")
print("wrote Peanocurve.pdf; checks passed for k = 1, 2, 3")
