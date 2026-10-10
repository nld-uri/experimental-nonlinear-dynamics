"""Lorents_PS_horiz_11_64.pdf and Duffing_PS_horiz_11_64.pdf -- the grid of
reconstructed phase portraits (Chapter 14, figures fig_L_PS and fig_D_PS).

Columns: the five noise levels. Rows: the noisy record, the record after ten
iterations of the local projective filter (podfilt, 64-point clouds), and after
ten iterations of the smooth projective filter (sodfilt, bundles of nine
11-point strands, tent weighting), both at d = 6, q = 3. Delay portraits
x_i against x_{i+tau} with tau = 12 (Lorenz) and tau = 7 (Duffing).

All records are standardized and every panel of the grid shares one set of axes,
fixed at +-3 sigma_x. That frames the clean attractor rather than the noise, so
the filtered panels are readable and comparable with each other; the noisiest
records run off the frame, which is the honest cost.

    python3 mkfig_nr_psgrid.py
"""
import numpy as np
from _figstyle import plt, COL, FIGDIR
import _ch14data as C

ROWS = [("noisy", None), ("POD filtered", "POD"), ("SOD filtered", "SOD")]
SMALL = 8          # one size for every label in these two figures
LIM = 3.0          # axis limit in units of sigma_x, the same in all 15 panels


def one(name, out):
    tau = C.TAU[name]
    fig, ax = plt.subplots(3, 5, figsize=(6.0, 3.8))
    clip = np.zeros((3, 5))
    for i, (lab, meth) in enumerate(ROWS):
        for j, p in enumerate(C.NOISE):
            x = C.noisy(name, p) if meth is None else C.final(name, p, meth)
            ax[i, j].plot(x[:-tau], x[tau:], "-", color=COL[0], lw=0.12)
            if i == 0:
                ax[i, j].set_title(r"$%d\,\%%$ noise" % round(100*p),
                                   fontsize=SMALL, pad=3)
            clip[i, j] = 100*np.mean(np.abs(x) > LIM)   # per cent outside the frame
    for i in range(3):
        for j in range(5):
            ax[i, j].set_xlim(-LIM, LIM)
            ax[i, j].set_ylim(-LIM, LIM)
            ax[i, j].set_aspect("equal")
            ax[i, j].tick_params(labelsize=SMALL)
            if j:
                ax[i, j].set_yticklabels([])
            if i < 2:
                ax[i, j].set_xticklabels([])

        ax[i, 0].set_ylabel("%s\n$x_{i+\\tau}$" % ROWS[i][0], fontsize=SMALL)
    for j in range(5):
        ax[2, j].set_xlabel("$x_i$", fontsize=SMALL)
    fig.tight_layout(pad=0.2, h_pad=0.4, w_pad=0.2)
    fig.savefig(FIGDIR + out)
    plt.close(fig)
    print(name, "axis limit +-%.1f sigma_x; per cent of points outside the "
          "frame, by row:" % LIM)
    for i in range(3):
        print("   %-13s %s" % (ROWS[i][0], np.round(clip[i], 2)))


if __name__ == "__main__":
    one("Lorenz", "Lorents_PS_horiz_11_64.pdf")
    one("Duffing", "Duffing_PS_horiz_11_64.pdf")
