"""Lorenz_in_out.pdf and Duffing_in_out.pdf -- input and output of the two
filters (Chapter 14, figures fig_Lyap and fig_Duff).

(a) clean delay portrait, (b) portrait of the added noise, (c) one-sided power
spectral densities of the clean record and of the five noisy ones, (d) the
POD-filtered and (e) the SOD-filtered portrait of the 5 per cent record after
ten iterations, (f) the spectra of the clean record and of the four filtered
ones at 5 and 20 per cent noise.

Records, delays and filters from _ch14data.py (N = 20 000, d = 6, q = 3, ten
iterations, tent-weighted bundles of nine 11-point strands for SOD, 64-point
neighbour clouds for POD). All records are standardized, so the axes are in
units of the clean record's standard deviation and the spectra in dB per Hz
relative to sigma_x^2/Hz. Welch estimate, 2048-point Hann segments, half
overlap (psdwelch of Chapter specest).

Running it prints the median spectral floor of every curve of panel (f) over
the top half of the band, which is the number the text quotes.

    python3 mkfig_nr_inout.py
"""
import numpy as np
from _figstyle import plt, COL, FIGDIR, panel
from psdwelch import psdwelch
import _ch14data as C

NSEG = 2048


def portrait(ax, x, tau, lab):
    ax.plot(x[:-tau], x[tau:], "-", color=COL[0], lw=0.15)
    ax.set_xlabel("$%s_i$" % lab)
    ax.set_ylabel("$%s_{i+\\tau}$" % lab)


def spec(x, fs):
    f, G = psdwelch(x, fs, NSEG, NSEG//2, "hann")
    return f, 10*np.log10(np.maximum(G, 1e-30))


SMALL = 8          # axis labels, tick labels and panel letters
LEGEND = 6         # the two spectrum legends; smaller so they clear the curves


def one(name, out, width):
    tau, dt = C.TAU[name], C.DT[name]
    fs = 1.0/dt
    xc = C.clean(name)
    fig, ax = plt.subplots(2, 3, figsize=(width, width*0.62))
    floors = {}

    portrait(ax[0, 0], xc, tau, "x")
    portrait(ax[0, 1], C.noisy(name, 0.05) - xc, tau, "n")
    ax[0, 1].set_xlabel("$n_i$")
    ax[0, 1].set_ylabel(r"$n_{i+\tau}$")

    f, g = spec(xc, fs)
    ax[0, 2].plot(f, g, color="k", lw=0.7, label="clean")
    for j, p in enumerate(C.NOISE):
        f, g = spec(C.noisy(name, p), fs)
        ax[0, 2].plot(f, g, color=COL[j], lw=0.6,
                      label=r"$%d\,\%%$" % round(100*p))
        floors["%d %% noisy" % round(100*p)] = np.median(g[f > f[-1]/2])

    portrait(ax[1, 0], C.final(name, 0.05, "POD"), tau, r"\bar{x}")
    portrait(ax[1, 1], C.final(name, 0.05, "SOD"), tau, r"\bar{x}")

    f, g = spec(xc, fs)
    ax[1, 2].plot(f, g, color="k", lw=0.7, label="clean")
    floors["clean"] = np.median(g[f > f[-1]/2])
    for j, (p, meth) in enumerate([(0.05, "POD"), (0.05, "SOD"),
                                   (0.20, "POD"), (0.20, "SOD")]):
        f, g = spec(C.final(name, p, meth), fs)
        k = "%d %% %s" % (round(100*p), meth)
        floors[k] = np.median(g[f > f[-1]/2])
        ax[1, 2].plot(f, g, color=COL[j], lw=0.6, label=k)
    for a in (ax[0, 2], ax[1, 2]):
        a.set_xlabel("frequency (Hz)")
        a.set_ylabel("PSD (dB/Hz)")
        a.set_xlim(0, fs/2)
        # headroom above the data so the legend sits clear of every curve
        a.set_ylim(-110, 55)
        a.set_yticks([-100, -75, -50, -25, 0, 25])
        a.legend(loc="upper right", handlelength=1.0, labelspacing=0.15,
                 ncol=2, columnspacing=0.6, fontsize=LEGEND, frameon=False,
                 borderaxespad=0.2)
    # every label at SMALL: axis labels, tick labels and panel letters. The
    # figure is drawn at the width it is included at, so nothing is rescaled
    # by LaTeX and SMALL is the size on the page.
    for k, a in enumerate(ax.ravel()):
        a.xaxis.label.set_size(SMALL)
        a.yaxis.label.set_size(SMALL)
        a.tick_params(labelsize=SMALL)
        panel(a, "abcdef"[k], x=-0.24, y=1.01)
        a.texts[-1].set_fontsize(SMALL)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGDIR + out)
    plt.close(fig)
    return floors


if __name__ == "__main__":
    for name, out, w in (("Lorenz", "Lorenz_in_out.pdf", 6.0),
                         ("Duffing", "Duffing_in_out.pdf", 6.0)):
        fl = one(name, out, w)
        print(name, " median PSD over the upper half of the band (dB/Hz)")
        for k, v in fl.items():
            print("   %-12s %8.1f" % (k, v))
