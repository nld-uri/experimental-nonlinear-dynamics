"""C2_LY_Lorenz.pdf and C2_LY_Duffing.pdf -- correlation sum and short-time
divergence before and after filtering (Chapter 14, figures fig_L_CORR and
fig_D_C2_Lyap).

Top row: log10 C(eps) against log10 eps at 5 (left) and 20 (right) per cent
noise, for the clean, the noisy, the POD-filtered and the SOD-filtered record
(c2r of Chapter dimension, whole record, Theiler window d*tau, 60 scales).
Bottom row: the mean logarithmic divergence ln delta_n against step number
(lyapunov of Chapter lyap, NF fiducial points, one neighbour, NSTEP steps).

Embedding for the estimators: Lorenz d = 3, tau = 12 (the setting Table TB
uses); Duffing d = 4, tau = 7, since its false-nearest-neighbour fraction
vanishes only at D = 4. Records and filtered records from _ch14data.py.

Running it prints the slope of each correlation sum over the window the chapter
uses for Table TB, and of each divergence curve over steps WLY.

    python3 mkfig_nr_c2ly.py
"""
import numpy as np
from _figstyle import plt, COL, FIGDIR, panel
from c2r import c2r
from lyapunov import lyapunov
import _ch14data as C

NC, NF, NSTEP = None, 4000, 300
DE = {"Lorenz": 3, "Duffing": 4}
WC2 = {"Lorenz": (-3.05, -1.58), "Duffing": (-3.05, -1.58)}   # ln eps
WLY = {"Lorenz": (236, 275), "Duffing": (60, 140)}            # steps
LEV = (0.05, 0.20)


def curves(name, x):
    d, tau = DE[name], C.TAU[name]
    lnr, lnc = c2r(x, d, tau, n=NC, r=60, rng=np.random.default_rng(0))
    ns, lnd = lyapunov(x, d, tau, NF, 1, NSTEP, rng=np.random.default_rng(0))
    return lnr, lnc, ns, lnd


def slopes(name, lnr, lnc, ns, lnd):
    a, b = WC2[name]
    k = (lnr >= a) & (lnr <= b) & np.isfinite(lnc)
    d2 = np.polyfit(lnr[k], lnc[k], 1)[0]
    a, b = WLY[name]
    k = (ns >= a) & (ns <= b)
    lam = np.polyfit(ns[k], lnd[k], 1)[0]/C.DT[name]
    return d2, lam


def one(name, out):
    fig, ax = plt.subplots(2, 2, figsize=(5.6, 4.4))
    xc = C.clean(name)
    base = curves(name, xc)
    print("\n%s   window ln eps %s, steps %s" % (name, WC2[name], WLY[name]))
    print("   clean            D2 = %6.3f   lambda_1 = %6.3f"
          % slopes(name, *base))
    for c, p in enumerate(LEV):
        sets = [("clean", xc, "k", "-"),
                (r"noisy", C.noisy(name, p), COL[1], ":"),
                ("POD", C.final(name, p, "POD"), COL[0], "-."),
                ("SOD", C.final(name, p, "SOD"), COL[4], "--")]
        for lb, x, col, ls in sets:
            lnr, lnc, ns, lnd = base if lb == "clean" else curves(name, x)
            d2, lam = slopes(name, lnr, lnc, ns, lnd)
            if lb != "clean":
                print("   %4.0f %% %-8s  D2 = %6.3f   lambda_1 = %6.3f"
                      % (100*p, lb, d2, lam))
            m = np.isfinite(lnc)
            ax[0, c].plot(lnr[m]/np.log(10), lnc[m]/np.log(10), ls,
                          color=col, lw=0.9, label=lb)
            ax[1, c].plot(ns, lnd, ls, color=col, lw=0.9, label=lb)
        ax[0, c].set_xlabel(r"$\log_{10}\epsilon$")
        ax[0, c].set_ylabel(r"$\log_{10} C(\epsilon)$")
        ax[1, c].set_xlabel("$n$ (time steps)")
        ax[1, c].set_ylabel(r"$\ln\delta_n$")
        ax[0, c].legend(loc="lower right", handlelength=1.6)
        ax[1, c].legend(loc="lower right", handlelength=1.6)
    for k, a in enumerate(ax.ravel()):
        panel(a, "abcd"[k], x=-0.20)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGDIR + out)
    plt.close(fig)


if __name__ == "__main__":
    one("Lorenz", "C2_LY_Lorenz.pdf")
    one("Duffing", "C2_LY_Duffing.pdf")
