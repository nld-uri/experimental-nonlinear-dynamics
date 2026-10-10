"""specest_mt.pdf: multitaper versus Welch on the line-plus-continuum record
of Section 6.x, and the harmonic F-test on the two-well record.

(a) one record, N = 1024: Welch (Hann, L = 256, K = 7) and adaptive
    multitaper (NW = 4, K = 7) against the exact continuous spectrum;
(b) mean of estimate/truth beside the strong line, 600 records;
(c) harmonic F statistic of the record in (a), zero-padded 8x;
(d) harmonic F statistic of the full two-well record (96,000 samples).
All numbers quoted in the text are printed by chk_multitaper.py.
"""
import numpy as np
from scipy.stats import f as fdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotprefs as pp
from _sestyle import ar2, ar2_psd
from psdwelch import psdwelch
from psdmt import psdmt, mtlines
from _twowell import twowell, FS, FD

FIGDIR = "../figures/"
pp.apply_preferences(sizes=(9, 9, 8))
C = pp.MATLAB
CW, CW2, CM, CR = C["blue"], C["plum"], C["orange"], C["darkgreen"]

N, NW, K, RHO, F1 = 1024, 4.0, 7, 0.05, 0.13
AMPS = (10.0, 0.5, 0.1)
LINES = [(h*F1, A) for h, A in zip((1, 2, 3), AMPS)]
W = NW/N
n = np.arange(N)


def record(rng):
    x = ar2(RHO, N, rng)
    for f0, A in LINES:
        x = x + A*np.cos(2*np.pi*f0*n + rng.uniform(0, 2*np.pi))
    return x


fig, ax = plt.subplots(2, 2, figsize=(6, 4.6))

# (a) one record -----------------------------------------------------------
x = record(np.random.default_rng(2026))
fw, gw = psdwelch(x, 1.0, 256, 128, "hann")
fm, gm = psdmt(x, 1.0, NW, K)
ff = np.linspace(0, 0.5, 2001)
a = ax[0, 0]
a.semilogy(fw, gw, color=CW, lw=0.9, label="Welch, $L=256$")
a.semilogy(fm, gm, color=CM, lw=0.9, label="multitaper, $TW=4$")
a.semilogy(ff, 2*ar2_psd(RHO, ff), color="k", lw=0.7, ls="--", label="continuum $G_c(f)$")
for f0, _ in LINES:
    a.plot(f0, 3e5, "v", color="k", ms=3.5)
a.set_xlim(0, 0.5); a.set_ylim(1e-2, 1e6)
a.set_xlabel("$f$ (Hz)"); a.set_ylabel(r"$\hat G_{xx}(f)$ ($x^2$/Hz)")
a.legend(loc="center right", fontsize=7, handlelength=1.6)

# (b) mean ratio beside the strong line ---------------------------------------
rng = np.random.default_rng(7)
acc = {"w256": [], "w512": [], "mt": [], "mtc": []}
for _ in range(600):
    x = record(rng)
    f1_, g = psdwelch(x, 1.0, 256, 128, "hann"); acc["w256"].append(g)
    f2_, g = psdwelch(x, 1.0, 512, 256, "hann"); acc["w512"].append(g)
    f3_, g = psdmt(x, 1.0, NW, K); acc["mt"].append(g)
    xc, _ = mtlines(x, 1.0, NW, K)
    f4_, g = psdmt(xc, 1.0, NW, K); acc["mtc"].append(g)
a = ax[0, 1]
for key, fq, col, lab, ls in (("w256", f1_, CW, "Welch, $L=256$", "-"),
                              ("w512", f2_, CW2, "Welch, $L=512$", "-"),
                              ("mt", f3_, CM, "MT", "-"),
                              ("mtc", f4_, CR, "MT, lines removed", "-")):
    r = np.mean(acc[key], 0)/(2*ar2_psd(RHO, fq))
    m = np.abs(fq - F1) < 11*W
    a.semilogy((fq[m] - F1)/W, r[m], color=col, lw=1.0, ls=ls, marker=".", ms=2.5, label=lab)
a.axvspan(-1, 1, color=C["lightgray"], alpha=0.5, lw=0)
a.axhline(1, color="k", lw=0.5)
a.set_xlim(-11, 11); a.set_ylim(0.7, 30)
a.set_xlabel(r"$(f - f_1)/W$"); a.set_ylabel(r"$\mathrm{E}[\hat G_{xx}]/G_c$")
a.legend(loc="upper left", fontsize=7, handlelength=1.2, borderaxespad=0.3)

# (c) F statistic of the record in (a) ----------------------------------------
x = record(np.random.default_rng(2026))
f, S = psdmt(x, 1.0, NW, K, full=True, nfft=8*N)
thr = fdist.ppf(1 - 1/N, 2, 2*K - 2)
a = ax[1, 0]
a.semilogy(f, S["F"], color=CM, lw=0.6)
a.axhline(thr, color="k", lw=0.7, ls="--")
for f0, _ in LINES:
    a.plot(f0, 2e3, "v", color="k", ms=3.5)
a.set_xlim(0, 0.5); a.set_ylim(1e-2, 1e4)
a.set_xlabel("$f$ (Hz)"); a.set_ylabel("$F(f)$")

# (d) F statistic of the two-well record --------------------------------------
xt = twowell(detrend=True)
f, S = psdmt(xt, FS, NW, K, full=True)
thr_t = fdist.ppf(1 - 1/xt.size, 2, 2*K - 2)
a = ax[1, 1]
a.semilogy(f, S["F"], color=CM, lw=0.4, rasterized=True)
a.axhline(thr_t, color="k", lw=0.7, ls="--")
for h in (1, 2, 3, 4):
    a.text(h*FD, 1.5e4, "%d" % (h*FD), ha="center", va="bottom", fontsize=7)
a.set_xlim(0, 80); a.set_ylim(1e-2, 1e5)
a.set_xlabel("$f$ (Hz)"); a.set_ylabel("$F(f)$")

for a, t in zip(ax.flat, ("(a)", "(b)", "(c)", "(d)")):
    a.text(0.02, 0.04, t, transform=a.transAxes)
pp.latexify(fig)
fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.8)
fig.savefig(FIGDIR + "specest_mt.pdf", dpi=300)
print("saved", FIGDIR + "specest_mt.pdf")
