"""Figure nr_twowell.pdf (Chapter 14, Section sec:nr_eval).

Noise reduction applied to the measured two-well oscillator record of
Section sec:twowelldata, at the settings the chapter's own procedure box
selects from the record itself:

    tau = 5 samples   (first minimum of the average mutual information)
    q   = 3           (ceil(D2), D2 = 2.68)
    d   = 6           (about 2q)
    POD podfilt(x, 64, 5, 6, 3)              64-point neighbour clouds
    SOD sodfilt(x, 11, 5, 6, 3, w=1, r=1)    11-point strands, tent weight,
                                             no bundle (the bundle rule of the
                                             procedure box needs an input SNR,
                                             which this record cannot supply)

Ten passes of each.  There is no clean reference, so the panels show only
what a user without one can measure:

  (a) root-mean-square correction per pass, and the cumulative departure
      from the unfiltered record;
  (b) local slope of the correlation sum against ln r;
  (c) short-time divergence curve;
  (d) Welch spectrum of the unfiltered record and of the two residuals.

Everything is cached in one NPZ under the system temp directory.  A cold
run takes about twenty minutes; a warm one a few seconds.

Copyright by David Chelidze.  Written for the second edition.
"""
import os
import tempfile
import numpy as np
from _figstyle import COL, plt, FIGDIR, panel
from _twowell import twowell, NPP
from podfilt import podfilt
from sodfilt import sodfilt
from c2r import c2r
from lyapunov import lyapunov
from psdwelch import psdwelch

TAU, D, Q, ITER = 5, 6, 3, 10
RPOD, LSOD = 64, 11
NC2, PC2 = 30000, 20            # points and Theiler window for c2r
FITLO, FITHI = -1.45, -0.35     # ln r fitting band of Chapter 11
NLY, NSTEP, WLY = 5000, 80, 32  # lyapunov settings of Chapter 12
STEPLO, STEPHI = 10, 64         # divergence fitting window
DE = 4                          # embedding dimension of the record
SEED = 3
CACHE = os.path.join(tempfile.gettempdir(), "nr_twowell_cache.npz")


def _filters():
    x0 = twowell()
    x0 = (x0 - x0.mean())/x0.std()
    out = {"x0": x0}
    for tag, fn in [("POD", lambda z: podfilt(z, RPOD, TAU, D, Q)),
                    ("SOD", lambda z: sodfilt(z, LSOD, TAU, D, Q, w=1, r=1))]:
        z, rms, cum = x0.copy(), [], []
        for _ in range(ITER):
            zn = fn(z)
            rms.append(np.sqrt(np.mean((zn - z)**2)))
            z = zn
            cum.append(np.sqrt(np.mean((z - x0)**2)))
        out[tag + "_rms"] = np.array(rms)
        out[tag + "_cum"] = np.array(cum)
        out[tag] = z
    return out


def _diagnostics(rec):
    out = {}
    for k, y in rec.items():
        lnr, lnc = c2r(y, DE, TAU, n=NC2, p=PC2,
                       rng=np.random.default_rng(SEED))
        ns, lnd = lyapunov(y, DE, TAU, N=NLY, r=1, n=NSTEP, w=WLY,
                           rng=np.random.default_rng(SEED))
        out[k + "_lnr"], out[k + "_lnc"] = lnr, lnc
        out[k + "_ns"], out[k + "_lnd"] = ns, lnd
    return out


if os.path.exists(CACHE):
    C = dict(np.load(CACHE))
else:
    C = _filters()
    C.update(_diagnostics({"raw": C["x0"], "POD": C["POD"], "SOD": C["SOD"]}))
    np.savez(CACHE, **C)

x0 = C["x0"]
lab = {"raw": "unfiltered", "POD": "POD, 10 passes", "SOD": "SOD, 10 passes"}
col = {"raw": COL[0], "POD": COL[1], "SOD": COL[4]}

# ----------------------------------------------------------------- numbers
band = lambda v: (v >= FITLO) & (v <= FITHI)
for k in ("raw", "POD", "SOD"):
    lnr, lnc = C[k + "_lnr"], C[k + "_lnc"]
    m = band(lnr)
    D2 = np.polyfit(lnr[m], lnc[m], 1)[0]
    ns, lnd = C[k + "_ns"], C[k + "_lnd"]
    mm = (ns >= STEPLO) & (ns <= STEPHI)
    p = np.polyfit(ns[mm], lnd[mm], 1)
    r = lnd[mm] - np.polyval(p, ns[mm])
    print(f"{k:4s} D2 = {D2:.3f}   lambda1 = {p[0]*NPP:.3f} per forcing period"
          f"   ln delta_0 = {lnd[0]:+.3f}   rms fit residual = {r.std():.4f}")
for k in ("POD", "SOD"):
    res = C[k] - x0
    ac = np.correlate(res - res.mean(), res - res.mean(), "full")
    ac = ac[ac.size//2:]/ac.max()
    print(f"{k}: rms correction pass 1 {C[k+'_rms'][0]:.4f}, pass 10 "
          f"{C[k+'_rms'][-1]:.4f}; cumulative {C[k+'_cum'][-1]:.4f} sigma; "
          f"residual lag-one autocorrelation {ac[1]:.3f} against the white "
          f"band {2/np.sqrt(x0.size):.4f}")

# ----------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.6))
a, b, c, e = ax.ravel()

it = np.arange(1, ITER + 1)
for k in ("POD", "SOD"):
    a.semilogy(it, C[k + "_rms"], "o-", ms=3, color=col[k], label=k + ", per pass")
    a.semilogy(it, C[k + "_cum"], "s--", ms=3, color=col[k], mfc="none",
               label=k + ", cumulative")
a.set_xlabel("pass number")
a.set_ylabel(r"correction ($\sigma_s$)")
a.legend(loc="center right", fontsize=7)

for k in ("raw", "POD", "SOD"):
    lnr, lnc = C[k + "_lnr"], C[k + "_lnc"]
    sl = np.gradient(lnc, lnr)
    g = np.isfinite(sl)
    b.plot(lnr[g], sl[g], color=col[k], lw=1.0, label=lab[k])
b.axvspan(FITLO, FITHI, color="0.85", zorder=0)
b.set_xlim(-5.0, 0.5)
b.set_ylim(0.0, 4.0)
b.set_xlabel(r"$\ln r$ ($r$ in units of $\sigma_s$)")
b.set_ylabel(r"$\mathrm{d}\ln C_2/\mathrm{d}\ln r$")
b.legend(loc="lower right", fontsize=7)

for k in ("raw", "POD", "SOD"):
    ns, lnd = C[k + "_ns"], C[k + "_lnd"]
    c.plot(ns/NPP, lnd, color=col[k], lw=1.0, label=lab[k])
c.axvspan(STEPLO/NPP, STEPHI/NPP, color="0.85", zorder=0)
c.set_xlabel("time (forcing periods)")
c.set_ylabel(r"$\langle \ln \delta_n \rangle$")
c.legend(loc="lower right", fontsize=7)

f, G0 = psdwelch(x0, 1.0, 8192)
e.semilogy(f, G0, color=col["raw"], lw=0.7, label="unfiltered record")
for k in ("POD", "SOD"):
    fr, Gr = psdwelch(C[k] - x0, 1.0, 8192)
    e.semilogy(fr, Gr, color=col[k], lw=0.7, label=k + " residual")
for j, s in zip((1, 2, 3), ("$f_\\mathrm{d}$", "$2f_\\mathrm{d}$",
                            "$3f_\\mathrm{d}$")):
    e.axvline(j/NPP, color="0.6", lw=0.5, zorder=0)
    e.text(j/NPP, 3e1, s, fontsize=7, ha="center", va="bottom")
e.set_xlim(0, 0.5)
e.set_ylim(1e-9, 3e3)
e.set_xlabel(r"$f$ (cycles/sample; $f_\mathrm{d} = f_s/16$)")
e.set_ylabel(r"$G_{ss}(f)$ ($\sigma_s^2$/cycle)")
e.legend(loc="upper right", fontsize=7)

for u, L in zip((a, b, c, e), "abcd"):
    panel(u, L, x=-0.16)
fig.tight_layout()
fig.savefig(FIGDIR + "nr_twowell.pdf")
print("wrote figures/nr_twowell.pdf")
