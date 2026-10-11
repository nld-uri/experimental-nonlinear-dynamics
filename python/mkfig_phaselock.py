"""phaselock.pdf (label `phaselock`) -- the trough below the scaling range of the
two-well record is produced by sampling locked to the forcing, not by sparse
sampling.

Model: the book's running two-well Duffing oscillator,
    x'' + 0.25 x' - x + x^3 = 0.3 cos t,
integrated noise-free (solve_ivp, rtol 1e-9, atol 1e-10, max_step 0.05) from
(x, x') = (-0.6708, -0.3693); 200 forcing periods discarded, then 6000 forcing
periods kept, so every record has the same physical duration. Three samplings of
the same trajectory:
    locked 16 per period      (the rate of the measured record)
    locked 64 per period
    unlocked 16.618 per period (16 + golden-ratio fraction: no sample ever
                                repeats a forcing phase)
Reconstruction: d = 5, delay and Theiler window fixed in physical time
(tau = 5 and w = 25 samples at 16 per period, scaled with the rate).
Correlation sum: record normalized to unit variance, Euclidean norm, NREF random
reference points against all points, ln(eps/sigma) on the book's 76-point grid.

Also shown, for comparison: the measured two-well record (_twowell.py), first
30,000 reconstructed points, d = 5, tau = 5, w = 25, locked at 16 per period.

(a) local slope d ln C / d ln eps.
(b) fraction of pairs closer than eps whose samples were taken at exactly the
    same forcing phase, i.e. (i-j) mod npp == 0. Under unlocked sampling no two
    samples share a phase, so that fraction is identically zero and not drawn.
    Dotted lines: the chance levels 1/16 and 1/64.

Printed: local slopes and same-phase fractions at selected scales.
Runtime about seven minutes on two CPUs.
"""
import time
from _figstyle import *
from scipy.integrate import solve_ivp
from _twowell import twowell

t0 = time.time()
LNR = np.linspace(-6.5, 1.0, 76)
R2 = np.exp(2*LNR)
NREF = 2000
T = 2*np.pi
NCYC, NTR = 6000, 200
CASES = [("model, locked 16", 16.0),
         ("model, locked 64", 64.0),
         ("model, unlocked 16.618", 16.0 + (np.sqrt(5) - 1)/2)]

grids = [np.arange(NTR*T, (NTR + NCYC)*T, T/v) for _, v in CASES]
allt = np.unique(np.concatenate(grids))
sol = solve_ivp(lambda t, u: [u[1], -0.25*u[1] + u[0] - u[0]**3 + 0.3*np.cos(t)],
                [0, allt[-1] + 1e-6], [-0.6708, -0.3693], t_eval=allt,
                rtol=1e-9, atol=1e-10, max_step=0.05)
print("integration %.0f s" % (time.time() - t0), flush=True)


def corrsum(x, ph, d, tau, w, hw, seed=3):
    """ln C on LNR, and the fraction of close pairs with forcing-phase
    difference below hw (radians, wrapped)."""
    x = (x - x.mean())/x.std()
    n = x.size - (d - 1)*tau
    y = np.stack([x[i*tau:i*tau + n] for i in range(d)], axis=1)
    ph = ph[:n]
    refs = np.sort(np.random.default_rng(seed).choice(n, NREF, replace=False))
    idx = np.arange(n)
    edges = np.concatenate([[0.0], R2])
    cnt = np.zeros(R2.size); cs = np.zeros(R2.size); tot = 0
    ch = max(4, int(1.5e7/(n*d)))
    for k in range(0, NREF, ch):
        rr = refs[k:k + ch]
        D2 = ((y[None, :, :] - y[rr, None, :])**2).sum(-1)
        ok = np.abs(idx[None, :] - rr[:, None]) > w
        dph = np.abs(np.angle(np.exp(1j*(ph[None, :] - ph[rr, None]))))
        dd = D2[ok]; tot += dd.size
        cnt += np.cumsum(np.histogram(dd, bins=edges)[0])
        cs += np.cumsum(np.histogram(D2[ok & (dph < hw)], bins=edges)[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(cnt/tot), cs/cnt


res = []
for (name, v), g in zip(CASES, grids):
    x = np.interp(g, sol.t, sol.y[0])
    tau, w = int(round(5*v/16)), int(round(25*v/16))
    lnc, frac = corrsum(x, g % T, 5, tau, w, hw=1e-6)
    ok = np.isfinite(lnc)
    s = np.full(LNR.size, np.nan)
    s[ok] = np.gradient(lnc[ok], LNR[ok])
    res.append((name, s, frac))
    out = "  ".join("%.2f:%.2f/%.3f" % (LNR[j], s[j], frac[j])
                    for j in [np.argmin(abs(LNR - e)) for e in
                              (-0.9, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0)])
    print("%-28s N=%d tau=%d w=%d  ln(eps/sigma): slope/same-phase  %s"
          % (name, x.size, tau, w, out), flush=True)
    m = (LNR >= -1.45) & (LNR <= -0.35)
    print("%-28s mean local slope over [-1.45,-0.35]: %.2f; over [-5,-2.5]: %.2f"
          % ("", np.nanmean(s[m]), np.nanmean(s[(LNR >= -5) & (LNR <= -2.5)])))

xr = twowell()[:30020]
lnc, frac = corrsum(xr, 2*np.pi*(np.arange(xr.size) % 16)/16, 5, 5, 25, hw=1e-6)
ok = np.isfinite(lnc)
s = np.full(LNR.size, np.nan)
s[ok] = np.gradient(lnc[ok], LNR[ok])
rec = ("record, locked 16", s, frac)
print("%-28s ln(eps/sigma): slope/same-phase  %s" % (rec[0], "  ".join(
    "%.2f:%.2f/%.3f" % (LNR[j], s[j], frac[j]) for j in
    [np.argmin(abs(LNR - e)) for e in (-0.9, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0)])))

fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.8))
sty = [dict(color=COL[0], ls="-"), dict(color=COL[1], ls="--"), dict(color="#055432", ls="-.")]  # dark green beside orange (figure style)
axs[0].plot(LNR, rec[1], color="0.55", lw=0.8, label=rec[0])
axs[1].plot(LNR, rec[2], color="0.55", lw=0.8)
for (name, s, frac), st in zip(res, sty):
    axs[0].plot(LNR, s, marker=".", ms=2.5, lw=0.9, label=name, **st)
    if "unlocked" not in name:
        axs[1].plot(LNR, frac, marker=".", ms=2.5, lw=0.9, **st)
for ax in axs:
    ax.axvspan(-1.45, -0.35, color="0.5", alpha=0.18, lw=0)
    ax.set_xlabel(r"$\ln (\epsilon/\sigma)$")
    ax.set_xlim(-5.5, 0.5)
axs[0].set_ylabel(r"$\mathrm{d}\ln\hat C/\mathrm{d}\ln\epsilon$")
axs[0].set_ylim(0.5, 3.0)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, fontsize=7, handlelength=2.2,
           columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
for c in (1/16, 1/64):
    axs[1].axhline(c, color="k", ls=":", lw=0.7)
axs[1].set_ylabel("same-phase fraction")
axs[1].set_ylim(0, 1.05)
panel(axs[0], "a", x=-0.20)
panel(axs[1], "b", x=-0.20)
fig.tight_layout(pad=0.4, w_pad=1.4, rect=(0, 0, 1, 0.92))
fig.savefig(FIGDIR + "phaselock.pdf")
print("wrote phaselock.pdf, total %.0f s" % (time.time() - t0))
