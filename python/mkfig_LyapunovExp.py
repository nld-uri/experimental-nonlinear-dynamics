"""LyapunovExp.pdf (label `Lyap`) -- fixed-mass divergence curves Delta_n from
`lyapunov` for the Lorenz (a-b) and Duffing (c-d) records, with the local slope
divided by dt against the reference exponent from the Jacobian (QR algorithm).

Records: the Chapter 10/11 records (_ch11data): Lorenz x, N = 20 000, dt = 0.02,
d = 3, tau = 8, w = 24; Duffing x, dt = 0.1, d = 4, tau = 14, w = 42, but with
N = 60 000 points (same integration, longer): the 20 000-point record gives a
local slope that decays from 0.15 to 0.04 without a plateau (see the report).
NFID fiducial points with r = 1 neighbour each, NSTEP steps ahead, seed 0. The fit
windows WIN were chosen where the local slope is flattest (after the reorientation
transient, before saturation) and are shaded; the least-squares slope over each
window, divided by dt, is printed with its standard error, together with the
spread of the estimate over the windows shifted by +-20 % of their length.
The reference exponents LAMREF come from the QR algorithm on the model Jacobians
(Lorenz 0.9079, eq. lorenzspec; Duffing 0.120 +- 0.005, see the report).
"""
from _figstyle import *
from _ch11data import record, tau_w, DEMB, DT
from _ch02data import duffing_record
from lyapunov import lyapunov

NFID, NSTEP = 4000, 400
NREC = {"Lorenz": 20000, "Duffing": 60000}   # the 20 000-point Duffing record shows no plateau
WIN = {"Lorenz": (60, 160), "Duffing": (80, 240)}
LAMREF = {"Lorenz": 0.9079, "Duffing": 0.120}

fig, axs = plt.subplots(2, 2, figsize=(6.0, 4.4))
for row, name in enumerate(["Lorenz", "Duffing"]):
    tau, w = tau_w(name); d = DEMB[name]; dt = DT[name]
    x = record(name) if NREC[name] == 20000 else duffing_record(NREC[name])[1]
    ns, lnd = lyapunov(x, d, tau, NFID, 1, NSTEP, w=w, rng=np.random.default_rng(0))
    lo, hi = WIN[name]
    def fit(lo, hi):
        m = (ns >= lo) & (ns <= hi)
        A = np.column_stack([ns[m], np.ones(m.sum())])
        coef = np.linalg.lstsq(A, lnd[m], rcond=None)[0]
        res = lnd[m] - A @ coef
        se = np.sqrt(res @ res/(m.sum() - 2)/np.sum((ns[m] - ns[m].mean())**2))
        return coef[0]/dt, se/dt, coef
    lam, se, coef = fit(lo, hi)
    L = hi - lo; sh = int(0.2*L)
    spread = [fit(lo + s, hi + s)[0] for s in (-sh, 0, sh)]
    a, b = axs[row]
    a.plot(ns, lnd, color=COL[0])
    a.axvspan(lo, hi, color=COL[0], alpha=0.12, lw=0)
    xx = np.array([0, NSTEP])
    a.plot(xx, np.polyval(coef, xx), "--", color=COL[1], lw=0.8)
    a.set_xlabel("$n$"); a.set_ylabel(r"$\Delta_n$"); a.set_xlim(0, NSTEP)
    a.set_ylim(lnd.min() - 0.3, lnd.max() + 0.6)
    a.text(0.97, 0.05, r"$\hat\lambda_1=%.3f\pm%.3f$" % (lam, se), transform=a.transAxes,
           ha="right", va="bottom", color=COL[1], fontsize=8)
    sl = np.gradient(lnd, ns)/dt
    b.plot(ns, sl, color=COL[0], lw=0.9)
    b.axvspan(lo, hi, color=COL[0], alpha=0.12, lw=0)
    b.axhline(LAMREF[name], color="0.4", ls=":", lw=0.9)
    b.text(NSTEP - 5, LAMREF[name], r"$\lambda_1=%.3f$ (Jacobian)" % LAMREF[name], ha="right",
           va="bottom", color="0.3", fontsize=8)
    b.set_xlabel("$n$"); b.set_ylabel(r"$(\mathrm{d}\Delta_n/\mathrm{d}n)/\Delta t$")
    b.set_xlim(0, NSTEP); b.set_ylim(0, 3*LAMREF[name])
    panel(a, "ac"[row], x=-0.2); panel(b, "bd"[row], x=-0.2)
    print("%s: window %d-%d  lambda_1 = %.4f +- %.4f (SE); shifted windows %s; reference %.4f"
          % (name, lo, hi, lam, se, ["%.3f" % v for v in spread], LAMREF[name]))
    print("   local slope/dt: min over n>%d %.3f; saturation Delta_n(%d) = %.2f, Delta_1 = %.2f"
          % (hi, sl[ns > hi].min(), NSTEP, lnd[-1], lnd[0]))
fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(FIGDIR + "LyapunovExp.pdf")
print("wrote LyapunovExp.pdf")
