"""Prediction.pdf (label `Pred`) -- iterated local linear prediction of the Chua
circuit x record with `loclin`, unweighted and with the weight of eq. (loclinw).

Record: the Chua-Matsumoto circuit of eq. (chuaeq) with the parameters of
Problem 10.1 (alpha = 9, beta = 100/7, a = 8/7, b = 5/7), from (0, 0, 0.6),
integrated with solve_ivp (RK45, rtol 1e-9), 200 time units of transient
discarded, sampled at DT = 0.1 (the first minimum of the AMI is then at lag 6,
the delay the text prescribes). NMOD points are used for the model and the
prediction starts at the first point after them. Embedding: d = 3, tau = 6,
r = 16 nearest neighbours. Records normalized to unit variance, so the errors
are in units of the standard deviation of the record.

Split in time with a gap: the model uses delay vectors 0..NMOD-1 (samples up to
x_{NMOD-1+(D-1)TAU}); every starting vector begins at column NMOD+GAP or later,
GAP = (D-1)TAU, so no sample of a test vector or of its forecast entered the model.
The plotted coordinate is the LAST one of the iterated delay vector, so the lead
h counts samples beyond the last observed sample x_{s+(D-1)TAU} of the start.

(a) the true continuation and the two predictions over NPRED steps from one
    starting point; (b) their errors; (c) the RMS error over NENS starting points
    against the horizon, with the exponential fit over the range FITH and the
    reference growth e^{lambda_1 h dt} (lambda_1 = 0.3326 from the Jacobian, QR).
"""
from _figstyle import *
from scipy.integrate import solve_ivp
from loclin import loclin
from nlutil import embed

AL, BE, A, B = 9.0, 100/7, 8/7, 5/7
DT, TTRANS, NMOD, NPRED, NENS = 0.1, 200.0, 20000, 600, 200
D, TAU, R = 3, 6, 16
LAM1 = 0.3326                      # largest Lyapunov exponent of the model (QR, Jacobian)
FITH = (8, 98)                     # lead range (samples) of the exponential fit:
# the range over which the ensemble RMS error grows on a straight line on log axes
GAP = (D - 1)*TAU                  # samples between the model and the first test vector


def chua(n, dt=DT):
    def f(t, x):
        return [AL*(x[1] - x[0] + B*x[0] + 0.5*(A - B)*(abs(x[0] + 1) - abs(x[0] - 1))),
                x[0] - x[1] + x[2], -BE*x[1]]
    t = TTRANS + dt*np.arange(n)
    return solve_ivp(f, (0, t[-1]), [0.0, 0.0, 0.6], t_eval=t, rtol=1e-9, atol=1e-12).y[0]


x = chua(NMOD + GAP + NPRED + NENS + (D - 1)*TAU + 10)
x = (x - x.mean())/x.std()
y, n = embed(x, D, TAU)                 # (D, n)
ym = y[:, :NMOD]                        # modeling trajectory

def predict(start, npred, w):
    yp = loclin(ym, y[:, start], R, npred, w=w)
    return yp[-1]                       # last coordinate = x_{start+(D-1)TAU+k}: lead k

k0 = NMOD + GAP
true = y[-1, k0:k0 + NPRED + 1]
p0 = predict(k0, NPRED, 0)
p1 = predict(k0, NPRED, 1)

# ensemble RMS error against horizon
rng = np.random.default_rng(0)
starts = NMOD + GAP + rng.permutation(n - NMOD - GAP - NPRED - 1)[:NENS]
E = {0: np.zeros(NPRED + 1), 1: np.zeros(NPRED + 1)}
for s in starts:
    t = y[-1, s:s + NPRED + 1]
    for w in (0, 1):
        E[w] += (predict(s, NPRED, w) - t)**2
for w in (0, 1):
    E[w] = np.sqrt(E[w]/NENS)
h = np.arange(NPRED + 1)
m = (h >= FITH[0]) & (h <= FITH[1])
lam_fit = {w: np.polyfit(h[m]*DT, np.log(E[w][m]), 1)[0] for w in (0, 1)}

fig, axs = plt.subplots(3, 1, figsize=(6.0, 5.4))
a, b, c = axs
hh = h*DT
a.plot(hh, true, color=COL[0], label="true")
a.plot(hh, p0, color=COL[1], lw=0.9, label="prediction")
a.plot(hh, p1, color=COL[2], lw=0.9, label="prediction, weighted")
a.set_ylabel("$x$"); a.set_xlim(0, NPRED*DT); a.set_ylim(-2.4, 2.9)
a.legend(loc="upper left", ncol=3, fontsize=7.5)
b.plot(hh, p0 - true, color=COL[1], lw=0.9, label="no weighting")
b.plot(hh, p1 - true, color=COL[2], lw=0.9, label="with weighting")
b.set_ylabel("prediction error"); b.set_xlim(0, NPRED*DT); b.legend(loc="upper left", ncol=2, fontsize=7.5)
b.set_xlabel("$t$")
c.semilogy(hh, E[0], color=COL[1], label="no weighting")
c.semilogy(hh, E[1], color=COL[2], label="with weighting")
c.axvspan(FITH[0]*DT, FITH[1]*DT, color="0.5", alpha=0.12, lw=0)
ref = E[0][FITH[0]]*np.exp(LAM1*(hh - FITH[0]*DT))
sel = (hh >= FITH[0]*DT) & (ref <= 3)
c.semilogy(hh[sel], ref[sel], "k:", lw=0.9,
           label=r"$e^{\lambda_1 t}$, $\lambda_1=%.3f$ (model)" % LAM1)
fitline = E[0][FITH[0]]*np.exp(lam_fit[0]*(hh - FITH[0]*DT))
sel = (hh >= FITH[0]*DT) & (fitline <= 3)
c.semilogy(hh[sel], fitline[sel], "k--", lw=0.9,
           label=r"fit, $\hat\lambda=%.3f$" % lam_fit[0])
c.set_xlabel("$t$"); c.set_ylabel(r"RMS error / $\sigma_x$")
c.set_xlim(0, NPRED*DT); c.set_ylim(1e-4, 6); c.legend(loc="lower right", fontsize=7, ncol=2)
for ax, L in zip(axs, "abc"):
    panel(ax, L, x=-0.08)
fig.tight_layout(pad=0.4)
fig.savefig(FIGDIR + "Prediction.pdf")

print("single trajectory: |error| < 0.05 up to h = %d (t = %.1f); "
      "unweighted first exceeds 0.5 at h = %d, weighted at h = %d"
      % (np.argmax(np.abs(p0 - true) > 0.05), DT*np.argmax(np.abs(p0 - true) > 0.05),
         np.argmax(np.abs(p0 - true) > 0.5), np.argmax(np.abs(p1 - true) > 0.5)))
for w, lab in [(0, "unweighted"), (1, "weighted  ")]:
    e = E[w]
    print("%s: RMS/sigma = %.2e at h=1, %.3f at h=10, %.3f at h=50, %.2f at h=100, "
          "%.2f at h=300, %.2f at h=600; reaches 0.5 at h = %d (t = %.1f, %.2f Lyapunov times); "
          "fitted growth rate over h in %s: %.3f (model lambda_1 = %.3f)"
          % (lab, e[1], e[10], e[50], e[100], e[300], e[600],
             np.argmax(e > 0.5), DT*np.argmax(e > 0.5), LAM1*DT*np.argmax(e > 0.5),
             FITH, lam_fit[w], LAM1))
hq = np.argmax(E[0] > 0.5)               # unweighted error reaches half sigma
print("weighting: max |E_w/E_u - 1| over h = 1..%d: %.3f; over h = 1..%d: %.3f"
      % (hq, np.max(np.abs(E[1][1:hq + 1]/E[0][1:hq + 1] - 1)),
         NPRED, np.max(np.abs(E[1][1:]/E[0][1:] - 1))))
print("starts span columns %d..%d; model columns 0..%d" % (starts.min(), starts.max(), NMOD - 1))
print("wrote Prediction.pdf  (N_model = %d, dt = %g, d = %d, tau = %d, r = %d, ensemble %d)"
      % (NMOD, DT, D, TAU, R, NENS))
