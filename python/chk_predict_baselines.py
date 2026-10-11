"""Baselines for Fig. Pred (Chapter 15): persistence and the global affine model
against the local linear predictor, on the record, split, gap and starting points
of mkfig_Prediction.py (repeated here; that script plots and is not importable).

Lead h counts samples beyond the last observed sample x_{s+(D-1)TAU} of the start.
Persistence holds that sample; the global affine model is the vector map
x_{j+1} = A x_j + b fitted by least squares on all NMOD modeling vectors (the
limit r -> N of the local model), iterated like loclin. Errors are RMS over the
NENS starts, in units of the standard deviation of the record.
"""
import numpy as np
from scipy.integrate import solve_ivp
from loclin import loclin
from nlutil import embed

AL, BE, A, B = 9.0, 100/7, 8/7, 5/7
DT, TTRANS, NMOD, NPRED, NENS = 0.1, 200.0, 20000, 600, 200   # as mkfig_Prediction.py
D, TAU, R = 3, 6, 16
GAP = (D - 1)*TAU
H = 10                                   # longest lead reported here


def chua(n, dt=DT):
    def f(t, x):
        return [AL*(x[1] - x[0] + B*x[0] + 0.5*(A - B)*(abs(x[0] + 1) - abs(x[0] - 1))),
                x[0] - x[1] + x[2], -BE*x[1]]
    t = TTRANS + dt*np.arange(n)
    return solve_ivp(f, (0, t[-1]), [0.0, 0.0, 0.6], t_eval=t, rtol=1e-9, atol=1e-12).y[0]


x = chua(NMOD + GAP + NPRED + NENS + (D - 1)*TAU + 10)
x = (x - x.mean())/x.std()
y, n = embed(x, D, TAU)
ym = y[:, :NMOD]
rng = np.random.default_rng(0)           # same starts as the figure
starts = NMOD + GAP + rng.permutation(n - NMOD - GAP - NPRED - 1)[:NENS]

Xg = np.vstack([ym[:, :-1], np.ones(NMOD - 1)])                 # global affine fit
Bg = np.linalg.lstsq(Xg.T, ym[:, 1:].T, rcond=None)[0].T       # (D, D+1)

E = {"local linear": 0, "persistence": 0, "global affine": 0}
for s in starts:
    true = y[-1, s:s + H + 1]
    E["local linear"] = E["local linear"] + (loclin(ym, y[:, s], R, H)[-1] - true)**2
    E["persistence"] = E["persistence"] + (y[-1, s] - true)**2
    z, g = y[:, s], [y[-1, s]]
    for k in range(H):
        z = Bg @ np.append(z, 1.0)
        g.append(z[-1])
    E["global affine"] = E["global affine"] + (np.array(g) - true)**2
for key in E:
    e = np.sqrt(E[key]/NENS)
    print("%-14s RMS/sigma at lead h=1: %.2e   h=%d: %.2e" % (key, e[1], H, e[H]))
