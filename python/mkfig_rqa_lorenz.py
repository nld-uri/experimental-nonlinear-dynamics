"""Figure rqa_lorenz.pdf and the numbers of the Chapter 18 worked example.
Two states of the Lorenz system, rho = 28 and rho = 35 (sigma = 10, beta = 8/3),
x-coordinate sampled at dt = 0.02, M = 4000 delay vectors (d = 3, tau = 9),
Theiler window w = 50, fixed recurrence rate RR = 2%, l_min = v_min = 2.
(a,b) recurrence plots (first 1500 points shown); (c) cumulative diagonal-line
distributions with the K2 fits over 60 <= l <= 200 samples.
Also: DET of the rho = 28 record against 99 IAAFT surrogates (RR re-solved per
surrogate) and the same for the AR(2) record of Chapter 3."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import lfilter
from rqa import recmat, eps_for_rr, rqa, k2_from_lines
from irsurr import irsurr
COL = [(0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.929, 0.694, 0.125)]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
rng = np.random.default_rng(28)
dt, M, d, tau, w, RR = 0.02, 4000, 3, 9, 50, 0.02
def embed(x): 
    m = len(x)-(d-1)*tau; return np.column_stack([x[j*tau:j*tau+m] for j in range(d)])
def lorenz(rho, n):
    f = lambda t, u: [10*(u[1]-u[0]), u[0]*(rho-u[2])-u[1], u[0]*u[1]-8/3*u[2]]
    s = solve_ivp(f, [0, 40+n*dt], [5, -5, 20], t_eval=40+np.arange(n)*dt, rtol=1e-9, atol=1e-9)
    return s.y[0]
def analyse(x):
    X = embed(x)[:M]; eps = eps_for_rr(X, RR, w); R = recmat(X, eps, w); m = rqa(R, w)
    return X, eps, R, m
res = {}
for rho in [28, 35]:
    x = lorenz(rho, M+(d-1)*tau); X, eps, R, m = analyse(x); res[rho] = (x, eps, R, m)
    K2, Pc, ell = k2_from_lines(m['P'], dt, 60, 200)
    print(f"rho={rho}: eps={eps:.3f} (sigma_x={x.std():.2f}) RR={m['RR']:.4f} DET={m['DET']:.4f} L={m['L']:.2f} "
          f"Lmax={m['Lmax']} ENTR={m['ENTR']:.3f} LAM={m['LAM']:.4f} TT={m['TT']:.2f}  K2[60,200]={K2:.3f}")
    for lo, hi in [(10, 30), (30, 60), (60, 120), (120, 250)]:
        print(f"    K2 fit l in [{lo},{hi}]: {k2_from_lines(m['P'], dt, lo, hi)[0]:.3f}")
# surrogate test on DET for rho=28 and for AR(2)
def det_test(x, label, B=99):
    Xd = embed(x)[:M]; eps = eps_for_rr(Xd, RR, w); m0 = rqa(recmat(Xd, eps, w), w)
    dets = []
    for b in range(B):
        s, _ = irsurr(x, 40, rng); Xs = embed(s)[:M]; e = eps_for_rr(Xs, RR, w)
        dets.append(rqa(recmat(Xs, e, w), w)['DET'])
    dets = np.array(dets); rank = np.sum(dets < m0['DET'])+1
    print(f"{label}: DET data {m0['DET']:.4f}; surrogates mean {dets.mean():.4f} sd {dets.std():.4f} "
          f"max {dets.max():.4f}; rank {rank}/{B+1}; effect size {(m0['DET']-dets.mean())/dets.std():.1f}")
det_test(res[28][0], "Lorenz rho=28")
ar2 = lfilter([1.0], [1.0, -(2-(np.pi/10)**2-0.05), -(0.05-1)], rng.standard_normal(M+(d-1)*tau+3000))[3000:]
det_test(ar2, "AR(2) rho=0.05")
# figure
fig = plt.figure(figsize=(6.3, 2.3))
for k, rho in enumerate([28, 35]):
    ax = fig.add_subplot(1, 3, k+1); R = res[rho][2][:1500, :1500]
    ax.imshow(R, cmap="Greys", origin="lower", interpolation="none", rasterized=True)
    ax.set_title(f"({'ab'[k]}) $\\rho={rho}$", fontsize=8); ax.set_xlabel("$i$"); ax.set_ylabel("$j$")
    ax.set_xticks([0, 500, 1000, 1500]); ax.set_yticks([0, 500, 1000, 1500]); ax.tick_params(labelsize=6)
ax = fig.add_subplot(1, 3, 3)
for k, rho in enumerate([28, 35]):
    m = res[rho][3]; K2, Pc, ell = k2_from_lines(m['P'], dt, 60, 200)
    sel = (ell >= 2) & (ell <= 300) & (Pc > 0)
    ax.semilogy(ell[sel]*dt, Pc[sel]/Pc[2], color=COL[k], lw=1, label=f"$\\rho={rho}$")
    ax.semilogy(np.array([60, 200])*dt, Pc[60]/Pc[2]*np.exp(-K2*(np.array([60, 200])-60)*dt), "--", color="k", lw=0.8)
ax.set_xlabel("line length $\\ell\\,\\Delta t$"); ax.set_ylabel("$P^c(\\ell)/P^c(2)$")
ax.legend(frameon=False, fontsize=7); ax.set_title("(c)", fontsize=8)
fig.tight_layout(pad=0.4)
fig.savefig("../figures/rqa_lorenz.pdf", dpi=200); print("wrote figures/rqa_lorenz.pdf")
