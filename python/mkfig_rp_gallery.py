"""Figure rp_gallery.pdf (Chapter 18): recurrence plots of eight records of equal
length M = 1500 embedded points, all at the same recurrence rate RR = 2% and the
same Theiler window w = 30 samples: (a) sinusoid, (b) two-frequency torus,
(c) Lorenz x, (d) Lorenz x with rho ramped 24 -> 60 over the record (drift),
(e) i.i.d. Gaussian noise, (f) the AR(2) process of Chapter 3 (rho = 0.05),
(g) Lorenz x + white noise at 10 dB SNR, (h) two-well Duffing x."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import lfilter
from rqa import recmat, eps_for_rr, rqa
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
rng = np.random.default_rng(18)
M, RR, w = 1500, 0.02, 30
def embed(x, d, tau):
    m = len(x)-(d-1)*tau; return np.column_stack([x[j*tau:j*tau+m] for j in range(d)])
def lorenz(rho_fun, dt=0.02, n=M+40, t0=40):
    f = lambda t, u: [10*(u[1]-u[0]), u[0]*(rho_fun(t)-u[2])-u[1], u[0]*u[1]-8/3*u[2]]
    T = t0 + n*dt
    s = solve_ivp(f, [0, T], [5, -5, 20], t_eval=t0+np.arange(n)*dt, rtol=1e-9, atol=1e-9)
    return s.y[0]
def duffing(dt=0.1, n=M+40, t0=200):
    f = lambda t, u: [u[1], -0.25*u[1]+u[0]-u[0]**3+0.3*np.cos(t)]
    s = solve_ivp(f, [0, t0+n*dt], [-0.67, -0.37], t_eval=t0+np.arange(n)*dt, rtol=1e-9, atol=1e-9)
    return s.y[0]
n = M+40
t = np.arange(n)
recs = {
 "(a) sinusoid": (np.sin(2*np.pi*t/60), 2, 15),
 "(b) quasiperiodic": (np.sin(2*np.pi*t/60)+0.8*np.sin(2*np.pi*t*np.sqrt(2)/60), 3, 15),
 "(c) Lorenz $x$": (lorenz(lambda t: 28.0), 3, 9),
 "(d) Lorenz, $\\rho$: 24$\\to$60": (lorenz(lambda t: 24+36*np.clip((t-40)/((M+40)*0.02), 0, 1)), 3, 9),
 "(e) i.i.d. Gaussian noise": (rng.standard_normal(n), 3, 1),
 "(f) AR(2), $\\rho=0.05$": (lfilter([1.0], [1.0, -(2-(np.pi/10)**2-0.05), -(0.05-1)], rng.standard_normal(n+2000))[2000:], 3, 5),
 "(g) Lorenz $x$ + noise, 10 dB": (None, 3, 9),
 "(h) Duffing $x$": (duffing(), 4, 8),
}
xl = recs["(c) Lorenz $x$"][0]
recs["(g) Lorenz $x$ + noise, 10 dB"] = (xl + rng.standard_normal(n)*xl.std()/np.sqrt(10), 3, 9)
fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.4))
for ax, (name, (x, d, tau)) in zip(axes.ravel(), recs.items()):
    X = embed(x, d, tau)[:M]
    eps = eps_for_rr(X, RR, w); R = recmat(X, eps, w); m = rqa(R, w)
    ax.imshow(R, cmap="Greys", origin="lower", interpolation="none", rasterized=True)
    ax.set_title(name, fontsize=7.5, pad=2)
    ax.set_xticks([0, 500, 1000, 1500]); ax.set_yticks([0, 500, 1000, 1500])
    ax.tick_params(labelsize=6)
    print(f"{name:32s} d={d} tau={tau} eps={eps:.3f} DET={m['DET']:.3f} L={m['L']:.1f} Lmax={m['Lmax']} ENTR={m['ENTR']:.2f} LAM={m['LAM']:.3f}")
for ax in axes[1]: ax.set_xlabel("$i$", fontsize=8)
for ax in axes[:, 0]: ax.set_ylabel("$j$", fontsize=8)
fig.tight_layout(pad=0.4)
fig.savefig("../figures/rp_gallery.pdf", dpi=200)
print("wrote figures/rp_gallery.pdf")
