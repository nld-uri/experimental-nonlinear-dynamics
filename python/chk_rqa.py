"""Sanity checks for rqa.py: (1) a periodic record gives DET -> 1 and L bounded by
record; (2) white noise gives DET small; (3) K2 estimate on Lorenz x vs lambda_1 = 0.906."""
import numpy as np
from scipy.integrate import solve_ivp
from rqa import *
rng = np.random.default_rng(0)
def embed(x, d, tau): 
    M = len(x)-(d-1)*tau; return np.column_stack([x[j*tau:j*tau+M] for j in range(d)])
# periodic
t = np.arange(1500)*0.05; X = embed(np.sin(t), 2, 8); w = 20
eps = eps_for_rr(X, 0.05, w); m = rqa(recmat(X, eps, w), w)
print("sine: RR %.3f DET %.3f L %.1f Lmax %d ENTR %.2f LAM %.3f" % (m['RR'], m['DET'], m['L'], m['Lmax'], m['ENTR'], m['LAM']))
X = embed(rng.standard_normal(1500), 3, 1); eps = eps_for_rr(X, 0.05, w); m = rqa(recmat(X, eps, w), w)
print("iid noise: RR %.3f DET %.3f L %.2f Lmax %d ENTR %.2f LAM %.3f" % (m['RR'], m['DET'], m['L'], m['Lmax'], m['ENTR'], m['LAM']))
def lor(t, u, rho=28): return [10*(u[1]-u[0]), u[0]*(rho-u[2])-u[1], u[0]*u[1]-8/3*u[2]]
dt = 0.02
for M in [3000, 6000]:
    sol = solve_ivp(lor, [0, 50+M*dt], [5, -5, 20], t_eval=np.arange(50, 50+M*dt, dt), rtol=1e-9, atol=1e-9)
    x = sol.y[0]; X = embed(x, 3, 9); w = 50
    for rr in [0.02, 0.05]:
        eps = eps_for_rr(X, rr, w); R = recmat(X, eps, w); m = rqa(R, w)
        K2, Pc, ell = k2_from_lines(m['P'], dt, 4, 40)
        print(f"Lorenz M={M} RR={rr}: eps={eps:.2f} DET {m['DET']:.3f} L {m['L']:.1f} Lmax {m['Lmax']} ENTR {m['ENTR']:.2f} LAM {m['LAM']:.3f}  K2 {K2:.3f} (lambda1=0.906)")
