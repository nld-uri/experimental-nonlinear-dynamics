"""Numeric checks for Chapter `embdim` (Selecting Embedding Dimension).

1. Uniform-noise original FNN fraction: P(|U1-U2| > a) = (1-a)^2 for 0<=a<=1,
   zero for a>1 (truncation), and the averaged form <(1-r eps)^2 Theta(1-r eps)>.
2. Accuracy of the nearest-neighbour distance heuristic (eq. nnds) for
   Gaussian white noise, sigma = 1, N = 2e4, d = 2..12.
3. E|delta| for Gaussian and uniform white noise: 2/sqrt(pi), 2/sqrt(3).
4. Hegger-Kantz bound d_max = ln N / ln r.
5. Cao's E1, E2 on Lorenz x, Henon x and Gaussian noise.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma
from scipy.integrate import solve_ivp

rng = np.random.default_rng(1)

# ---- 1. uniform-noise fraction -------------------------------------------
print("1. P(|U1-U2| > a) vs (1-a)^2")
u = rng.random((2, 10**6))
dif = np.abs(u[0] - u[1])
for a in [0.1, 0.3, 0.5, 0.9, 1.0, 1.2]:
    print(f"   a={a:4.1f}: sim {np.mean(dif > a):.4f}   formula {max(0, 1-a)**2:.4f}")
# averaged form: for a set of eps_i, fraction = <(1 - r eps)^2 Theta(1 - r eps)>
N, d, r = 5000, 3, 10
x = rng.random(N + d)          # uniform data on [0,1]
Y = np.array([x[np.arange(N) + k] for k in range(d)]).T   # tau = 1
tree = cKDTree(Y)
dist, idx = tree.query(Y, k=2)
eps = dist[:, 1]; j = idx[:, 1]
delta = np.abs(rng.random(N) - rng.random(N))   # independent (d+1)-th coords
f_sim = np.mean(delta > r*eps)
f_trunc = np.mean(np.where(r*eps < 1, (1-r*eps)**2, 0.0))
f_notrunc = np.mean((1-r*eps)**2)
print(f"   averaged: sim {f_sim:.4f}  truncated {f_trunc:.4f}  untruncated {f_notrunc:.4f}"
      f"  (frac of pairs with r eps>1: {np.mean(r*eps>1):.3f})")

# ---- 2. NN-distance heuristic ---------------------------------------------
print("2. eq. nnds vs simulated mean NN distance, Gaussian sigma=1, N=2e4")
N = 20000
for d in [2, 3, 4, 5, 8, 12]:
    X = rng.standard_normal((N, d))
    dist, _ = cKDTree(X).query(X, k=2)
    sim = dist[:, 1].mean()
    heur = np.sqrt(2)*gamma((d+1)/2)/gamma(d/2)*(np.sqrt(d)/N)**(1/d)
    print(f"   d={d:2d}: sim {sim:.4f}  heuristic {heur:.4f}  ratio heur/sim {heur/sim:.3f}")

# ---- 3. E|delta| ----------------------------------------------------------
print("3. E|x_i - x_j| / sigma")
g = rng.standard_normal((2, 10**6))
print(f"   Gaussian: sim {np.abs(g[0]-g[1]).mean():.4f}  2/sqrt(pi) = {2/np.sqrt(np.pi):.4f}")
u = rng.random((2, 10**6)); su = np.sqrt(1/12)
print(f"   Uniform : sim {np.abs(u[0]-u[1]).mean()/su:.4f}  2/sqrt(3)  = {2/np.sqrt(3):.4f}")

# ---- 4. d_max -------------------------------------------------------------
print("4. d_max = ln N / ln r")
for N_, r_ in [(1e5, 10), (1e4, 10), (6e4, 10), (1e5, 4)]:
    print(f"   N={N_:.0e}, r={r_}: d_max = {np.log(N_)/np.log(r_):.2f}")

# ---- 5. Cao E1, E2 ----------------------------------------------------------
def cao(x, tau, dmax):
    x = np.asarray(x, float)
    E, Es = [], []
    for d in range(1, dmax + 2):
        n = len(x) - d*tau                       # need x_{i+d tau}
        Y = np.array([x[np.arange(n) + k*tau] for k in range(d)]).T
        Y1 = np.array([x[np.arange(n) + k*tau] for k in range(d + 1)]).T
        tree = cKDTree(Y)
        dist, idx = tree.query(Y, k=2, p=np.inf)
        j = idx[:, 1]; e = dist[:, 1]
        e1 = np.max(np.abs(Y1 - Y1[j]), axis=1)
        ok = e > 0
        E.append(np.mean(e1[ok]/e[ok]))
        Es.append(np.mean(np.abs(x[np.arange(n) + d*tau] - x[j + d*tau])))
    E, Es = np.array(E), np.array(Es)
    return E[1:]/E[:-1], Es[1:]/Es[:-1]        # E1(d), E2(d), d=1..dmax

def lorenz(t, u):
    return [10*(u[1]-u[0]), u[0]*(28-u[2])-u[1], u[0]*u[1]-8/3*u[2]]
sol = solve_ivp(lorenz, (0, 400), [5, -5, 20], t_eval=np.arange(100, 400, 0.02),
                rtol=1e-9, atol=1e-9)
xl = sol.y[0]
E1, E2 = cao(xl, 8, 8)
print("5. Cao: Lorenz x (tau=8, N=%d)" % len(xl))
print("   E1:", np.round(E1, 3)); print("   E2:", np.round(E2, 3))
h = np.zeros(15000); hx, hy = 0.1, 0.1
for i in range(15000):
    hx, hy = 1.4 - hx**2 + 0.3*hy, hx; h[i] = hx
E1, E2 = cao(h[1000:], 1, 8)
print("   Henon x (tau=1)"); print("   E1:", np.round(E1, 3)); print("   E2:", np.round(E2, 3))
E1, E2 = cao(rng.standard_normal(15000), 1, 8)
print("   Gaussian noise"); print("   E1:", np.round(E1, 3)); print("   E2:", np.round(E2, 3))

# ---- 1b. truncation matters when N^{1/d} < r ------------------------------
print("1b. truncated vs untruncated averaged fraction")
for N, d, r in [(2000, 6, 10), (20000, 3, 10), (500, 4, 10)]:
    x = rng.random(N + d); Y = np.array([x[np.arange(N) + k] for k in range(d)]).T
    dist, idx = cKDTree(Y).query(Y, k=2); eps = dist[:, 1]
    delta = np.abs(rng.random(N) - rng.random(N))
    print(f"   N={N}, d={d}, r={r}: sim {np.mean(delta > r*eps):.4f}  trunc "
          f"{np.mean(np.where(r*eps < 1, (1-r*eps)**2, 0)):.4f}  untrunc "
          f"{np.mean((1-r*eps)**2):.4f}  N^(1/d)={N**(1/d):.1f}")
