"""Numerical checks for Chapter `metrics` (characteristic distances, SOD tracking).

1. Lowest-harmonic ratios <q^2>/<qdot^2> for q = t, t^2 (raw and centred) and
   sin, sin^2 (centred), against the closed forms quoted in the text.
2. Far-field expansion  E||y-x|| = ||r|| + (tr C - rhat'C rhat)/(2||r||) + O(||r||^-2),
   r = y - xbar, on a non-Gaussian cloud (logistic-map delay vectors).
3. Noise corollary: isotropic noise of variance s^2 per coordinate shifts D by
   ~ s^2 (m-1)/(2||r||), independent of the attractor's shape.
4. Toy tracking problem: logistic map x -> r x (1-x) with r drifting slowly through
   the periodic windows between 3.5 and 4.0.  Features = characteristic distances
   to L reference points.  Checks: (a) individual features jump; (b) the jump vectors
   span a low-dimensional subspace and the leading SPM is nearly orthogonal to it;
   (c) the smoothest SOC is affine in r (correlation); (d) moment-feature baseline;
   (e) far-only vs near-only reference points; (f) centring matters; (g) noise.
5. Duffing (0.25, -1, 1, f = 0.375, w = 1), dt = pi/36: AMI first minimum in samples.
"""
import os
import sys
import numpy as np
from gsvd import gsvd

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- 1. ratios
print("== 1. lowest-harmonic ratios <q^2>/<qdot^2>")
T = 1.0
t = np.linspace(0, T, 200001)
def ratio(q, dt):
    qd = np.gradient(q, dt)
    return np.mean(q**2)/np.mean(qd**2)
dt = t[1]-t[0]
print(f" q=t   raw  : {ratio(t, dt):.5f}   closed form T^2/3    = {T**2/3:.5f}")
print(f" q=t^2 raw  : {ratio(t**2, dt):.5f}   closed form 3T^2/20  = {3*T**2/20:.5f}")
print(f" q=t   ctr  : {ratio(t-t.mean(), dt):.5f}   closed form T^2/12   = {T**2/12:.5f}")
print(f" q=t^2 ctr  : {ratio(t**2-np.mean(t**2), dt):.5f}   closed form T^2/15   = {T**2/15:.5f}")
w = 2*np.pi
tt = np.linspace(0, 1, 200001); dtt = tt[1]-tt[0]
s1 = np.sin(w*tt); s2 = np.sin(w*tt)**2
print(f" q=sin   ctr: {ratio(s1-s1.mean(), dtt):.5f}   closed form 1/w^2    = {1/w**2:.5f}")
print(f" q=sin^2 ctr: {ratio(s2-s2.mean(), dtt):.5f}   closed form 1/(4w^2) = {1/(4*w**2):.5f}")
print(f" q=sin^2 raw: {ratio(s2, dtt):.5f}   closed form 3/(4w^2) = {3/(4*w**2):.5f}")

# ------------------------------------------------- helpers: logistic map, D
def logistic_orbit(r, n, burn=500, x0=None):
    x = rng.uniform(0.2, 0.8) if x0 is None else x0
    for _ in range(burn):
        x = r*x*(1-x)
    out = np.empty(n)
    for i in range(n):
        x = r*x*(1-x); out[i] = x
    return out

def embed(x, m=2, tau=1):
    n = len(x)-(m-1)*tau
    return np.column_stack([x[j*tau:j*tau+n] for j in range(m)])

def cdist(X, Y):
    """D(y_j) = mean_i ||x_i - y_j|| for each row y_j of Y."""
    d = np.sqrt(((X[:, None, :]-Y[None, :, :])**2).sum(-1))
    return d.mean(0)

# ------------------------------------------------ 2. far-field expansion
print("\n== 2. far-field expansion on logistic delay vectors (r=3.9, m=2)")
X = embed(logistic_orbit(3.9, 20000), 2)
xbar = X.mean(0); C = np.cov(X.T, bias=True)
for R in [1, 2, 5, 10, 20]:
    u = np.array([0.6, -0.8]); y = xbar + R*u
    exact = cdist(X, y[None, :])[0]
    approx1 = R
    approx2 = R + (np.trace(C) - u@C@u)/(2*R)
    print(f" ||r||={R:5.1f}: exact {exact:.6f}  0th {approx1:.6f} (err {exact-approx1:+.2e})"
          f"  1st {approx2:.6f} (err {exact-approx2:+.2e}, ~||r||^-2 = {R**-2:.1e})")

# ------------------------------------------------ 3. noise corollary
print("\n== 3. noise shift of D for far point, isotropic noise sigma")
Xbig = embed(logistic_orbit(3.9, 400000), 2)
for sig in [0.05, 0.1, 0.2]:
    for R in [2, 10]:
        u = np.array([0.6, -0.8]); y = xbar + R*u
        X = Xbig
        Xn = X + sig*rng.standard_normal(X.shape)
        shift = cdist(Xn, y[None, :])[0] - cdist(X, y[None, :])[0]
        pred = sig**2*(X.shape[1]-1)/(2*R)
        print(f" sigma={sig:.2f} ||r||={R:4.1f}: measured shift {shift:+.5f}, predicted {pred:+.5f}")

# ------------------------------------------------ 4. toy tracking problem
print("\n== 4. toy tracking: logistic map, r drifting 3.5 -> 4.0 sinusoidally")
Nwin, npts, m = 400, 3000, 2
tw = np.linspace(0, 1, Nwin)
def drift(t):
    """Non-sinusoidal, non-monotone drift: rise, plateau, partial return."""
    s = np.where(t < 0.4, 3*(t/0.4)**2 - 2*(t/0.4)**3, 1.0)
    d = np.clip((t-0.6)/0.4, 0, 1)
    return s - 0.5*(3*d**2 - 2*d**3)
r_true = 3.5 + 0.5*drift(tw)                                 # 3.5 -> 4.0 -> 3.75
Xs = [embed(logistic_orbit(r, npts), m) for r in r_true]
allpts = np.vstack(Xs)
lo, hi = allpts.min(0), allpts.max(0); c0 = (lo+hi)/2; half = (hi-lo)/2

L = 60
Yref_near = c0 + rng.uniform(-1, 1, (L, m))*half*1.0        # inside bounding box
Yref_far = c0 + 8*half*rng.standard_normal((L, m))/np.sqrt(m) # far away, |r|~8 box half-widths
Yref_far = c0 + (Yref_far-c0)/np.linalg.norm(Yref_far-c0, axis=1)[:, None]*8*half.mean()

def features(Yref, noise=0.0):
    F = np.empty((Nwin, len(Yref)))
    for l, Xl in enumerate(Xs):
        Xn = Xl + noise*rng.standard_normal(Xl.shape) if noise else Xl
        F[l] = cdist(Xn, Yref)
    return F

def moments(noise=0.0):
    F = np.empty((Nwin, m + m*(m+1)//2))
    for l, Xl in enumerate(Xs):
        Xn = Xl + noise*rng.standard_normal(Xl.shape) if noise else Xl
        mu = Xn.mean(0); Cc = np.cov(Xn.T, bias=True)
        F[l] = np.r_[mu, Cc[np.triu_indices(m)]]
    return F

def sod_track(Y, centre=True, tol=1e-8):
    """SOD of the tracking matrix. Columns centred; if Y is numerically rank
    deficient it is first reduced to its leading POD subspace (relative
    singular-value tolerance tol), since the GSVD needs full column rank."""
    Yc = Y - Y.mean(0) if centre else Y.copy()
    Uo, so, Vto = np.linalg.svd(Yc, full_matrices=False)
    k = int(np.sum(so > tol*so[0]))
    B = Vto[:k].T                                  # L x k basis
    Z = Yc @ B                                     # reduced tracking matrix
    DZ = np.diff(Z, axis=0)
    U, V, Xg, c, s = gsvd(Z, DZ)                   # Z = U diag(c) Xg.T
    lam2 = (c/s)**2                                # SOVs, descending
    PsiZ = np.linalg.inv(Xg.T)                     # SPMs in reduced space
    Q = Z @ PsiZ
    Psi = B @ PsiZ                                 # back in feature space: Q = Yc Psi
    return lam2, Q, Psi

def corr(a, b):
    return abs(np.corrcoef(a, b)[0, 1])

Ynear = features(Yref_near)
Yfar = features(Yref_far)
Ymom = moments()

# (a) individual features jump: largest window-to-window increment vs median
inc = np.abs(np.diff(Ynear, axis=0))
print(f" (a) near features: median |dY| {np.median(inc):.2e}, max |dY| {inc.max():.2e}, ratio {inc.max()/np.median(inc):.0f}")

# (b) jump subspace: windows where the increment is a jump (> 10x median over columns)
score = np.linalg.norm(inc, axis=1)
thr = 10*np.median(score)
jumps = np.where(score > thr)[0]
Delta = inc[jumps]                                  # J x L jump vectors
J = len(jumps)
sv = np.linalg.svd(Delta, compute_uv=False)
print(f" (b) J = {J} jump windows detected (>{thr:.2e}); L = {L}; rank of jump matrix "
      f"(sv > 1e-2*max): {np.sum(sv > 1e-2*sv[0])}")
lam2, Q, Psi = sod_track(Ynear)
psi1 = Psi[:, 0]/np.linalg.norm(Psi[:, 0])
proj = np.linalg.norm(Delta @ psi1)/np.linalg.norm(Delta)   # fraction of jump energy along psi1
rand = np.linalg.norm(Delta @ (v := rng.standard_normal(L)/np.sqrt(L)))/np.linalg.norm(Delta)
print(f"     ||Delta psi1||/||Delta|| = {proj:.3e}  (random unit direction: {rand:.3e})")
print(f"     first four SOVs: {lam2[:4]}")
print(f" (c) corr(SOC1, r)  near points, centred : {corr(Q[:,0], r_true):.5f}")
# fit r = a + b SOC1 and residual
A = np.column_stack([np.ones(Nwin), Q[:, 0]]); coef = np.linalg.lstsq(A, r_true, rcond=None)[0]
res = r_true - A@coef
print(f"     affine fit residual rms {res.std():.4f} over range {np.ptp(r_true):.2f}")
lam2u, Qu, _ = sod_track(Ynear, centre=False)
print(f" (f) corr(SOC1, r)  near points, NOT centred: {corr(Qu[:,0], r_true):.5f}; SOV1/SOV2 = {lam2u[0]/lam2u[1]:.2e}; std(SOC1)/|mean| = {Qu[:,0].std()/abs(Qu[:,0].mean()):.2e}")
lam2f, Qf, _ = sod_track(Yfar)
print(f" (e) corr(SOC1, r)  far points            : {corr(Qf[:,0], r_true):.5f}; numerical rank of far Y: {int(np.sum(np.linalg.svd(Yfar-Yfar.mean(0),compute_uv=False) > 1e-8*np.linalg.svd(Yfar-Yfar.mean(0),compute_uv=False)[0]))}")
lam2m, Qm, _ = sod_track(Ymom)
print(f" (d) corr(SOC1, r)  moment features (5)   : {corr(Qm[:,0], r_true):.5f}")
# far-field features regressed on moments: how much is explained?
A = np.column_stack([np.ones(Nwin), Ymom]);
for name, F in [("far", Yfar), ("near", Ynear)]:
    fit = A @ np.linalg.lstsq(A, F, rcond=None)[0]
    r2 = 1 - ((F-fit)**2).sum()/((F-F.mean(0))**2).sum()
    print(f"     R^2 of {name:4s} features on (mean, cov) : {r2:.6f}")
# fewer near points
for Ls in [3, 5, 10, 20]:
    l2, Qs, _ = sod_track(Ynear[:, :Ls])
    print(f"     near points L={Ls:3d}: corr(SOC1,r) = {corr(Qs[:,0], r_true):.5f}")
# (g) noise
for sig in [0.02, 0.05, 0.1]:
    Yn = features(Yref_near, noise=sig); Yfn = features(Yref_far, noise=sig)
    _, Qn, _ = sod_track(Yn); _, Qfn, _ = sod_track(Yfn)
    print(f" (g) noise sigma={sig:.2f} (signal std {allpts.std():.2f}): corr near {corr(Qn[:,0], r_true):.4f}, far {corr(Qfn[:,0], r_true):.4f}")

# (h) null: stationary record, same L, same windows
Xs_null = [embed(logistic_orbit(3.9, npts), m) for _ in range(Nwin)]
Ynull = np.array([cdist(Xl, Yref_near) for Xl in Xs_null])
l2n, Qn0, _ = sod_track(Ynull)
print(f" (h) null (no drift): SOV1 = {l2n[0]:.1f}, SOV1/SOV2 = {l2n[0]/l2n[1]:.2f}  vs drifting SOV1 = {lam2[0]:.1f}, SOV1/SOV2 = {lam2[0]/lam2[1]:.2f}")
print(f"     null SOC1 std / drifting SOC1 std = {Qn0[:,0].std()/Q[:,0].std():.3f}")
# (i) lowest harmonic: SOC1 vs time-only smooth functions
print(f" (i) corr(SOC1, t) = {corr(Q[:,0], tw):.4f}, corr(r, t) = {corr(r_true, tw):.4f}  (SOC1 follows r, not time)")

# ------------------------------------------------ 5. Duffing AMI at dt = pi/36
print("\n== 5. Duffing AMI first minimum, dt = pi/36 (72 samples/period)")
from scipy.integrate import solve_ivp
def duff(t, u, g=0.25, a=-1.0, b=1.0, f=0.375, w=1.0):
    x, v = u
    return [v, -g*v - a*x - b*x**3 + f*np.cos(w*t)]
def ami(x, kmax, nb=32):
    edges = np.linspace(x.min(), x.max()+1e-12, nb+1)
    b = np.digitize(x, edges)-1
    out = []
    for k in range(1, kmax+1):
        h = np.histogram2d(b[:-k], b[k:], bins=[np.arange(nb+1)]*2)[0]
        p = h/h.sum(); px = p.sum(1); py = p.sum(0)
        nz = p > 0
        out.append((p[nz]*np.log(p[nz]/(px[:, None]*py[None, :])[nz])).sum())
    return np.array(out)
for f, spp in [(0.35, 72), (0.375, 72), (0.4, 72), (0.3, 72), (0.3, 30)]:
    dt5 = 2*np.pi/spp; n5 = 30000
    ts = np.arange(n5+60*spp)*dt5
    sol = solve_ivp(duff, (0, ts[-1]), [0.1, 0.0], t_eval=ts, rtol=1e-9, atol=1e-11,
                    method="DOP853", args=(0.25, -1.0, 1.0, f, 1.0))
    x = sol.y[0][60*spp:]
    I = ami(x, int(1.2*spp))
    loc = [k+1 for k in range(1, len(I)-1) if I[k] < I[k-1] and I[k] < I[k+1]]
    print(f" f={f:.3f}, {spp} samples/period: first AMI minimum at tau = {loc[0]} samples "
          f"= {loc[0]/spp:.2f} period; local minima {loc[:4]}")
print(" (ch17 uses tau = 53 at 72 samples/period = 0.74 period; ch14 uses tau = 7 at 30 = 0.23 period)")
