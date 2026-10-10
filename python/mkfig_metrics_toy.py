"""metrics_toy.pdf -- toy slow-variable tracking with characteristic distances.

Logistic map x -> r x(1-x); r drifts slowly (rise, plateau, partial return)
through the periodic windows in [3.5, 4.0]. Each of N = 400 windows holds 3000
steady-state points at a fixed r_l, embedded with m = 2. Features are the
characteristic distances to L = 60 reference points drawn uniformly in the
bounding box of the pooled attractor. SOD (GSVD of the centred tracking matrix
and its first difference) returns the smoothest coordinate, compared with r_l.

Panels: (a) drift r_l; (b) three columns of Y (discontinuous in l);
(c) ten largest SOVs, drifting record and stationary null; (d) SOC_1 against
window index with r_l affinely scaled onto it; (e) SOC_1 against r_l;
(f) SOC_1 against r_l for the two baselines: the five moment features
(mean and covariance of the embedded window) and L = 60 far reference points,
which the far-field expansion makes functions of those same moments.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gsvd import gsvd

COL = [(0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.929, 0.694, 0.125),
       (0.494, 0.184, 0.556), (0.466, 0.674, 0.188)]
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in", "mathtext.fontset": "cm",
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "legend.frameon": False,
    "legend.fontsize": 7, "axes.labelsize": 8})
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures") + os.sep
rng = np.random.default_rng(7)

def logistic_orbit(r, n, burn=500):
    x = rng.uniform(0.2, 0.8)
    for _ in range(burn):
        x = r*x*(1-x)
    out = np.empty(n)
    for i in range(n):
        x = r*x*(1-x); out[i] = x
    return out

def embed(x, m=2):
    n = len(x)-(m-1)
    return np.column_stack([x[j:j+n] for j in range(m)])

def cdist(X, Y):
    return np.sqrt(((X[:, None, :]-Y[None, :, :])**2).sum(-1)).mean(0)

def drift(t):
    s = np.where(t < 0.4, 3*(t/0.4)**2 - 2*(t/0.4)**3, 1.0)
    d = np.clip((t-0.6)/0.4, 0, 1)
    return s - 0.5*(3*d**2 - 2*d**3)

def sod_track(Y, tol=1e-8):
    Yc = Y - Y.mean(0)
    Uo, so, Vto = np.linalg.svd(Yc, full_matrices=False)
    k = int(np.sum(so > tol*so[0])); B = Vto[:k].T
    Z = Yc @ B
    U, V, Xg, c, s = gsvd(Z, np.diff(Z, axis=0))
    PsiZ = np.linalg.inv(Xg.T)
    return (c/s)**2, Z @ PsiZ, B @ PsiZ

Nwin, npts, m, L = 400, 3000, 2, 60
tw = np.linspace(0, 1, Nwin)
r_true = 3.5 + 0.5*drift(tw)
Xs = [embed(logistic_orbit(r, npts), m) for r in r_true]
allpts = np.vstack(Xs); lo, hi = allpts.min(0), allpts.max(0)
Yref = (lo+hi)/2 + rng.uniform(-1, 1, (L, m))*(hi-lo)/2
Y = np.array([cdist(Xl, Yref) for Xl in Xs])
lam2, Q, Psi = sod_track(Y)
# stationary null, same windows and reference points
Ynull = np.array([cdist(embed(logistic_orbit(3.9, npts), m), Yref) for _ in range(Nwin)])
lam2n, Qn, _ = sod_track(Ynull)
# jump vectors and their projection on the SPMs
inc = np.abs(np.diff(Y, axis=0)); score = np.linalg.norm(inc, axis=1)
jumps = np.where(score > 10*np.median(score))[0]; Delta = inc[jumps]
Pn = Psi/np.linalg.norm(Psi, axis=0)
frac = np.linalg.norm(Delta @ Pn, axis=0)**2/np.linalg.norm(Delta)**2
vr = rng.standard_normal((L, 200)); vr /= np.linalg.norm(vr, axis=0)
frac_rand = np.mean(np.linalg.norm(Delta @ vr, axis=0)**2)/np.linalg.norm(Delta)**2
# baselines: moment features, far reference points
Ymom = np.array([np.r_[Xl.mean(0), np.cov(Xl.T, bias=True)[np.triu_indices(m)]] for Xl in Xs])
Yfar_ref = (lo+hi)/2 + rng.standard_normal((L, m)); Yfar_ref = (lo+hi)/2 + (Yfar_ref-(lo+hi)/2)/np.linalg.norm(Yfar_ref-(lo+hi)/2, axis=1)[:, None]*8*((hi-lo)/2).mean()
Yfar = np.array([cdist(Xl, Yfar_ref) for Xl in Xs])
_, Qm, _ = sod_track(Ymom); _, Qf, _ = sod_track(Yfar)
rho_m = np.corrcoef(Qm[:, 0], r_true)[0, 1]; rho_f = np.corrcoef(Qf[:, 0], r_true)[0, 1]
# affine fit of r onto SOC1
A = np.column_stack([np.ones(Nwin), Q[:, 0]]); a0, a1 = np.linalg.lstsq(A, r_true, rcond=None)[0]
rho = np.corrcoef(Q[:, 0], r_true)[0, 1]
print(f"J={len(jumps)} jumps, rank {np.linalg.matrix_rank(Delta, tol=1e-2*np.linalg.svd(Delta,compute_uv=False)[0])}, "
      f"corr={rho:.5f}, SOV1={lam2[0]:.0f}, SOV2={lam2[1]:.0f}, null SOV1={lam2n[0]:.2f}, null SOV2={lam2n[1]:.2f}, "
      f"jump energy along SPM1={frac[0]:.2e} (random direction {frac_rand:.2e}); "
      f"moment baseline corr={rho_m:.4f}, far-point corr={rho_f:.4f}")

fig, ax = plt.subplots(2, 3, figsize=(6.5, 3.5))
ax = ax.ravel()
ax[0].plot(np.arange(Nwin), r_true, color=COL[0])
ax[0].set_xlabel(r"window $\ell$"); ax[0].set_ylabel(r"$r_\ell$")
for j, c in zip([3, 17, 41], COL[:3]):
    ax[1].plot(np.arange(Nwin), Y[:, j], color=c, lw=0.8)
ax[1].set_xlabel(r"window $\ell$"); ax[1].set_ylabel(r"$D(\phi_\ell,\mathbf{y}_j)$")
k = np.arange(1, 11)
ax[2].semilogy(k, lam2[:10], "o", color=COL[0], ms=3.5, label="drifting")
ax[2].semilogy(k, lam2n[:10], "s", color=COL[1], ms=3.5, mfc="white", label="stationary")
ax[2].set_xlabel(r"$p$"); ax[2].set_ylabel(r"$\lambda_p^2$"); ax[2].legend(loc="upper right")
ax[2].set_xticks(k)
sgn = np.sign(a1)
ax[3].plot(np.arange(Nwin), sgn*Q[:, 0], color=COL[0], label=r"$\mathrm{SOC}_1$")
ax[3].plot(np.arange(Nwin), (r_true-a0)/a1*sgn, "--", color=COL[1], label=r"$(r_\ell-a_0)/a_1$")
ax[3].set_xlabel(r"window $\ell$"); ax[3].set_ylabel(r"$q_1$"); ax[3].legend(loc="lower right")
ax[4].plot(r_true, sgn*Q[:, 0], ".", color=COL[1], ms=2.5)
rr = np.linspace(r_true.min(), r_true.max(), 2)
ax[4].plot(rr, sgn*(rr-a0)/a1, "k-", lw=0.8)
ax[4].set_xlabel(r"$r_\ell$"); ax[4].set_ylabel(r"$q_1$")
ax[4].text(0.05, 0.9, rf"$\rho = {rho:.4f}$", transform=ax[4].transAxes)
def unit(q): return (q-q.mean())/q.std()*np.sign(np.corrcoef(q, r_true)[0, 1])
ax[5].plot(r_true, unit(Qm[:, 0]), ".", color=COL[3], ms=2.5, label=rf"moments, ${rho_m:.3f}$")
ax[5].plot(r_true, unit(Qf[:, 0]), ".", color=COL[4], ms=2.5, label=rf"far points, ${rho_f:.4f}$")
ax[5].set_xlabel(r"$r_\ell$"); ax[5].set_ylabel(r"$q_1$ (standardized)"); ax[5].legend(loc="upper left")
for a, lab in zip(ax, "abcdef"):
    a.text(-0.28, 1.02, f"({lab})", transform=a.transAxes, fontsize=8)
fig.tight_layout(w_pad=1.0, h_pad=0.8)
fig.savefig(FIGDIR + "metrics_toy.pdf")
print("wrote", FIGDIR + "metrics_toy.pdf")
# check: is SOC_2 a nonlinear function of r_l?
A2 = np.column_stack([np.ones(Nwin), r_true, r_true**2, r_true**3])
for p in (0, 1, 2):
    fit = A2 @ np.linalg.lstsq(A2, Q[:, p], rcond=None)[0]
    print(f"SOC_{p+1}: R^2 on cubic in r = {1-((Q[:,p]-fit)**2).sum()/((Q[:,p]-Q[:,p].mean())**2).sum():.4f}")
# ---- remaining numbers quoted in the exbox / text (same reference set as the figure)
def corr(a, b): return abs(np.corrcoef(a, b)[0, 1])
print(f"max/median column increment = {inc.max()/np.median(inc):.0f}")
print(f"affine-fit residual rms = {np.std(r_true - A@np.array([a0, a1])):.4f}; corr(SOC1, time) = {corr(Q[:,0], tw):.3f}")
print(f"null SOC1 std / drifting SOC1 std = {Qn[:,0].std()/Q[:,0].std():.3f}")
# uncentred
U_, V_, Xg_, c_, s_ = gsvd(Y, np.diff(Y, axis=0)); Qu = Y @ np.linalg.inv(Xg_.T); l2u = (c_/s_)**2
print(f"uncentred: corr(SOC1, r) = {corr(Qu[:,0], r_true):.3f}, std/|mean| = {Qu[:,0].std()/abs(Qu[:,0].mean()):.1e}, SOV1/SOV2 = {l2u[0]/l2u[1]:.1e}")
Am = np.column_stack([np.ones(Nwin), Ymom])
for name, F in [("far", Yfar), ("near", Y)]:
    fit = Am @ np.linalg.lstsq(Am, F, rcond=None)[0]
    print(f"R^2 of {name} features on moments = {1-((F-fit)**2).sum()/((F-F.mean(0))**2).sum():.6f}")
print(f"jump-matrix rank (rel tol 1e-2) = {np.linalg.matrix_rank(Delta, tol=1e-2*np.linalg.svd(Delta,compute_uv=False)[0])}")
sig = 0.1
Yn = np.array([cdist(Xl + sig*rng.standard_normal(Xl.shape), Yref) for Xl in Xs])
Yfn = np.array([cdist(Xl + sig*rng.standard_normal(Xl.shape), Yfar_ref) for Xl in Xs])
_, Qn1, _ = sod_track(Yn); _, Qfn1, _ = sod_track(Yfn)
print(f"noise sigma={sig} (signal std {allpts.std():.2f}): corr near {corr(Qn1[:,0], r_true):.4f}, far {corr(Qfn1[:,0], r_true):.4f}")
