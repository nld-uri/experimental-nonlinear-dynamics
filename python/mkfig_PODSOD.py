"""POD_Geometry.pdf and SOD_Geometry.pdf -- the two decompositions on one 2-D toy
data set, computed (SVD and GSVD via the generalized eigenproblem) rather than drawn.

Data: m = M points y_k = a1 cos(w1 k) u1 + a2 cos(w2 k + 1) u2, k = 0..M-1, sampled at
dt = 1, plus Gaussian measurement noise of standard deviation SIG in each channel
(seed 0), with incommensurate frequencies w1, w2 (rad/sample), amplitudes a1, a2 and
NON-orthogonal unit component directions u1 (at 0 deg) and u2 (at ANG deg). Columns
are centred. POD (SVD of Y) returns the orthogonal axes of the covariance ellipse,
which are not the component directions; SOD (pencil R_Y psi = lambda^2 R_V psi with
v_k = y_{k+1} - y_k) returns SOMs phi_p along u1 and u2 and SPMs psi_p bi-orthogonal
to them, with SOVs lambda_p^2 = 1/(2 sin(w_p/2))^2 (slow component first).
POD_Geometry: the cloud, the 1-sigma covariance ellipse (semi-axes sigma_p/sqrt(m-1))
and the POMs scaled to those semi-axes; the component directions are dashed.
SOD_Geometry: y_k, y_{k+1}, v_k for one k = K, the unit SPMs, the SOMs, the
projections q_kp = y_k^T psi_p on the SPMs and the decomposition y_k = sum q_kp phi_p.
"""
from _figstyle import *
from scipy.linalg import eigh

M, W1, W2, A1, A2, ANG, K, SIG = 2000, 0.15, 0.9, 2.0, 1.0, 60.0, 76, 0.05
k = np.arange(M)
u1 = np.array([1.0, 0.0]); u2 = np.array([np.cos(np.radians(ANG)), np.sin(np.radians(ANG))])
Y = np.outer(A1*np.cos(W1*k), u1) + np.outer(A2*np.cos(W2*k + 1.0), u2)
Y = Y + SIG*np.random.default_rng(0).standard_normal(Y.shape)   # measurement noise
Y = Y - Y.mean(axis=0)
V = Y[1:] - Y[:-1]                      # finite-difference velocities, dt = 1
# POD
U, sig, PhiT = np.linalg.svd(Y, full_matrices=False)
Phi = PhiT.T
sd = sig/np.sqrt(M - 1)                # semi-axes of the covariance ellipse
# SOD: generalized eigenproblem of the pencil (R_Y, R_V) on the same rows
RY = Y[:-1].T @ Y[:-1]/(M - 2); RV = V.T @ V/(M - 2)
lam2, Psi = eigh(RY, RV)               # ascending; take descending (slow first)
order = np.argsort(lam2)[::-1]; lam2 = lam2[order]; Psi = Psi[:, order]
Psi = Psi/np.linalg.norm(Psi, axis=0)  # unit SPMs
PhiS = np.linalg.inv(Psi).T            # SOMs, bi-orthogonal to the SPMs
for p in range(2):                     # orient the SOMs along +u1, +u2 for the drawing
    if PhiS[:, p] @ (u1 if p == 0 else u2) < 0:
        PhiS[:, p] *= -1; Psi[:, p] *= -1
print("POD: sigma_p/sqrt(m-1) = %.3f, %.3f; POM angles %.1f, %.1f deg" %
      (sd[0], sd[1], *np.degrees(np.arctan2(Phi[1], Phi[0]))))
print("SOD: SOVs %.2f, %.2f  (theory 1/(2 sin(w/2))^2 = %.2f, %.2f); SOM angles %.1f, %.1f deg "
      "(components at 0, %.0f); SPM angles %.1f, %.1f deg" %
      (lam2[0], lam2[1], 1/(2*np.sin(W1/2))**2, 1/(2*np.sin(W2/2))**2,
       *np.degrees(np.arctan2(PhiS[1], PhiS[0])), ANG, *np.degrees(np.arctan2(Psi[1], Psi[0]))))

# ---- POD_Geometry --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(Y[:, 0], Y[:, 1], ".", ms=2.0, color="0.8", mew=0, zorder=0)
th = np.linspace(0, 2*np.pi, 200)
ell = Phi @ np.vstack([sd[0]*np.cos(th), sd[1]*np.sin(th)])
ax.plot(ell[0], ell[1], color=COL[1], lw=1.2)
for p, lab in enumerate([r"$\sigma_1\hat{\phi}_1/\sqrt{m-1}$", r"$\sigma_2\hat{\phi}_2/\sqrt{m-1}$"]):
    v = Phi[:, p]*sd[p]
    v = v if v[0] >= 0 else -v
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=COL[1], lw=1.4))
    pos = (1.55, 0.2) if p == 0 else (0, -0.85)
    ax.text(*pos, lab, color=COL[1], fontsize=8.5)
for u, lab, dx in [(u1, r"$\hat{u}_1$", (2.7, 0)), (u2, r"$\hat{u}_2$", (1.1, 1.7))]:
    ax.plot([-2.6*u[0], 2.6*u[0]], [-2.6*u[1], 2.6*u[1]], "--", color="0.55", lw=0.7)
    ax.text(*dx, lab, color="0.4", fontsize=8.5)
ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-1.7, 1.9)
ax.set_xlabel("$y_1$"); ax.set_ylabel("$y_2$")
fig.tight_layout(pad=0.3)
fig.savefig(FIGDIR + "POD_Geometry.pdf")
print("wrote POD_Geometry.pdf")

# ---- SOD_Geometry --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.6, 3.6))
ax.plot(Y[:, 0], Y[:, 1], ".", ms=1.5, color="0.8", mew=0, zorder=0)
yk, yk1 = Y[K], Y[K + 1]; vk = yk1 - yk
q = Psi.T @ yk                          # q_kp = y_k^T psi_p
S = 2.6                                 # drawing length of the basis arrows
def arrow(v, o=(0, 0), col="k", lw=1.2, ls="-"):
    ax.annotate("", xy=(o[0] + v[0], o[1] + v[1]), xytext=o,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, ls=ls, shrinkA=0, shrinkB=0))
# SPM lines through the origin, and unit SPM arrows
for p in range(2):
    ps = Psi[:, p]
    ax.plot([-1.2*S*ps[0], 1.2*S*ps[0]], [-1.2*S*ps[1], 1.2*S*ps[1]], color=COL[1], lw=0.5, alpha=0.6)
    arrow(S*ps, col=COL[1], lw=1.2)
    ax.text(*(S*ps + (np.array([0.15,0.1]) if p==0 else np.array([0.2,0]))), r"$\hat{\psi}_%d$" % (p + 1), color=COL[1], fontsize=9,
            ha="center", va="center")
    ph = PhiS[:, p]/np.linalg.norm(PhiS[:, p])
    arrow(S*ph, col=COL[0], lw=1.2)
    ax.text(*(S*ph*1.1), r"$\phi_%d$" % (p + 1), color=COL[0], fontsize=9, ha="center", va="center")
    # projection of y_k on psi_p (perpendicular foot) = q_kp psi_p
    foot = q[p]*ps
    ax.plot([yk[0], foot[0]], [yk[1], foot[1]], ":", color=COL[1], lw=0.8)
    arrow(foot, col=COL[1], lw=1.0)
    lab = r"$q_{k%d}\hat{\psi}_%d$" % (p + 1, p + 1)
    ax.text(foot[0] + (-0.15 if p==0 else -0.3) , foot[1], lab, color=COL[1],
            fontsize=8, ha="center", va="bottom" if foot[1] >= 0 else "top")
# decomposition y_k = q_k1 phi_1 + q_k2 phi_2 (parallelogram)
c1, c2 = q[0]*PhiS[:, 0], q[1]*PhiS[:, 1]
ax.plot([c1[0], yk[0]], [c1[1], yk[1]], "--", color=COL[0], lw=0.8)
ax.plot([c2[0], yk[0]], [c2[1], yk[1]], "--", color=COL[0], lw=0.8)
arrow(c1, col=COL[0], lw=1.0); arrow(c2, col=COL[0], lw=1.0)
ax.text(c1[0]+0.2, - c1[1]*0.75 - 0.25, r"$q_{k1}\phi_1$", color=COL[0], fontsize=8,
        ha="center", va="bottom")
ax.text(c2[0] + 0.05, c2[1] +0.2, r"$q_{k2}\phi_2$", color=COL[0], fontsize=8,
        ha="right", va="center")
# the trajectory points and velocity
arrow(yk, col="k", lw=1.3); arrow(yk1, col="k", lw=1.0)
arrow(vk, o=yk, col=COL[3], lw=1.3)
ax.plot(*yk, "ko", ms=3); ax.plot(*yk1, "ko", ms=3)
ax.text(yk[0] + 0.15, yk[1] + 0.14, r"$y_k$", fontsize=9, ha="right", va="bottom")
ax.text(yk1[0]+0.1, yk1[1] - 0.2, r"$y_{k+1}$", fontsize=9, ha="left", va="bottom")
ax.text(*(yk + 0.5*vk + np.array([0.1, 0.02])), r"$v_k$", color=COL[3], fontsize=9,
        ha="left", va="center")
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-3.4, 3.2); ax.set_ylim(-2.6, 3.2)
fig.tight_layout(pad=0.1)
fig.savefig(FIGDIR + "SOD_Geometry.pdf")
print("wrote SOD_Geometry.pdf  k = %d, y_k = (%.2f, %.2f), q_k = (%.2f, %.2f), |v_k| = %.2f" %
      (K, yk[0], yk[1], q[0], q[1], np.linalg.norm(vk)))
