"""Numerical checks for Chapter 19 (linmodal). Every formula printed in the
chapter is tested here against the quantity it claims to be."""
import numpy as np
from scipy.linalg import eig, eigh
from scipy.integrate import solve_ivp
np.set_printoptions(precision=4, suppress=True)

# ---------------- SDOF ---------------------------------------------------
wn, z = 2.0, 0.05
tau, Delta = -2*z*wn, wn**2
lam = np.roots([1, -tau, Delta])
print("SDOF poles from (tau,Delta):", lam, " expected", -z*wn + 1j*wn*np.sqrt(1-z**2))
wd = wn*np.sqrt(1-z**2)
# log decrement: simulate free decay and measure successive peaks
t = np.linspace(0, 40, 400001)
x = np.exp(-z*wn*t)*np.cos(wd*t)
from scipy.signal import find_peaks
pk, _ = find_peaks(x)
delta = np.log(x[pk[0]]/x[pk[1]])
print("log decrement:", delta, " 2 pi zeta/sqrt(1-zeta^2) =", 2*np.pi*z/np.sqrt(1-z**2))
zeta_est = delta/np.sqrt(4*np.pi**2 + delta**2)
print("  zeta from delta:", zeta_est)
# half power bandwidth of receptance
w = np.linspace(1.5, 2.5, 2000001)
H = 1/(wn**2 - w**2 + 2j*z*wn*w)          # unit mass receptance
Hm = np.abs(H)
ipk = np.argmax(Hm)
print("receptance peak at", w[ipk], " expected wn sqrt(1-2 z^2)=", wn*np.sqrt(1-2*z**2))
print("peak magnitude", Hm[ipk], " expected 1/(2 z wn^2 sqrt(1-z^2))=", 1/(2*z*wn**2*np.sqrt(1-z**2)))
idx = np.where(Hm >= Hm[ipk]/np.sqrt(2))[0]
bw = w[idx[-1]] - w[idx[0]]
print("half power bandwidth", bw, " 2 zeta wn =", 2*z*wn, " ratio", bw/(2*z*wn))
# static receptance = 1/k, phase -90 at wn
print("phase at wn (deg):", np.degrees(np.angle(1/(wn**2-wn**2+2j*z*wn*wn))))
# Nyquist circle of mobility: i w H  -> circle center (1/(2c), 0) radius 1/(2c), c = 2 z wn (unit mass)
c = 2*z*wn
Y = 1j*w*H
print("mobility Nyquist: max Re", Y.real.max(), " expect 1/c =", 1/c, "; Re at wn", (1j*wn/(2j*z*wn*wn)).real)

# ---------------- 3-DOF chain -------------------------------------------
M = np.diag([2., 1., 1.])
K = np.array([[2., -1, 0], [-1, 2, -1], [0, -1, 1]])
lam2, U = eigh(K, M)          # scipy eigh returns U^T M U = I already
w_p = np.sqrt(lam2)
print("\nchain omega_p:", w_p)
Um = U/np.sign(U[0])          # sign convention: first entry positive
print("mass-normalized U:\n", Um)
print("U^T M U =\n", Um.T @ M @ Um)
Un = Um/np.linalg.norm(Um, axis=0)
print("unit-norm U:\n", Un)
print("unit-norm U^T M U (diag = modal masses):\n", Un.T @ M @ Un)
print("K-orthogonality: U^T K U =\n", Um.T @ K @ Um, "\n vs Omega^2", lam2)
# check against ch21's printed (unit norm) values
Uch21 = np.array([[0.3602, 0.7071, 0.2338], [0.5928, 0, -0.8524], [0.7204, -0.7071, 0.4676]])
print("max |Un - ch21 U| =", np.abs(np.abs(Un)-np.abs(Uch21)).max())

# ---------------- Rayleigh damping ---------------------------------------
w1, w3 = w_p[0], w_p[2]
z1, z3 = 0.02, 0.05
A = np.array([[1/w1, w1], [1/w3, w3]])/2
alpha, beta = np.linalg.solve(A, [z1, z3])
print("\nRayleigh alpha, beta:", alpha, beta)
C = alpha*M + beta*K
zeta_p = (alpha/w_p + beta*w_p)/2
print("zeta_p from formula:", zeta_p)
print("U^T C U diagonal / (2 w_p):", np.diag(Um.T @ C @ Um)/(2*w_p))
print("U^T C U off-diagonal max:", np.abs(Um.T @ C @ Um - np.diag(np.diag(Um.T @ C @ Um))).max())
# poles from state space
def statematrix(M, C, K):
    n = M.shape[0]
    Mi = np.linalg.inv(M)
    return np.block([[np.zeros((n, n)), np.eye(n)], [-Mi@K, -Mi@C]])
As = statematrix(M, C, K)
lams = np.linalg.eigvals(As)
lams = lams[np.argsort(np.abs(lams.imag))]
print("state poles:", lams)
print("  |lam| = omega_p:", np.abs(lams[::2]), " -Re/|lam| = zeta_p:", -lams[::2].real/np.abs(lams[::2]))
# units trap: coefficients that reproduce zeta only with f in Hz
f_p = w_p/(2*np.pi)
Ahz = np.array([[1/f_p[0], f_p[0]], [1/f_p[2], f_p[2]]])/2
alpha_hz, beta_hz = np.linalg.solve(Ahz, [z1, z3])
print("coefficients fitted in Hz:", alpha_hz, beta_hz)
print("  true zeta they give (rad/s formula):", (alpha_hz/w_p + beta_hz*w_p)/2)
print("  zeta they claim (Hz formula):", (alpha_hz/f_p + beta_hz*f_p)/2)
# minimum of the Rayleigh curve
wmin = np.sqrt(alpha/beta); zmin = np.sqrt(alpha*beta)
print("Rayleigh min at omega =", wmin, " zeta_min =", zmin)

# ---------------- non-proportional damping -------------------------------
Cn = np.diag([0.3, 0.05, 0.0])
An = statematrix(M, Cn, K)
lamn, Vn = eig(An)
order = np.argsort(np.abs(lamn.imag))
lamn, Vn = lamn[order], Vn[:, order]
print("\nnon-proportional poles:", lamn)
wn_p = np.abs(lamn[::2]); zn_p = -lamn[::2].real/wn_p
print("  omega_p:", wn_p, " zeta_p:", zn_p)
print("  U^T Cn U (not diagonal):\n", Um.T @ Cn @ Um)
# complex modes: displacement partition of eigenvectors, one per conjugate pair
Phi = Vn[:3, ::2]
# normalize each column so that its largest-magnitude entry is real positive
for k in range(3):
    j = np.argmax(np.abs(Phi[:, k])); Phi[:, k] *= np.exp(-1j*np.angle(Phi[j, k]))/np.abs(Phi[j, k])
print("complex modes (largest entry =1):\n", Phi)
print("phase of components (deg):\n", np.degrees(np.angle(Phi)))
# MAC between proportional (real) and non-proportional (complex) modes
def mac(a, b):
    return np.abs(np.vdot(a, b))**2/(np.vdot(a, a).real*np.vdot(b, b).real)
MACm = np.array([[mac(Um[:, i], Phi[:, j]) for j in range(3)] for i in range(3)])
print("MAC(real modes, complex modes):\n", MACm)
# left-right check: eigenvectors of A obey biorthogonality with left eigenvectors
# modal phase collinearity-ish: ratio of imag to real
# check the 2n first order symmetric form: A_s x_dot + B_s x = 0 with A_s=[[C,M],[M,0]], B_s=[[K,0],[0,-M]]
As2 = np.block([[Cn, M], [M, np.zeros((3, 3))]]); Bs2 = np.block([[K, np.zeros((3, 3))], [np.zeros((3, 3)), -M]])
lam_sym = eig(-Bs2, As2, right=False)
print("poles from symmetric pencil match:", np.allclose(np.sort_complex(lam_sym), np.sort_complex(lamn)))

# ---------------- FRF matrix -------------------------------------------
w = np.linspace(0.01, 2.5, 5000)
def frf_direct(M, C, K, w):
    return np.array([np.linalg.inv(K - wi**2*M + 1j*wi*C) for wi in w])
def frf_modal(Um, w_p, zeta_p, w):
    H = np.zeros((w.size, 3, 3), complex)
    for p in range(3):
        H += np.outer(Um[:, p], Um[:, p])[None]/(w_p[p]**2 - w[:, None, None]**2 + 2j*zeta_p[p]*w_p[p]*w[:, None, None])
    return H
Hd = frf_direct(M, C, K, w); Hm_ = frf_modal(Um, w_p, zeta_p, w)
print("\nmodal-sum FRF vs direct inverse (proportional): max err", np.abs(Hd-Hm_).max(), " max |H|", np.abs(Hd).max())
print("reciprocity H13 vs H31:", np.abs(Hd[:, 0, 2]-Hd[:, 2, 0]).max())
# residue form for general damping: H = sum_r phi_r phi_r^T / (a_r (i w - lam_r)) + c.c., with a_r = phi_r^T (2 lam_r M + C) phi_r
Hn = frf_direct(M, Cn, K, w)
lam_all, V_all = eig(An)
Hres = np.zeros_like(Hn)
for r in range(6):
    ph = V_all[:3, r]; lr = lam_all[r]
    a_r = ph @ (2*lr*M + Cn) @ ph
    Hres += np.outer(ph, ph)[None]/(a_r*(1j*w[:, None, None]-lr))
print("residue-form FRF vs direct (non-proportional): max err", np.abs(Hn-Hres).max())
# drive point FRF: all residues same sign -> antiresonance between each pair of resonances
H11 = Hd[:, 0, 0]; H31 = Hd[:, 2, 0]
print("drive-point residues u_1p^2:", Um[0]**2, " transfer residues u_1p u_3p:", Um[0]*Um[2])
from scipy.signal import argrelmin
print("H11 local minima (antiresonances) at w =", w[argrelmin(np.abs(H11))[0]])
print("H31 local minima at w =", w[argrelmin(np.abs(H31))[0]])
# mobility and accelerance at the peak
print("mobility |iwH| at w1:", np.abs(1j*w_p[0]*Hm_[np.argmin(abs(w-w_p[0])), 0, 0]))

# ---------------- free decay of the chain, log decrement ----------------------
def rhs(t, y):
    return As @ y
# release in the first mode only: q(0) = u_1
y0 = np.concatenate([Um[:, 0], np.zeros(3)])
sol = solve_ivp(rhs, [0, 120], y0, t_eval=np.linspace(0, 120, 60001), rtol=1e-10, atol=1e-12)
q3 = sol.y[2]
pk, _ = find_peaks(q3)
dl = np.log(q3[pk[:-1]]/q3[pk[1:]])
print("\nfree decay in mode 1: log decrement per cycle", dl[:3].mean(), " expected", 2*np.pi*zeta_p[0]/np.sqrt(1-zeta_p[0]**2))
print("  zeta est:", dl.mean()/np.sqrt(4*np.pi**2+dl.mean()**2), " true", zeta_p[0])
print("  period from peaks:", np.diff(sol.t[pk]).mean(), " expected 2 pi / wd =", 2*np.pi/(w_p[0]*np.sqrt(1-zeta_p[0]**2)))
# release from q(0) = [1,0,0]: modal participation
eta0 = Um.T @ M @ np.array([1., 0, 0])
print("modal participation of q(0)=[1,0,0]:", eta0)

# ---------------- ITD/ERA sanity: shift pair eigenvalues equal exp(lam dt) --------
dt = 0.1
Y = sol.y[:, ::100]  # dt = 0.2? t_eval spacing 0.002 -> 100 steps = 0.2
dt = 0.2
Y1, Y2 = Y[:, :-1], Y[:, 1:]
T = Y2 @ np.linalg.pinv(Y1)
th = np.linalg.eigvals(T)
print("\nshift-pair discrete poles vs exp(lam dt): ", np.sort_complex(np.log(th[np.abs(th) > 1e-6])/dt)[:2], np.sort_complex(lams)[:2])
