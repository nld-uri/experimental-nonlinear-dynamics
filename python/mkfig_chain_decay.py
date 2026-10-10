"""chain_decay.pdf -- free decay of the Rayleigh-damped 3-DOF chain and
the logarithmic-decrement estimate (Ch. 19). (a) q_3(t) released in the
first mode alone, q(0) = u_1, and released from q(0) = [1,0,0]^T, which
mixes all three; (b) log of the successive positive peaks of the pure
mode-1 decay against cycle number, whose slope is -delta."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.linalg import eigh

COLS = ["#0072BD", "#D95319", "#EDB120"]
plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "mathtext.fontset": "cm", "font.family": "serif"})

M = np.diag([2., 1., 1.])
K = np.array([[2., -1, 0], [-1, 2, -1], [0, -1, 1]])
lam2, U = eigh(K, M); wp = np.sqrt(lam2); U *= np.sign(U[0])
A2 = np.array([[1/wp[0], wp[0]], [1/wp[2], wp[2]]])/2
alpha, beta = np.linalg.solve(A2, [0.02, 0.05]); C = alpha*M + beta*K
zp = (alpha/wp + beta*wp)/2
A = np.block([[np.zeros((3, 3)), np.eye(3)],
              [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]])
t = np.linspace(0, 120, 24001)
def run(q0):
    s = solve_ivp(lambda t, y: A @ y, [0, t[-1]], np.r_[q0, 0, 0, 0],
                  t_eval=t, rtol=1e-10, atol=1e-12)
    return s.y[2]                                # q_3(t)
q_mode = run(U[:, 0]); q_mix = run([1., 0, 0])
pk, _ = find_peaks(q_mode)
dl = np.log(q_mode[pk[:-1]]/q_mode[pk[1:]])
delta = dl.mean(); zest = delta/np.sqrt(4*np.pi**2 + delta**2)
print("delta =", delta, " zeta_est =", zest, " zeta_1 =", zp[0],
      " period =", np.diff(t[pk]).mean(), " 2pi/w_d =", 2*np.pi/(wp[0]*np.sqrt(1-zp[0]**2)))

fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.3), gridspec_kw={"width_ratios": [2, 1]})
a = ax[0]
a.plot(t, q_mix, color=COLS[1], lw=0.7, label=r"$\mathbf{q}(0)=[1,0,0]^{\mathrm{T}}$")
a.plot(t, q_mode, color=COLS[0], label=r"$\mathbf{q}(0)=\mathbf{u}_1$")
a.plot(t[pk], q_mode[pk], "o", ms=2.5, color=COLS[0], mfc="none")
a.plot(t, q_mode[pk[0]]*np.exp(-zp[0]*wp[0]*(t - t[pk[0]])), "k:", lw=0.8)
a.set_xlabel(r"$t$  (s)"); a.set_ylabel(r"$q_3(t)$  (m)")
a.legend(frameon=False, fontsize=7.5, loc="upper right")
a.text(0.02, 0.06, "(a)", transform=a.transAxes)
a = ax[1]
n = np.arange(pk.size)
a.plot(n, np.log(q_mode[pk]), "o", ms=3, color=COLS[0])
p = np.polyfit(n, np.log(q_mode[pk]), 1)
a.plot(n, np.polyval(p, n), "k-", lw=0.8)
a.text(0.35, 0.8, rf"slope $=-\delta=-{-p[0]:.4f}$" "\n" rf"$\hat\zeta={zest:.4f}$",
       transform=a.transAxes, fontsize=7.5)
a.set_xlabel("cycle $j$"); a.set_ylabel(r"$\ln x_j$")
a.text(0.04, 0.06, "(b)", transform=a.transAxes)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "chain_decay.pdf"))
