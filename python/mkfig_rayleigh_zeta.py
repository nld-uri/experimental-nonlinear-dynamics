"""rayleigh_zeta.pdf -- Rayleigh damping zeta(omega) = alpha/(2 omega) +
beta omega/2 for the 3-DOF chain (Ch. 19). (a) the curve fitted through
zeta_1 = 0.02 and zeta_3 = 0.05 with its two branches; (b) the units trap:
the same two targets fitted with frequencies in Hz, and the damping ratios
those coefficients actually produce."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E"]
plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
    "mathtext.fontset": "cm", "font.family": "serif"})

M = np.diag([2., 1., 1.])
K = np.array([[2., -1, 0], [-1, 2, -1], [0, -1, 1]])
wp = np.sqrt(np.linalg.eigvals(np.linalg.solve(M, K))); wp.sort()
z1, z3 = 0.02, 0.05
def fit(f):
    A = np.array([[1/f[0], f[0]], [1/f[2], f[2]]])/2
    return np.linalg.solve(A, [z1, z3])
alpha, beta = fit(wp)                       # rad/s, correct
a_hz, b_hz = fit(wp/(2*np.pi))              # fitted as if omega were in Hz
print("rad/s: alpha, beta =", alpha, beta)
print("Hz   : alpha, beta =", a_hz, b_hz, " -> zeta in rad/s formula:",
      (a_hz/wp + b_hz*wp)/2)

w = np.linspace(0.15, 2.2, 600)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.4))
a = ax[0]
a.plot(w, alpha/(2*w), color=COLS[1], ls="--", label=r"$\alpha/(2\omega)$")
a.plot(w, beta*w/2, color=COLS[2], ls="--", label=r"$\beta\omega/2$")
a.plot(w, alpha/(2*w) + beta*w/2, color=COLS[0], label=r"$\zeta(\omega)$")
a.plot(wp, alpha/(2*wp) + beta*wp/2, "o", color=COLS[0], ms=4)
for p in range(3):
    a.annotate(rf"$\zeta_{p+1}$", (wp[p], alpha/(2*wp[p]) + beta*wp[p]/2),
               xytext=(4, -9), textcoords="offset points", fontsize=7.5)
a.set_ylim(0, 0.08); a.set_xlabel(r"$\omega$  (rad/s)"); a.set_ylabel(r"$\zeta$")
a.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3)
a.text(0.03, 0.9, "(a)", transform=a.transAxes)
a = ax[1]
a.plot(w, alpha/(2*w) + beta*w/2, color=COLS[0], label="fitted in rad/s")
a.plot(w, a_hz/(2*w) + b_hz*w/2, color=COLS[3], label="fitted in Hz, read in rad/s")
a.plot([wp[0], wp[2]], [z1, z3], "kx", ms=5, label="targets")
a.plot(wp, (a_hz/wp + b_hz*wp)/2, "o", color=COLS[3], ms=4)
a.set_ylim(0, 0.35); a.set_xlabel(r"$\omega$  (rad/s)"); a.set_ylabel(r"$\zeta$")
a.legend(frameon=False, fontsize=7.5, loc="upper left")
a.text(0.03, 0.06, "(b)", transform=a.transAxes)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "rayleigh_zeta.pdf"))
