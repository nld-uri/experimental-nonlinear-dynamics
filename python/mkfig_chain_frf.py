"""chain_frf.pdf -- drive-point and transfer receptances of the 3-DOF
chain with Rayleigh damping fitted to zeta_1 = 0.02, zeta_3 = 0.05 (Ch. 19).
Uses mdofmodes.py."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mdofmodes import mdofmodes

COLS = ["#0072BD", "#D95319", "#EDB120"]
plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
    "mathtext.fontset": "cm", "font.family": "serif"})

M = np.diag([2., 1., 1.])
K = np.array([[2., -1, 0], [-1, 2, -1], [0, -1, 1]])
wp0 = np.sqrt(np.linalg.eigvals(np.linalg.solve(M, K))); wp0.sort()
A = np.array([[1/wp0[0], wp0[0]], [1/wp0[2], wp0[2]]])/2
alpha, beta = np.linalg.solve(A, [0.02, 0.05])
C = alpha*M + beta*K
w = np.linspace(0.05, 2.2, 6000)
wp, zp, U, H = mdofmodes(M, C, K, w)
print("alpha, beta =", alpha, beta, " zeta_p =", zp)

fig, ax = plt.subplots(2, 1, figsize=(6.0, 3.9), sharex=True,
                       gridspec_kw={"height_ratios": [2.2, 1]})
labels = [r"$H_{11}$ (drive point)", r"$H_{21}$", r"$H_{31}$"]
for j, (c, lab) in enumerate(zip(COLS, labels)):
    ax[0].semilogy(w, np.abs(H[:, j, 0]), color=c, label=lab)
    ax[1].plot(w, np.degrees(np.unwrap(np.angle(H[:, j, 0]))), color=c)
for p in range(3):
    ax[0].axvline(wp[p], color="k", lw=0.5, ls=":")
ax[0].set_ylabel(r"$|H_{j1}(\omega)|$  (m/N)")
ax[0].legend(frameon=False, fontsize=7.5)
ax[1].set_ylabel(r"$\arg H_{j1}$  (deg)"); ax[1].set_xlabel(r"$\omega$  (rad/s)")
ax[1].set_yticks([0, -180, -360, -540])
ax[0].text(0.02, 0.06, "(a)", transform=ax[0].transAxes)
ax[1].text(0.02, 0.08, "(b)", transform=ax[1].transAxes)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "chain_frf.pdf"))
