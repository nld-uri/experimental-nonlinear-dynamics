"""chain_complexmodes.pdf -- the three modes of the 3-DOF chain with
non-proportional damping C = diag(0.3, 0.05, 0) (Ch. 19). Top row: real
and imaginary parts of each complex mode against the coordinate index,
with the real modes of the undamped chain for comparison; each mode is
scaled so its largest entry is 1, the real modes likewise. Bottom row: the
phase of each component."""
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
Cn = np.diag([0.3, 0.05, 0.0])
wp0, _, U0, _ = mdofmodes(M, 0*M, K)
wp, zp, Phi, _ = mdofmodes(M, Cn, K)
j = np.argmax(np.abs(U0), axis=0)
U0 = U0/U0[j, np.arange(3)]                  # largest entry = 1, like Phi
U0 *= np.sign(np.real(np.sum(U0*np.conj(Phi), axis=0)))   # same sign as Phi
print("omega_p:", wp, " zeta_p:", zp)
print("Phi =\n", np.round(Phi, 4))
ph = np.degrees(np.angle(Phi)); ph[ph < -90] += 360   # keep near-antiphase at +180
print("phase (deg) =\n", np.round(ph, 1))

k = np.arange(1, 4)
fig, ax = plt.subplots(2, 3, figsize=(6.5, 3.2), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1.2]})
for p in range(3):
    a = ax[0, p]
    a.axhline(0, color="k", lw=0.5)
    a.plot(k, U0[:, p], "s", ms=6, mfc="none", mec="k", mew=0.8, label="undamped $\\bm{u}_p$".replace("\\bm", "\\mathbf"))
    m, s, b = a.stem(k - 0.08, Phi[:, p].real, linefmt=COLS[0], markerfmt="o", basefmt=" ")
    plt.setp(m, color=COLS[0], ms=4); plt.setp(s, lw=1.2, color=COLS[0], zorder=3)
    m, s, b = a.stem(k + 0.08, Phi[:, p].imag, linefmt=COLS[1], markerfmt="o", basefmt=" ")
    plt.setp(m, color=COLS[1], ms=4); plt.setp(s, lw=1.2, color=COLS[1], zorder=3)
    a.set_ylim(-1.25, 1.25); a.set_xlim(0.5, 3.5)
    a.text(0.04, 0.86, f"({'abc'[p]}) mode {p+1}", transform=a.transAxes, fontsize=8)
    b = ax[1, p]
    b.axhline(0, color="k", lw=0.5); b.axhline(180, color="k", lw=0.5, ls=":")
    b.plot(k, ph[:, p], "o", color=COLS[2], ms=4)
    b.set_ylim(-25, 205); b.set_yticks([0, 90, 180]); b.set_xticks([1, 2, 3])
    b.set_xlabel("coordinate $k$")
ax[0, 0].set_ylabel(r"$u_{kp}$"); ax[1, 0].set_ylabel(r"$\arg u_{kp}$  (deg)")
h = [plt.Line2D([], [], color=COLS[0], marker="o", ms=4, label=r"$\Re\,u_{kp}$"),
     plt.Line2D([], [], color=COLS[1], marker="o", ms=4, label=r"$\Im\,u_{kp}$"),
     plt.Line2D([], [], color="k", marker="s", ms=6, mfc="none", ls="", label="undamped")]
ax[0, 2].legend(handles=h, frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "chain_complexmodes.pdf"))
