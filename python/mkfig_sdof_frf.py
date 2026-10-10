"""sdof_frf.pdf -- SDOF receptance for three damping ratios: magnitude,
phase and Nyquist (Ch. 19). Unit mass, omega_n = 1 rad/s, so H = 1/k at
low frequency with k = 1 N/m."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLS = ["#0072BD", "#D95319", "#EDB120"]
plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
    "mathtext.fontset": "cm", "font.family": "serif"})

wn = 1.0
w = np.linspace(0.0, 2.0, 4001)
fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.2))
for z, c in zip([0.02, 0.05, 0.10], COLS):
    H = 1/(wn**2 - w**2 + 2j*z*wn*w)
    ax[0].semilogy(w/wn, np.abs(H), color=c, label=rf"$\zeta={z}$")
    ax[1].plot(w/wn, np.degrees(np.angle(H)), color=c)
    wz = np.linspace(0.6, 1.4, 4001)           # Nyquist near resonance only
    Hz = 1/(wn**2 - wz**2 + 2j*z*wn*wz)
    ax[2].plot(Hz.real, Hz.imag, color=c)
    ax[2].plot(0, -1/(2*z*wn**2), "o", ms=3, color=c)   # the point omega = omega_n
# half-power markers for zeta = 0.05
z = 0.05
Hp = 1/(2*z*wn**2*np.sqrt(1-z**2))
ax[0].plot([1-z, 1+z], [Hp/np.sqrt(2)]*2, "k", lw=0.8)
ax[0].annotate(r"$\Delta\omega\approx 2\zeta\omega_n$", (1+z, Hp/np.sqrt(2)),
               xytext=(5, -3), textcoords="offset points", fontsize=7.5)
ax[0].set_xlabel(r"$\omega/\omega_n$"); ax[0].set_ylabel(r"$|H(\omega)|$  (m/N)")
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
ax[1].set_xlabel(r"$\omega/\omega_n$"); ax[1].set_ylabel(r"$\arg H(\omega)$  (deg)")
ax[1].set_yticks([0, -45, -90, -135, -180]); ax[1].axhline(-90, color="k", lw=0.5, ls=":")
ax[2].set_xlabel(r"$\Re\,H$  (m/N)"); ax[2].set_ylabel(r"$\Im\,H$  (m/N)")
ax[2].set_aspect("equal"); ax[2].axhline(0, color="k", lw=0.5); ax[2].axvline(0, color="k", lw=0.5)
for a, lab in zip(ax, "abc"):
    a.text(0.03, 0.05, f"({lab})", transform=a.transAxes)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "sdof_frf.pdf"))
