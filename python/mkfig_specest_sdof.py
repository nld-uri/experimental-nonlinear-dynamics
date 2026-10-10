"""specest_sdof.pdf: receptance of the SDOF oscillator (f_n = 5 Hz, zeta = 0.05,
m = 1 kg): magnitude with half-power points, and phase."""
import numpy as np
from _sestyle import *
fn, m = 5.0, 1.0
wn = 2*np.pi*fn; k = m*wn**2
f = np.linspace(0, 12, 4001)
fig, ax = plt.subplots(1, 2, figsize=(6, 2.5))
for zeta, c in [(0.05, COL[0]), (0.15, COL[1])]:
    cc = 2*zeta*wn*m
    H = 1/(k - m*(2*np.pi*f)**2 + 1j*cc*2*np.pi*f)
    ax[0].plot(f, np.abs(H)*k, color=c, label=f"$\\zeta={zeta}$")
    ax[1].plot(f, np.degrees(np.angle(H)), color=c)
    if zeta == 0.05:
        pk = np.abs(H).max()*k; hp = f[np.abs(H)*k >= pk/np.sqrt(2)]
        ax[0].plot([hp[0], hp[-1]], [pk/np.sqrt(2)]*2, color="k", lw=0.8, marker="|", ms=6)
        ax[0].annotate("$\\Delta f_{1/2} \\approx 2\\zeta f_n$", (hp[-1]+0.2, pk/np.sqrt(2)), fontsize=8, va="center")
        print("half-power width:", hp[-1]-hp[0], "2 zeta fn:", 2*zeta*fn)
ax[0].set_xlabel("$f$ (Hz)"); ax[0].set_ylabel("$k\\,|H(f)|$"); ax[0].legend(); ax[0].set_ylim(0, 11)
ax[1].set_xlabel("$f$ (Hz)"); ax[1].set_ylabel("$\\angle H(f)$ (deg)"); ax[1].set_yticks([0, -45, -90, -135, -180])
ax[1].axhline(-90, color="k", lw=0.4, ls=":"); ax[1].axvline(fn, color="k", lw=0.4, ls=":")
for a, t in zip(ax, ["(a)", "(b)"]): a.text(0.03, 0.06, t, transform=a.transAxes); a.set_xlim(0, 12)
fig.tight_layout(); fig.savefig(FIGDIR+"specest_sdof.pdf")
