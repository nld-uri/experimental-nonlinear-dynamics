"""Figure noise.pdf (Chapter 17, Section 17.6): characteristic distance to one
reference point against the Rossler parameter a, for the clean record and with
isotropic Gaussian measurement noise at 20, 60 and 100 percent of the signal
RMS.  Also checks the predicted shift (m-1) sigma^2/(2 rho) of eq. (noiseshift).
Runs in about half a minute."""
import time
import numpy as np
from _figstyle import COL, plt, FIGDIR
from _ch17data import rossler_setup, chardist, R_L

LEVELS = [0.0, 0.2, 0.6, 1.0]
t0 = time.time()
a, traj, ref = rossler_setup()
rms = np.sqrt((traj.reshape(-1, 3) - traj.reshape(-1, 3).mean(0))**2).mean()
rms = float(np.sqrt(np.mean((traj.reshape(-1, 3)
                             - traj.reshape(-1, 3).mean(0))**2)))
print(f"per-coordinate RMS of the Rossler record: {rms:.3f}")

# The figure uses the reference point nearest the attractor, the same one as
# cdistance.pdf(a), because there the curve has structure to look at.  The
# far-field prediction of eq. (noiseshift) is checked on it and on the
# median-distance point, where the expansion is valid.
rho = np.linalg.norm(ref - traj.reshape(-1, 3).mean(axis=0), axis=1)
order = np.argsort(rho)
jn, jf = int(order[0]), int(order[R_L//2])
y2 = ref[[jn, jf]]
print(f"near reference point {np.round(ref[jn], 2)}, rho = {rho[jn]:.2f}")
print(f"far  reference point {np.round(ref[jf], 2)}, rho = {rho[jf]:.2f}")

rng = np.random.default_rng(3)
curves, curvesf = [], []
for lev in LEVELS:
    sig = lev*rms
    tr = traj + sig*rng.standard_normal(traj.shape)
    D2 = chardist(tr, y2)
    curves.append(D2[:, 0]); curvesf.append(D2[:, 1])
    print(f"noise {100*lev:3.0f}%: sigma = {sig:5.2f}; near point shift "
          f"{np.mean(curves[-1] - curves[0]):6.3f} vs predicted "
          f"{2*sig**2/(2*rho[jn]):6.3f}; far point shift "
          f"{np.mean(curvesf[-1] - curvesf[0]):6.3f} vs predicted "
          f"{2*sig**2/(2*rho[jf]):6.3f}")

fig, ax = plt.subplots(figsize=(5.0, 3.0))
styles = ["-", "-", "--", ":"]
for D, lev, st, c in zip(curves, LEVELS, styles, [COL[1], COL[0], COL[0], COL[0]]):
    lab = "original" if lev == 0 else f"{100*lev:.0f}% noise"
    ax.plot(a, D, st, color=c, lw=0.7, label=lab)
ax.set_xlabel("$a$"); ax.set_ylabel("$D$")
ax.set_xlim(0.1, 0.4)
ax.legend(loc="upper left", fontsize=7)
fig.tight_layout()
fig.savefig(FIGDIR + "noise.pdf")
print("wrote figures/noise.pdf  (%.0f s)" % (time.time() - t0))
