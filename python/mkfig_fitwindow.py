"""fitwindow.pdf -- the figure for the Sec. 11.5 fit-window example.

Two panels, one y-axis each (never a dual axis): the divergence curve with
three candidate fit lines, and its local slope against the reference
exponent. Colours are the book's own col1/col2 (MATLAB's first two line
colours), which pass the CVD and contrast checks as a pair; every line is
directly labelled, so identity never rests on colour alone.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL1 = "#0072BD"     # book col1
COL2 = "#D95319"     # book col2
GREY = "#59595c"
LAM = 0.9079         # Lorenz lambda_1 from the Jacobian, QR/RK4
DT = 0.01

lnd = np.load("/tmp/lnd.npy")
sl = np.load("/tmp/slope.npy")
n = np.arange(1, lnd.size + 1)

WINS = [(5, 35, 2.246), (50, 150, 0.903), (90, 290, 0.830)]

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "mathtext.fontset": "cm", "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.35))

# ---- (a) the divergence curve and three candidate fits ----------------
a = ax[0]
LEV = [-5.28, -5.60, -5.92]          # one bracket level per window
LABX = [118, 205, 296]               # where each fit line is labelled
VA = ["bottom", "bottom", "top"]
DY = [5, 5, -5]
for i, ((lo, hi, lam), lev, lx) in enumerate(zip(WINS, LEV, LABX)):
    p = np.polyfit(n[lo - 1:hi], lnd[lo - 1:hi], 1)
    a.plot([1, 300], np.polyval(p, [1, 300]), color=COL2, lw=0.8,
           ls=(0, (4, 3)), zorder=2)
    a.plot([lo, hi], [lev, lev], color=COL2, lw=2.4,
           solid_capstyle="butt", zorder=4)
    if i < 2:
        a.annotate(rf"${lo}\!-\!{hi}$", xy=(hi, lev), xytext=(4, -2.5),
                   textcoords="offset points", ha="left", va="center",
                   color=COL2, fontsize=7.5, zorder=5)
    else:
        a.annotate(rf"${lo}\!-\!{hi}$", xy=((lo + hi)/2, lev),
                   xytext=(0, -8), textcoords="offset points",
                   ha="center", va="top", color=COL2, fontsize=7.5,
                   zorder=5)
    a.annotate(rf"$\hat\lambda_1 = {lam:.2f}$",
               xy=(lx, np.polyval(p, lx)), xytext=(0, DY[i]),
               textcoords="offset points", ha="right", va=VA[i],
               color=COL2, fontsize=8, zorder=5)
a.plot(n, lnd, color=COL1, lw=1.6, zorder=3)
a.set_xlabel(r"$n$ (samples)")
a.set_ylabel(r"$\Delta_n$")
a.set_xlim(0, 300)
a.set_ylim(-6.55, -0.4)
a.set_yticks([-5, -4, -3, -2, -1])
a.grid(True, color="0.88", lw=0.5)
a.set_axisbelow(True)
a.text(0.03, 0.95, "(a)", transform=a.transAxes, fontsize=9, va="top")

# ---- (b) the local slope against the reference ------------------------
b = ax[1]
b.axhspan(0, 3, xmin=0, xmax=80/300, color="0.94", zorder=0)
b.plot(n, sl, color=COL1, lw=1.1, zorder=3)
b.axhline(LAM, color=GREY, lw=1.0, ls=":", zorder=2)
b.annotate(rf"$\lambda_1 = {LAM:.3f}$, from the Jacobian",
           xy=(300, LAM), xytext=(-4, 6), textcoords="offset points",
           ha="right", color=GREY, fontsize=8)
b.annotate("transient", xy=(40, 0.30), ha="center", color=GREY, fontsize=8)
b.annotate("scaling region", xy=(195, 0.30), ha="center", color=GREY,
           fontsize=8)
b.set_xlabel(r"$n$ (samples)")
b.set_ylabel(r"$\mathrm{d}\Delta_n/\mathrm{d}n \,/\, \Delta t$")
b.set_xlim(0, 300)
b.set_ylim(0, 2.8)
b.grid(True, color="0.88", lw=0.5)
b.set_axisbelow(True)
b.text(0.03, 0.95, "(b)", transform=b.transAxes, fontsize=9, va="top")

for q in ax:
    q.set_box_aspect(1/1.618)
fig.tight_layout(pad=0.4)
fig.savefig("/tmp/fitwindow.pdf")
fig.savefig("/tmp/fitwindow.png", dpi=200)
print("wrote /tmp/fitwindow.pdf")
