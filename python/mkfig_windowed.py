"""windowed.pdf (label recwin): the rectangular window w(t) = 1 on |t| <= a, 0 elsewhere
(a = 5 s), and the windowed signal x_w(t) = w(t) x(t) for the band-limited signal of
_ch05data shifted so that the record is centred on t = 0 (x(t + 10))."""
import numpy as np
from _figstyle import *
from _ch05data import signal

A = 5.0
t = np.arange(-10, 10, 1e-3); x = signal(t+10)
w = (np.abs(t) <= A).astype(float)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.5, 2.2), sharex=True)
a1.plot(t, w, color=COL[0], lw=1.0); sketch_axes(a1, "$t$", "$w(t)$"); a1.set_ylim(-0.2, 1.4)
for xx, lab in ((-A, "$-a$"), (0, "0"), (A, "$a$")): a1.text(xx+0.15, -0.08, lab, fontsize=8, va="top")
a1.text(0.15, 1.02, "1", fontsize=8, va="bottom")
a2.plot(t, w*x, color=COL[0], lw=0.7); sketch_axes(a2, "$t$", "$x_w(t)$"); a2.set_ylim(-3.2, 3.2)
for ax in (a1, a2):
    ax.axvline(-A, color="k", lw=0.5, ls=":"); ax.axvline(A, color="k", lw=0.5, ls=":")
a1.set_xlim(-10, 10)
fig.subplots_adjust(left=0.04, right=0.95, bottom=0.03, top=0.92, hspace=0.35); fig.savefig(FIGDIR+"windowed.pdf")
