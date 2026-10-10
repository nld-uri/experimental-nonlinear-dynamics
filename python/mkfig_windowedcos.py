"""windowedcos.pdf (label windowedcos): (a) X(f) = [delta(f - f_0) + delta(f + f_0)]/2, the
transform of cos(2 pi f_0 t), drawn as arrows of weight 1/2; (b) the transform of the
signal seen through the rectangular window |t| <= a, X_w(f) = [sin(2 pi (f+f_0) a)/(pi (f+f_0))
+ sin(2 pi (f-f_0) a)/(pi (f-f_0))]/2, with f_0 = 8/a (a = 1 s, f_0 = 8 Hz): peaks of height
a and main lobes of width 1/a between zeros, sidelobes leaking into all frequencies."""
import numpy as np
from _figstyle import *

a, F0 = 1.0, 8.0
f = np.linspace(-2*F0, 2*F0, 8001)
Xw = 0.5*(2*a*np.sinc(2*(f+F0)*a) + 2*a*np.sinc(2*(f-F0)*a))
print("peak of X_w = %.4f (= a = %g); first sidelobe relative to peak %.1f dB" % (Xw.max(), a, 20*np.log10(abs(Xw[(f > F0+0.5/a) & (f < F0+1/a)]).max()/Xw.max())))
fig, (p, q) = plt.subplots(1, 2, figsize=(6, 1.8), gridspec_kw=dict(width_ratios=[1, 1.4], wspace=0.25))
for s in (-F0, F0):
    p.annotate("", (s, 0.5), (s, 0), arrowprops=dict(arrowstyle="->", lw=1.0, color=COL[0]))
sketch_axes(p, "$f$", "$X(f)$"); p.set_xlim(-2*F0, 2*F0); p.set_ylim(-0.15, 0.75); p.xaxis.set_label_coords(1.02, 0.18); p.yaxis.set_label_coords(0.52, 1.0)
p.set_xticks([-F0, 0, F0]); p.set_xticklabels(["$-f_0$", "0", "$f_0$"], fontsize=8); p.tick_params(length=2)
p.text(F0+0.5, 0.5, r"$\frac{1}{2}$", fontsize=8, va="center"); panel(p, "a", x=-0.02, y=1.0)
q.plot(f, Xw, color=COL[0], lw=0.7)
sketch_axes(q, "$f$", "$X_w(f)$"); q.set_xlim(-2*F0, 2*F0); q.set_ylim(-0.3*a, 1.4*a); q.xaxis.set_label_coords(1.02, 0.18); q.yaxis.set_label_coords(0.52, 1.0)
q.set_xticks([-F0, 0, F0]); q.set_xticklabels(["$-f_0$", "0", "$f_0$"], fontsize=8); q.tick_params(length=2)
q.set_yticks([a]); q.set_yticklabels(["$a$"], fontsize=8)
q.annotate("", (F0-0.5/a, 1.15*a), (F0+0.5/a, 1.15*a), arrowprops=dict(arrowstyle="<->", lw=0.6)); q.text(F0+0.7/a, 1.15*a, r"$\frac{1}{a}$", fontsize=8, va="center")
panel(q, "b", x=-0.02, y=1.0)
fig.subplots_adjust(left=0.03, right=0.97, bottom=0.17, top=0.86); fig.savefig(FIGDIR+"windowedcos.pdf")
