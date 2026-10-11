"""chk_d2_sensitivity.py -- sensitivity of the correlation dimension of the
two-well oscillator record (Section~sec:d2twowell, eq. twowellD2) and the
phase-lock structure of its close pairs.

Not a listing. Ported from the referee check of 11 October 2026
(Parts A-D). Correlation sums are estimated with a random subsample of
reference points (all n points as partners), Theiler window |i-j| > w,
Euclidean norm, record normalized to zero mean and unit variance, as in c2r.py.

  A  decades spanned by the stated scaling range
  B  sensitivity of the slope: radius interval, Theiler window, three disjoint
     blocks of 30,000 points, d = 4..6; reference-subsample noise
  C  fraction of close pairs with (i-j) mod 16 == 0 (same forcing phase)
  D  phase-lock precision of the record (drift of the fs/16 line)

The noise-free model comparison (locked 16 and 64 per period, unlocked 16.618)
is in mkfig_phaselock.py. Runtime about three minutes on two CPUs.
"""

import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _twowell import twowell

t0 = time.time()
LNR = np.linspace(-6.5, 1.0, 76)          # the book's grid of ln(eps/sigma)
R2 = np.exp(2*LNR)


def embed(x, d, tau, n):
    return np.stack([x[i*tau:i*tau + n] for i in range(d)], axis=1).astype(np.float64)


def corrsum(x, d, tau, n, w, nref=3000, seed=0, start=0, retmod=None):
    """ln C(eps) on the LNR grid; reference points subsampled.
    retmod: if an int P, also return per-eps fraction of close pairs with (i-j) % P == 0."""
    x = (x - x.mean())/x.std()
    y = embed(x[start:], d, tau, n)
    rng = np.random.default_rng(seed)
    refs = np.sort(rng.choice(n, size=nref, replace=False))
    cnt = np.zeros(R2.size)
    cnt_mod = np.zeros(R2.size)
    tot = 0
    idx = np.arange(n)
    ch = max(4, int(1.5e7/(n*d)))
    for k in range(0, nref, ch):
        rr = refs[k:k + ch]
        D2 = ((y[None, :, :] - y[rr, None, :])**2).sum(-1)      # (ch, n)
        ok = np.abs(idx[None, :] - rr[:, None]) > w
        dd = D2[ok]
        tot += dd.size
        h, _ = np.histogram(dd, bins=np.concatenate([[0.0], R2]))
        cnt += np.cumsum(h)
        if retmod:
            same = ((idx[None, :] - rr[:, None]) % retmod == 0)[ok]
            h2, _ = np.histogram(dd[same], bins=np.concatenate([[0.0], R2]))
            cnt_mod += np.cumsum(h2)
    with np.errstate(divide="ignore", invalid="ignore"):
        lnc = np.log(cnt/tot)
        if retmod:
            return lnc, cnt_mod/cnt, cnt
    return lnc


def slope(lnc, lo, hi):
    m = (LNR >= lo - 1e-9) & (LNR <= hi + 1e-9) & np.isfinite(lnc)
    return np.polyfit(LNR[m], lnc[m], 1)[0]


def local(lnc):
    g = np.isfinite(lnc)
    s = np.full(lnc.size, np.nan)
    s[g] = np.gradient(lnc[g], LNR[g])
    return s


# ------------------------------------------------------------------ A
print("A. stated interval eps/sigma in [0.23, 0.70]: %.3f decades;"
      " ln-interval [-1.45,-0.35]: %.3f decades, eps/sigma %.3f..%.3f"
      % (np.log10(0.70/0.23), 1.1/np.log(10), np.exp(-1.45), np.exp(-0.35)))

# ------------------------------------------------------------------ B
x = twowell()
TAU, N = 5, 30000
RANGES = {"stated [-1.45,-0.35]": (-1.45, -0.35),
          "shift down 0.3": (-1.75, -0.65),
          "shift up 0.3": (-1.15, -0.05),
          "narrow [-1.2,-0.6]": (-1.2, -0.6),
          "lower half [-1.45,-0.9]": (-1.45, -0.9),
          "upper half [-0.9,-0.35]": (-0.9, -0.35),
          "wide [-1.8,0.0]": (-1.8, 0.0)}
BLOCKS = {"block 1 (0-30k)": 0, "block 2 (32k-62k)": 32000, "block 3 (64k-94k)": 64000}
print("\nB. slope sensitivity, tau = 5, n = 30000 per block, nref = 3000")
rows = []
for d in (4, 5, 6):
    for wname, w in (("w=5d", 5*d), ("w=160", 160)):
        for bname, st in BLOCKS.items():
            lnc = corrsum(x, d, TAU, N, w, nref=3000, seed=1, start=st)
            vals = [slope(lnc, *RANGES[k]) for k in RANGES]
            rows.append((d, wname, bname, vals))
            print("d=%d %-6s %-18s " % (d, wname, bname)
                  + " ".join("%.3f" % v for v in vals), flush=True)
print("columns:", list(RANGES))
V = np.array([r[3] for r in rows])
print("stated range, all 18 configs: min %.3f max %.3f mean %.3f sd %.3f"
      % (V[:, 0].min(), V[:, 0].max(), V[:, 0].mean(), V[:, 0].std()))
print("all ranges x all configs:     min %.3f max %.3f mean %.3f sd %.3f"
      % (V.min(), V.max(), V.mean(), V.std()))
for j, k in enumerate(RANGES):
    print("  %-26s across configs: %.3f .. %.3f (mean %.3f)"
          % (k, V[:, j].min(), V[:, j].max(), V[:, j].mean()))
for b in BLOCKS:
    sel = [i for i, r in enumerate(rows) if r[2] == b]
    print("  %-18s stated range: %.3f .. %.3f" % (b, V[sel, 0].min(), V[sel, 0].max()))
for wn in ("w=5d", "w=160"):
    sel = [i for i, r in enumerate(rows) if r[1] == wn]
    print("  %-6s stated range: mean %.3f" % (wn, V[sel, 0].mean()))
# subsampling noise: same config, different reference subsets
ss = [slope(corrsum(x, 5, TAU, N, 25, nref=3000, seed=s), -1.45, -0.35) for s in (11, 12, 13)]
print("reference-subsample noise, d=5 w=25 block 1, seeds 11-13: "
      + " ".join("%.3f" % v for v in ss))
print("elapsed %.0f s" % (time.time() - t0), flush=True)

# ------------------------------------------------------------------ C
print("\nC. measured record, d=5, block 1, w=25: fraction of close pairs with (i-j) % 16 == 0"
      " (chance level among admissible pairs ~ 1/16 = 0.0625)")
lnc, frac, cnt = corrsum(x, 5, TAU, N, 25, nref=3000, seed=2, retmod=16)
s = local(lnc)
for e in (-0.35, -0.9, -1.45, -1.8, -2.4, -3.0, -3.5, -4.0):
    j = np.argmin(abs(LNR - e))
    print("  ln(eps/sigma) = %5.2f  local slope %.2f  same-phase fraction %.3f  (pairs %d)"
          % (LNR[j], s[j], frac[j], cnt[j]))

# ------------------------------------------------------------------ D
print("\nD. phase lock of the drive line: complex amplitude at fs/16 in blocks of 300 periods")
xx = x - x.mean()
L = 4800
nb = xx.size//L
ph = []
for b in range(nb):
    seg = xx[b*L:(b + 1)*L]
    n_ = np.arange(b*L, (b + 1)*L)
    ph.append(np.angle(np.sum(seg*np.exp(-2j*np.pi*n_/16))))
ph = np.unwrap(np.array(ph))
tb = (np.arange(nb) + 0.5)*L
p = np.polyfit(tb, ph, 1)
print("  block phases (rad):", " ".join("%.3f" % v for v in ph))
print("  phase drift slope %.3e rad/sample -> relative frequency offset %.2e;"
      " total drift over record %.3f rad = %.3f samples"
      % (p[0], p[0]/(2*np.pi/16), p[0]*xx.size, p[0]*xx.size/(2*np.pi/16)))
print("elapsed %.0f s" % (time.time() - t0), flush=True)

print("total elapsed %.0f s" % (time.time() - t0))
