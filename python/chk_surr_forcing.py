"""Checks behind the revised two-well surrogate test (Section sec:surrmeasured).

Reproduces, with the seed and call order of mkfig_surr_twowell.py, the
numbers quoted in Chapter 16 for

  1. the time-reversal statistic used signed and in absolute value, and the
     effect of reversing the sensor polarity (T_rev(-x) = -T_rev(x));
  2. what the block mean / block standard deviation check does and does not
     show: the phase-locked (synchronous-average) component and the
     forcing-phase dependence of the residual standard deviation;
  3. a forcing-aware null: surrogate = synchronous average of the record plus
     an IAAFT surrogate of the residual about it (two surrogate ensembles);
  4. the arcsine counterexample to "phase randomization gives a Gaussian
     record" (a single sinusoid);
  5. the usage note that irsurr(s, 0) returns the unconverged starting series.

irsurr.py is used unmodified. Run from python/. About three minutes.
"""
import numpy as np
from scipy.stats import kurtosis
from _twowell import twowell
from irsurr import irsurr

B, NIT, LAG, NSEG, NPP = 39, 20, 5, 24000, 16


def trev(y, lag=LAG):
    """Normalized time-reversal asymmetry at the given lag, eq. (eq:trev)."""
    dy = np.asarray(y, float)[lag:] - np.asarray(y, float)[:-lag]
    return float(np.mean(dy**3)/np.mean(dy**2)**1.5)


def ranks(d, s):
    """Signed upper-tail rank and |T| rank of d among d and the surrogates s."""
    return int((s < d).sum() + 1), int((np.abs(s) < abs(d)).sum() + 1)


x = twowell()
segs = [("first 24,000", x[:NSEG]), ("whole 96,000", x)]

print("1. Book test (seed 5, as mkfig_surr_twowell.py): signed vs |T_rev|")
for name, y in segs:
    rng = np.random.default_rng(5)
    s = np.array([trev(irsurr(y, NIT, rng)[0]) for _ in range(B)])
    d = trev(y)
    rs, ra = ranks(d, s)
    print(f"  {name}: T_rev {d:.4f}; surr mean {s.mean():.4f} sd {s.std(ddof=1):.4f}"
          f" max|T| {np.abs(s).max():.4f}; signed rank {rs}/40; |T| rank {ra}/40;"
          f" z(|T|) {(abs(d) - np.abs(s).mean())/np.abs(s).std(ddof=1):.1f}")
    print(f"     polarity reversed: T_rev {trev(-y):.4f}, signed upper-tail rank"
          f" {ranks(-d, -s)[0]}/40, |T| rank {ranks(-d, -s)[1]}/40")

print("2. Block moments and cyclostationarity")
y = x - x.mean(); P = y.size//NPP; y = y[:P*NPP]
W = y.reshape(P, NPP); per = np.tile(W.mean(0), P); res = y - per
blk = (x[:P*NPP]).reshape(10, -1)
print(f"  block means {blk.mean(1).min():.3f} .. {blk.mean(1).max():.3f} V;"
      f" block sd spread {blk.std(1).max()/blk.std(1).min() - 1:.2%}")
sdp = (W - W.mean(0)).std(0)
print(f"  synchronous average carries {per.var()/y.var():.1%} of the variance;"
      f" T_rev(per) {trev(per):.4f}, T_rev(res) {trev(res):.4f}")
print(f"  residual sd by forcing phase {sdp.min():.3f} .. {sdp.max():.3f} V"
      f" (ratio {sdp.max()/sdp.min():.2f})")

print("3. Forcing-aware null: synchronous average + IAAFT(residual)")
for name, yy in segs:
    yy = yy - yy.mean(); P = yy.size//NPP; yy = yy[:P*NPP]
    pr = np.tile(yy.reshape(P, NPP).mean(0), P); rr = yy - pr
    d = trev(yy)
    for seed in (5, 6):
        rng = np.random.default_rng(seed)
        s = np.array([trev(pr + irsurr(rr, NIT, rng)[0]) for _ in range(B)])
        rs, ra = ranks(d, s)
        print(f"  {name}, seed {seed}: data {d:.4f}; surr mean {s.mean():.4f}"
              f" sd {s.std(ddof=1):.4f} max|T| {np.abs(s).max():.4f};"
              f" signed rank {rs}/40; |T| rank {ra}/40;"
              f" z {(d - s.mean())/s.std(ddof=1):.1f}")

print("4. Phase randomization of a single sinusoid (N = 4096, 200 surrogates)")
rng = np.random.default_rng(3)
N = 4096; n = np.arange(N); c = np.cos(2*np.pi*64*n/N); ks = []
for _ in range(200):
    Y = np.fft.rfft(c); ph = 2*np.pi*rng.random(Y.size); ph[0] = ph[-1] = 0
    ks.append(kurtosis(np.fft.irfft(np.abs(Y)*np.exp(1j*ph), n=N), fisher=False))
y = twowell(NSEG, detrend=True)
Y = np.fft.rfft(y); ph = 2*np.pi*rng.random(Y.size); ph[0] = ph[-1] = 0
u = np.fft.irfft(np.abs(Y)*np.exp(1j*ph), n=y.size)
print(f"  sinusoid: kurtosis {min(ks):.4f} .. {max(ks):.4f} (arcsine 1.5, Gaussian 3)")
print(f"  two-well, first 24,000: kurtosis data {kurtosis(y, fisher=False):.2f},"
      f" FT surrogate {kurtosis(u, fisher=False):.2f}")

print("5. irsurr(s, 0)")
rb, sb = irsurr(y, 0, np.random.default_rng(1))
S = np.abs(np.fft.fft(y))
print(f"  histogram exact: {np.array_equal(np.sort(rb), np.sort(y))};"
      f" max spectral error / max|S|:"
      f" {np.max(np.abs(np.abs(np.fft.fft(rb)) - S))/S.max():.3f}")

print("6. Five independent IAAFT ensembles on the first 24,000 samples (seeds 5..9)")
y = x[:NSEG]; d = trev(y)
for seed in range(5, 10):
    rng = np.random.default_rng(seed)
    s = np.array([trev(irsurr(y, NIT, rng)[0]) for _ in range(B)])
    rs, ra = ranks(d, s)
    print(f"  seed {seed}: surr mean {s.mean():.4f} sd {s.std(ddof=1):.4f}"
          f" max|T| {np.abs(s).max():.4f}; signed rank {rs}/40; |T| rank {ra}/40;"
          f" z {(d - s.mean())/s.std(ddof=1):.1f}")
