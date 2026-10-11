#!/usr/bin/env python3
"""test_python.py -- checks on the Python ports of the Appendix A codes.

Each check is one the algorithm must pass on its own terms, so a failure
points at the code and not at a tolerance. Run it from this directory.

    python3 test_python.py

Expected, on the reference machine:

    ami       AMI(0) equals H(X) to 0, for every bin count
    ami_x     reduces to ami on two identical series, to 0
    irsurr    rb matches the sorted data exactly; sb matches |FFT| to 1e-12
    gsvd      |A - U C X'| and |B - V S X'| at 1e-15, c^2 + s^2 = 1
    c2r       iid noise in d = 1, 2, 3 gives D2 = 1.00, 2.01, 2.80
    pwdim     Lorenz fixed mass D_P = 2.03   (literature 2.05)
    lyapunov  Lorenz flattest window gives lambda_1 = 0.87 (reference 0.906)
    sodfilt   constant series returns the constant exactly, all weightings
    podfilt   the same, both branches
    filters   +6.3 dB (SOD) and +4.6 dB (POD, strands) on 20 % noise
    podfilt   the defaults (cloud, u = 1) gain SNR monotonically

Copyright by David Chelidze. Python port.
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp

from ami import ami
from ami_x import ami_x
from irsurr import irsurr
from gsvd import gsvd
from c2r import c2r
from pwdim import pwdim
from lyapunov import lyapunov
from sodfilt import sodfilt
from podfilt import podfilt

FAIL = []


def check(name, ok, detail):
    print(f"  {'pass' if ok else 'FAIL'}  {name:<34} {detail}")
    if not ok:
        FAIL.append(name)


def lorenz(T=320.0, dt=0.01, drop=2000):
    """state vector s = [x; y; z]"""
    f = lambda t, s: [10*(s[1] - s[0]),
                      s[0]*(28 - s[2]) - s[1],
                      s[0]*s[1] - (8/3)*s[2]]
    S = solve_ivp(f, (0, T), [5, -5, 20], t_eval=np.arange(0, T, dt),
                  rtol=1e-10, atol=1e-12).y
    return S[0, drop:]


def main():
    rng = np.random.default_rng(0)
    print("Appendix A, Python ports\n")

    # --- A.1 average mutual information --------------------------------
    x = rng.standard_normal(20000)
    worst = 0.0
    for nb in (16, 32, 64):
        I = ami(x, 5, nb)
        h, _ = np.histogram(x, bins=nb)
        p = h/h.sum()
        p = p[p > 0]
        worst = max(worst, abs(I[0] + np.sum(p*np.log2(p))))
    check("ami: AMI(0) = H(X)", worst < 1e-9, f"max error {worst:.1e}")
    e = np.abs(ami_x(x, x, 4)[4:] - ami(x, 4)).max()
    check("ami_x reduces to ami", e < 1e-12, f"max error {e:.1e}")

    # --- A.8 surrogates ------------------------------------------------
    s = np.cumsum(rng.standard_normal(2048))
    rb, sb = irsurr(s, 50, rng)
    check("irsurr: rb matches the pdf",
          np.allclose(np.sort(rb), np.sort(s)), "exact")
    e = np.abs(np.abs(np.fft.fft(sb)) - np.abs(np.fft.fft(s))).max()
    check("irsurr: sb matches the psd", e < 1e-9, f"max error {e:.1e}")

    # --- the gsvd the SOD routines rest on ------------------------------
    worst = 0.0
    for (p, q, n) in [(11, 10, 6), (20, 19, 8), (7, 6, 4)]:
        A, B = rng.standard_normal((p, n)), rng.standard_normal((q, n))
        U, V, X, c, sg = gsvd(A, B)
        worst = max(worst,
                    np.abs(A - U @ np.diag(c) @ X.T).max(),
                    np.abs(B - V @ np.diag(sg) @ X.T).max(),
                    np.abs(c**2 + sg**2 - 1).max())
    check("gsvd identities", worst < 1e-12, f"max residual {worst:.1e}")

    # --- A.2 correlation dimension --------------------------------------
    got = []
    for d in (1, 2, 3):
        lnr, lnc = c2r(rng.random(4000), d, 1, n=2500, r=40, p=0, rng=rng)
        got.append(np.mean(np.diff(lnc[10:30])/np.diff(lnr[10:30])))
    ok = all(abs(g - d) < 0.35 for g, d in zip(got, (1, 2, 3)))
    check("c2r: iid noise gives D2 = d", ok,
          "D2 = " + ", ".join(f"{g:.2f}" for g in got))

    # --- A.3 pointwise dimension ----------------------------------------
    xl = lorenz()
    lnD, lnN = pwdim(xl, 12, 3, k=200, N=500, rng=rng)
    dp = np.mean(np.diff(lnN[40:160, 1])/np.diff(lnD[40:160, 1]))
    check("pwdim: Lorenz fixed mass", abs(dp - 2.05) < 0.2,
          f"D_P = {dp:.3f}   (literature 2.05)")

    # --- A.6 largest short-time Lyapunov exponent -----------------------
    ns, lnd = lyapunov(xl, 3, 12, 4000, 1, 300, rng=rng)
    sl = np.gradient(lnd, 1.0)/0.01
    lo = min(range(5, 240), key=lambda i: sl[i:i + 40].std())
    lam = sl[lo:lo + 40].mean()
    check("lyapunov: Lorenz lambda_1", abs(lam - 0.906) < 0.15,
          f"{lam:.3f} over steps {lo}-{lo+40}   (reference 0.906)")

    # --- A.4, A.5 the weighting, isolated -------------------------------
    # A constant series: the padding cannot matter, so any correct
    # weighted average returns the constant. This is the check the
    # unnormalized edge weights used to fail.
    cs = np.full(2000, 5.0) + 1e-12*rng.standard_normal(2000)
    for w, nm in ((0, "comb"), (1, "tent"), (2, "exponential")):
        a = sodfilt(cs, 11, 12, 6, 3, w)
        check(f"sodfilt weights: {nm}", np.abs(a - 5).max() < 1e-9,
              f"max deviation {np.abs(a-5).max():.1e}")
    for f_, nm in ((0, "strands"), (1, "cloud")):
        a = podfilt(cs, 11, 12, 6, 3, f=f_)
        check(f"podfilt weights: {nm}", np.abs(a - 5).max() < 1e-9,
              f"max deviation {np.abs(a-5).max():.1e}")

    # --- and that they filter -------------------------------------------
    xc = xl[:10000]
    xc = (xc - xc.mean())/xc.std()
    xn = xc + 0.2*rng.standard_normal(xc.size)      # 20 % noise
    snr = lambda c_, e_: 20*np.log10(c_.std()/(e_ - c_).std())
    base = snr(xc, xn)
    xf = xn.copy()
    for _ in range(5):
        xf = sodfilt(xf, 11, 12, 6, 3, w=1)
    g1 = snr(xc, xf) - base
    xf = xn.copy()
    for _ in range(5):
        xf = podfilt(xf, 11, 12, 6, 3, f=0, u=1)      # strands
    g2 = snr(xc, xf) - base
    xf = xn.copy()                                    # the defaults:
    seq = []                                          # cloud, u = 1
    for _ in range(5):
        xf = podfilt(xf, 11, 12, 6, 3)
        seq.append(snr(xc, xf))
    g3 = seq[-1] - base
    mono = all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
    check("podfilt defaults are stable", mono and g3 > 2.0,
          f"{g3:+.2f} dB, monotone over 5 passes: "
          + " ".join(f"{v:.2f}" for v in seq))
    check("sodfilt gains SNR", g1 > 4.0, f"{g1:+.2f} dB over {base:.2f} dB")
    check("podfilt gains SNR", g2 > 2.0, f"{g2:+.2f} dB over {base:.2f} dB")
    check("SOD beats POD", g1 > g2, f"{g1:+.2f} vs {g2:+.2f} dB")

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed: " + ", ".join(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
