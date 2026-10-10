"""Every number quoted in Section 6.x (multitaper estimates), recomputed.

Run from python/:  python3 chk_multitaper.py
Writes nothing; prints the values that the chapter text and Fig. specest_mt
quote, with the Monte Carlo sizes used.  Seeds are fixed.

Test signal (the "line-plus-continuum" record of the section):
    x_n = AR(2) of eq:se_ar2 with rho = 0.05, dt = 1 s       (continuous part)
        + sum_h A_h cos(2 pi h f1 n + phi_h),  f1 = 0.13 Hz, h = 1,2,3
    A = (10, 0.5, 0.1), phases uniform, N = 1024.
The lines are off-bin (f1 N = 133.12).
"""
import numpy as np
from scipy.stats import f as fdist
from scipy.signal.windows import dpss
from _sestyle import ar2, ar2_psd
from psdwelch import psdwelch
from psdmt import psdmt, mtlines

N, NW, K, RHO, F1 = 1024, 4.0, 7, 0.05, 0.13
AMPS = (10.0, 0.5, 0.1)
LINES = [(h*F1, A) for h, A in zip((1, 2, 3), AMPS)]
W = NW/N
n = np.arange(N)


def record(rng):
    x = ar2(RHO, N, rng)
    for f0, A in LINES:
        x = x + A*np.cos(2*np.pi*f0*n + rng.uniform(0, 2*np.pi))
    return x


def hdr(s):
    print("\n" + s + "\n" + "-"*len(s))


# ---------------------------------------------------------------- tapers
hdr("1. Slepian tapers, N = 1024, NW = 4")
v, lam = dpss(N, NW, Kmax=K, return_ratios=True)
print("out-of-band energy 1-lambda_k:", " ".join("%.1e" % (1-l) for l in lam))
print("orthonormality error:", np.abs(v@v.T - np.eye(K)).max())
print("W = %.5f Hz, 2W = %.5f Hz = %.1f bins of 1/N" % (W, 2*W, 2*NW))
print("Hann main-lobe half-width 2/L: L=256 -> %.5f Hz, L=512 -> %.5f Hz"
      % (2/256, 2/512))

# ---------------------------------------------------------------- white noise
hdr("2. White noise, 3000 records: mean, dof, Parseval, F false alarms")
rng = np.random.default_rng(1)
GA, GP, FF, PV = [], [], [], []
for _ in range(3000):
    x = rng.standard_normal(N)
    f, S = psdmt(x, 1.0, NW, K, full=True)
    GA.append(S["G"]); GP.append(S["Gavg"]); FF.append(S["F"])
    PV.append(S["G"].sum()*f[1]/x.var())
GA, GP, FF = map(np.array, (GA, GP, FF))
inn = slice(20, f.size-20)
for nm, G in (("adaptive", GA), ("plain average", GP)):
    m, s2 = G[:, inn].mean(), G[:, inn].var()
    print("%-14s mean/(2 sigma^2 dt) = %.3f   nu = 2m^2/var = %.1f  (2K = %d)"
          % (nm, m/2, 2*m*m/s2, 2*K))
print("Parseval sum(G) df / var(x): %.4f" % np.mean(PV))
for p in (0.99,):
    thr = fdist.ppf(p, 2, 2*K-2)
    print("F_{2,%d} %.0f%% point %.2f; exceedance on white bins %.4f"
          % (2*K-2, 100*p, thr, (FF[:, inn] > thr).mean()))

# ---------------------------------------------------------------- bias & dof
hdr("3. Line-plus-continuum record, 1000 records: bias and dof")
rng = np.random.default_rng(7)
M = 1000
est = {"Welch L=256": [], "Welch L=512": [], "MT average": [], "MT adaptive": []}
NU = []
for _ in range(M):
    x = record(rng)
    fw, g = psdwelch(x, 1.0, 256, 128, "hann"); est["Welch L=256"].append(g)
    f5, g = psdwelch(x, 1.0, 512, 256, "hann"); est["Welch L=512"].append(g)
    fm, S = psdmt(x, 1.0, NW, K, full=True)
    est["MT average"].append(S["Gavg"]); est["MT adaptive"].append(S["G"])
    NU.append(S["nu"])
fq = {"Welch L=256": fw, "Welch L=512": f5, "MT average": fm, "MT adaptive": fm}
print("Welch segments: L=256 K=%d, L=512 K=%d" % ((N-256)//128+1, (N-512)//256+1))
for nm, G in est.items():
    G = np.array(G); f = fq[nm]
    T = 2*ar2_psd(RHO, f)
    r = G/T; mr = r.mean(0); nu = 2*mr**2/r.var(0)
    far = (f > 0.01) & (f < 0.49)
    for f0, _ in LINES:
        far &= np.abs(f - f0) > 0.03
    near = [mr[np.argmin(np.abs(f - (F1 + d)))] for d in (0.005, 0.01, 0.02)]
    pk = mr[np.argmin(np.abs(f - 0.0503))]
    print("%-12s far-field mean ratio %.3f (range %.2f-%.2f)  nu %.1f | "
          "at f1+0.005,+0.01,+0.02: %.2f %.2f %.2f | AR peak %.3f"
          % (nm, np.median(mr[far]), mr[far].min(), mr[far].max(),
             np.median(nu[far]), *near, pk))
NU = np.array(NU).mean(0)
far = (fm > 0.2) & (fm < 0.49) & (np.abs(fm - 0.26) > 0.03) & (np.abs(fm - 0.39) > 0.03)
print("MT adaptive nu from weights: median %.1f above 0.2 Hz, %.1f at AR peak"
      % (np.median(NU[far]), NU[np.argmin(np.abs(fm - 0.05))]))

# ---------------------------------------------------------------- F-test
hdr("4. Harmonic F-test, 1000 records, threshold at 1 - 1/N")
thr = fdist.ppf(1 - 1/N, 2, 2*K-2)
print("threshold F_{2,12}(1-1/N) = %.2f" % thr)
rng = np.random.default_rng(3)
for pad in (1, 8):
    det = np.zeros(3); fa = 0; mu1 = []
    for _ in range(M):
        x = record(rng)
        f, S = psdmt(x, 1.0, NW, K, full=True, nfft=pad*N)
        hit = S["F"] > thr; near = np.zeros(f.size, bool)
        for i, (f0, A) in enumerate(LINES):
            m = np.abs(f - f0) <= W; near |= m
            det[i] += hit[m].any()
            if i == 0:
                mu1.append(np.abs(S["mu"][m][np.argmax(S["F"][m])]))
        out = hit & ~near
        fa += np.sum(np.diff(np.r_[0, out.astype(int)]) == 1)
    mu1 = np.array(mu1)
    print("nfft = %dN: detection P = %.3f %.3f %.3f; false clusters/record %.2f; "
          "|mu(f1)| = %.3f +- %.3f (true %.3f)"
          % (pad, *(det/M), fa/M, mu1.mean(), mu1.std(), AMPS[0]/2))


# ---------------------------------------------------------------- line removal
hdr("4b. Lines removed by mtlines, then adaptive MT of the residual, 600 records")
rng = np.random.default_rng(7)
R = []
for _ in range(600):
    xc, L = mtlines(record(rng), 1.0, NW, K)
    fm, g = psdmt(xc, 1.0, NW, K); R.append(g)
R = np.array(R)/(2*ar2_psd(RHO, fm)); mr = R.mean(0); nu = 2*mr**2/R.var(0)
far = (fm > 0.01) & (fm < 0.49)
outside = far.copy()
for f0, _ in LINES:
    outside &= np.abs(fm - f0) > W
print("mean ratio outside +-W of the lines: median %.3f, range %.2f-%.2f; nu median %.1f"
      % (np.median(mr[outside]), mr[outside].min(), mr[outside].max(), np.median(nu[far])))
for d in (0.005, 0.01, 0.02):
    print("  at f1+%.3f: %.2f" % (d, mr[np.argmin(np.abs(fm - (F1 + d)))]))
for f0, _ in LINES:
    m = np.abs(fm - f0) <= W
    print("  inside +-W of %.2f Hz: mean ratio %.2f   ((K-1)/K = %.2f)" % (f0, mr[m].mean(), (K-1)/K))
rng = np.random.default_rng(99)
nl = [len(mtlines(rng.standard_normal(N), 1.0, NW, K)[1]) for _ in range(1000)]
print("white noise, p = 1/N, pad 8: %.2f spurious lines per record" % np.mean(nl))

# ---------------------------------------------------------------- two-well
hdr("5. Two-well record: F-test against the synchronous average")
from _twowell import twowell, FS, FD, NPP
x = twowell(detrend=True); Nt = x.size
P = x.reshape(-1, NPP).mean(0); C = np.fft.rfft(P)/NPP
f, S = psdmt(x, FS, NW, K, full=True)
thr_t = fdist.ppf(1 - 1/Nt, 2, 2*K-2)
print("N = %d, threshold F(1-1/N) = %.1f, F_{2,12}(0.99) = %.2f"
      % (Nt, thr_t, fdist.ppf(0.99, 2, 2*K-2)))
for h in range(1, 8):
    j = np.argmin(np.abs(f - h*FD))
    print("  %2d Hz  F = %8.1f  |mu| = %.3e V   synchronous average |c_h| = %.3e V"
          % (h*FD, S["F"][j], np.abs(S["mu"][j]), np.abs(C[h])))
det = f[S["F"] > thr_t]
print("all detections above 1-1/N (Hz):", np.round(det, 3))
lp = 2*sum(np.abs(S["mu"][np.argmin(np.abs(f - h*FD))])**2 for h in range(1, 8))
print("power in harmonic lines %.4f V^2 of variance %.4f V^2 (%.1f%%)"
      % (lp, x.var(), 100*lp/x.var()))
print("synchronous-average power %.4f V^2" % (2*np.sum(np.abs(C[1:NPP//2])**2) + np.abs(C[NPP//2])**2))

# ---------------------------------------------------------------- leakage limit
hdr("6. Two-well record: the F-test where the spectrum is 90 dB down")
for k_ in (7, 5):
    f, S = psdmt(x, FS, NW, k_, full=True)
    q50, q99 = fdist.ppf([0.5, 0.99], 2, 2*k_-2)
    for lo, hi in ((12, 18), (42, 48), (65, 78)):
        m = (f > lo) & (f < hi)
        print("K = %d, %2d-%2d Hz: median F %.2f (F_{2,%d} median %.2f), "
              "P(F > 99%% point) = %.4f, G ~ %.1e V^2/Hz, adaptive nu %.1f"
              % (k_, lo, hi, np.median(S["F"][m]), 2*k_-2, q50,
                 np.mean(S["F"][m] > q99), np.median(S["G"][m]), np.median(S["nu"][m])))
j = np.argmin(np.abs(f - FD))
print("peak level at 10 Hz (K=5 adaptive) %.2e V^2/Hz" % S["G"][j])

# ---------------------------------------------------------------- misc numbers
hdr("7. Numbers quoted in the prose")
L = 1024; w = np.hanning(L+1)[:-1]                   # periodic Hann
Wf = np.abs(np.fft.fft(w, 64*L))**2; fr = np.fft.fftfreq(64*L)
print("Hann energy outside main lobe +-2/L: %.1e" % (Wf[np.abs(fr) > 2/L].sum()/Wf.sum()))
fq = np.array([0.13, 0.26, 0.39])
Gc = 2*ar2_psd(RHO, fq)
for (f0, A), g in zip(LINES, Gc):
    print("line %.2f Hz: power A^2/2 = %.4g; continuum in a 1/256 Hz bin %.3g -> %.1f dB"
          % (f0, A*A/2, g/256, 10*np.log10(A*A/2/(g/256))))
ff = np.linspace(0, 0.5, 200001); S = 2*ar2_psd(RHO, ff)
print("AR(2) dynamic range %.1f dB" % (10*np.log10(S.max()/S[-1])))
hp = ff[S >= S.max()/2]; print("AR(2) half-power width %.4f Hz = %.2f W" % (hp[-1]-hp[0], (hp[-1]-hp[0])/W))
rng = np.random.default_rng(7)
acc = {"Welch L=256": [], "MT adaptive": []}
for _ in range(600):
    x = record(rng)
    fw, g = psdwelch(x, 1.0, 256, 128, "hann"); acc["Welch L=256"].append(g)
    fm, g = psdmt(x, 1.0, NW, K); acc["MT adaptive"].append(g)
for nm, fq in (("Welch L=256", fw), ("MT adaptive", fm)):
    r = np.mean(acc[nm], 0)/(2*ar2_psd(RHO, fq))
    m = (fq > F1) & (fq < F1 + 0.012)
    print(nm, "profile (offset/W: ratio):", " ".join("%.2f:%.2f" % ((a-F1)/W, b) for a, b in zip(fq[m], r[m])))
