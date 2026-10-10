"""Numeric checks for Chapter 6 (specest). Every formula in the chapter is
checked here against a Monte Carlo or a closed-form quantity."""
import numpy as np
from scipy import signal
rng = np.random.default_rng(1)

# ---- 1. periodogram scaling, variance, chi-square dof --------------------
N, dt = 256, 0.01
fs = 1/dt
M = 4000
S0 = 3.0                          # two-sided PSD of white noise, x^2/Hz
sig = np.sqrt(S0*fs)              # variance = S0*fs  (int_{-fs/2}^{fs/2} S0 df)
x = sig*rng.standard_normal((M, N))
X = np.fft.fft(x, axis=1)
P = dt/N*np.abs(X)**2             # periodogram, two-sided
print("periodogram mean / S0 (interior bins):", P[:, 1:N//2].mean()/S0)
print("periodogram var / S0^2 (interior):", P[:, 1:N//2].var()/S0**2)
print("periodogram var / S0^2 at f=0:", P[:, 0].var()/S0**2, " at Nyquist:", P[:, N//2].var()/S0**2)
# chi-square 2 dof: P/S0 ~ chi2_2/2, i.e. exponential(1): check quantiles
q = np.quantile(P[:, 1:N//2].ravel()/S0, [0.05, 0.5, 0.95])
print("quantiles P/S0 (should be exp(1): 0.0513, 0.693, 2.996):", q)
# inconsistency: variance independent of N
for NN in [64, 1024]:
    xx = sig*rng.standard_normal((M, NN)); PP = dt/NN*np.abs(np.fft.fft(xx, axis=1))**2
    print(f"  N={NN}: var/S0^2 =", PP[:, 1:NN//2].var()/S0**2)

# ---- 2. Fejer kernel unit area, Hann/rect kernels ------------------------
T = 1.0; f = np.linspace(-200, 200, 400001); df = f[1]-f[0]
Fej = T*np.sinc(f*T)**2           # np.sinc is sin(pi x)/(pi x)
print("Fejer kernel area:", Fej.sum()*df)

# ---- 3. window table: main-lobe (-3 dB, -6 dB) widths in bins, peak sidelobe
def wstats(w):
    L = w.size; nfft = 1<<20
    W = np.abs(np.fft.fft(w, nfft)); W /= W[0]
    ff = np.arange(nfft)/nfft*L    # in bins
    WdB = 20*np.log10(W+1e-300)
    # -3 dB half width
    i3 = np.argmax(WdB[:nfft//2] < -3); i6 = np.argmax(WdB[:nfft//2] < -6)
    # first null then max sidelobe
    d = np.diff(W[:nfft//2]); inull = np.argmax(d > 0)
    psl = WdB[inull:nfft//2].max()
    enbw = L*np.sum(w**2)/np.sum(w)**2
    coh_gain = w.mean()
    return 2*ff[i3], 2*ff[i6], psl, enbw, coh_gain
L = 1024
wins = {"rect": np.ones(L), "hann": signal.windows.hann(L, sym=False),
        "hamming": signal.windows.hamming(L, sym=False),
        "blackman": signal.windows.blackman(L, sym=False),
        "flattop": signal.windows.flattop(L, sym=False)}
print("window: 3dB-width[bins] 6dB-width[bins] peak-sidelobe[dB] ENBW[bins] coherent-gain")
for k, w in wins.items():
    print(k, ["%.3f" % v for v in wstats(w)])

# ---- 4. Welch: variance reduction and dof ---------------------------------
Ntot = 256*32; x = sig*rng.standard_normal((1000, Ntot))
for name, nper, nov, win in [("Bartlett rect no overlap", 256, 0, "boxcar"),
                             ("Welch hann 50%", 256, 128, "hann"),
                             ("Welch hann 0%", 256, 0, "hann"),
                             ("Welch rect 50%", 256, 128, "boxcar")]:
    fr, Pw = signal.welch(x, fs=fs, window=win, nperseg=nper, noverlap=nov,
                          return_onesided=False, detrend=False, scaling="density", axis=-1)
    K = (Ntot-nov)//(nper-nov)
    m = Pw[:, 1:nper//2].mean(); v = Pw[:, 1:nper//2].var()
    print(f"{name}: K={K}, mean/S0={m/S0:.4f}, var*K/S0^2={v*K/S0**2:.3f}, dof=2m^2/v={2*m*m/v:.1f} (2K={2*K})")

# ---- 5. Var of sample mean with correlation; effective N; Bartlett band ----
# AR(2) of ch.3
def ar2(rho, N, rng, burn=2000):
    a1 = 2-(np.pi/10)**2-rho; a2 = rho-1
    e = rng.standard_normal(N+burn); x = np.zeros(N+burn)
    for n in range(1, N+burn-1):
        x[n+1] = a1*x[n] + a2*x[n-1] + e[n]
    return x[burn:]
def ar2_acf(rho, kmax):
    # exact ACF from the Yule-Walker recursion
    a1 = 2-(np.pi/10)**2-rho; a2 = rho-1
    r = np.zeros(kmax+1); r[0] = 1; r[1] = a1/(1-a2)
    for k in range(2, kmax+1): r[k] = a1*r[k-1] + a2*r[k-2]
    return r
def ar2_psd(rho, f, dt=1.0, s2=1.0):
    a1 = 2-(np.pi/10)**2-rho; a2 = rho-1; w = 2*np.pi*f*dt
    return s2*dt/np.abs(1 - a1*np.exp(-1j*w) - a2*np.exp(-2j*w))**2
rho = 0.05; N = 1000; M = 2000
xs = np.array([ar2(rho, N, rng) for _ in range(M)])
var_x = xs.var(); r = ar2_acf(rho, N-1)
k = np.arange(1, N)
pred = var_x/N*(1 + 2*np.sum((1-k/N)*r[1:]))
print("Var[xbar] MC:", xs.mean(axis=1).var(), " formula:", pred, " iid would be:", var_x/N)
print("effective N/(1+2 sum rho):", N/(1+2*r[1:].sum()))
# exact variance of AR(2) from PSD integral (Parseval)
ff = np.linspace(-0.5, 0.5, 200001); Sf = ar2_psd(rho, ff)
print("AR(2) var from PSD integral:", np.trapezoid(Sf, ff), " sample var:", var_x)
# Bartlett white noise band 2/sqrt(N)
w = rng.standard_normal((M, N)); w -= w.mean(axis=1, keepdims=True)
def acf(x, kmax):
    N = x.shape[-1]; F = np.fft.fft(x, 2*N, axis=-1); c = np.fft.ifft(np.abs(F)**2, axis=-1).real[..., :kmax+1]/N
    return c/c[..., :1]
rw = acf(w, 20)
print("white noise: std of rho_hat_k (k=1..20) * sqrt(N):", (rw[:, 1:].std(axis=0)*np.sqrt(N)).mean(),
      " frac outside +-2/sqrtN:", np.mean(np.abs(rw[:, 1:]) > 2/np.sqrt(N)))
# Bartlett formula for AR(2) at large lag: Var ~ (1/N) sum_j rho_j^2
ra = acf(xs, 400)
print("AR(2): sqrt(N)*std(rho_hat_300):", np.sqrt(N)*ra[:, 300].std(), " Bartlett sqrt(sum rho_j^2):", np.sqrt(np.sum(np.concatenate([r[1:][::-1], r])**2)))

# ---- 6. coherence: unaveraged = 1, bias (1-g^2)^2/nd, random error --------
def csd_est(x, y, nper, win="hann", nov=None):
    f, Pxx = signal.welch(x, window=win, nperseg=nper, noverlap=nov, return_onesided=False, detrend=False, axis=-1)
    f, Pyy = signal.welch(y, window=win, nperseg=nper, noverlap=nov, return_onesided=False, detrend=False, axis=-1)
    f, Pxy = signal.csd(x, y, window=win, nperseg=nper, noverlap=nov, return_onesided=False, detrend=False, axis=-1)
    return f, Pxx, Pyy, Pxy
nper = 128
x = rng.standard_normal((1, nper)); y = 0.5*x + rng.standard_normal((1, nper))
f, Pxx, Pyy, Pxy = csd_est(x, y, nper, nov=0)
print("unaveraged coherence (should be identically 1):", (np.abs(Pxy)**2/(Pxx*Pyy)).min(), (np.abs(Pxy)**2/(Pxx*Pyy)).max())
# bias with nd averages, no overlap rect window (so nd exact)
M = 4000
for g2 in [0.0, 0.5, 0.9]:
    for nd in [8, 32]:
        x = rng.standard_normal((M, nd*nper))
        # y = a x + n with a chosen so coherence g2: g2 = a^2/(a^2+1)  => a^2 = g2/(1-g2)
        a = np.sqrt(g2/(1-g2)) if g2 < 1 else 1
        y = a*x + rng.standard_normal((M, nd*nper))
        f, Pxx, Pyy, Pxy = csd_est(x, y, nper, win="boxcar", nov=0)
        C = np.abs(Pxy)**2/(Pxx*Pyy); Ci = C[:, 1:nper//2]
        bias_mc = Ci.mean()-g2; bias_th = (1-g2)**2/nd
        if g2 > 0:
            eps_mc = Ci.std()/g2; eps_th = np.sqrt(2)*(1-g2)/(np.sqrt(g2)*np.sqrt(nd))
        else:
            eps_mc = eps_th = np.nan
        print(f"coh: g2={g2}, nd={nd}: bias MC={bias_mc:.4f} th={bias_th:.4f};  eps MC={eps_mc:.4f} th={eps_th:.4f}")

# ---- 7. H1/H2 bias with input and output noise; SDOF facts -----------------
fn, zeta, fs = 5.0, 0.02, 100.0
wn = 2*np.pi*fn; k = wn**2; m = 1.0; c = 2*zeta*wn*m
sysd = signal.cont2discrete(([1.0], [m, c, k]), 1/fs, method="zoh")
b, a = sysd[0].ravel(), sysd[1].ravel()
nper, nd = 1024, 64
u = rng.standard_normal(nper*nd)
yc = signal.lfilter(b, a, u)
fr, Hz = signal.freqz(b, a, worN=nper, whole=True, fs=fs)
Hc = 1/(k - m*(2*np.pi*fr)**2 + 1j*c*2*np.pi*fr)
i5 = np.argmin(np.abs(fr-fn))
print("ZOH vs continuous receptance at fn: |Hd|/|Hc| =", np.abs(Hz[i5])/np.abs(Hc[i5]), " phase diff [deg]:", np.degrees(np.angle(Hz[i5]/Hc[i5])))
print("peak |H| theory 1/(k 2 zeta sqrt(1-zeta^2)) =", 1/(k*2*zeta*np.sqrt(1-zeta**2)), " numeric max|Hc| =", np.abs(Hc).max(), " at f=", fr[np.argmax(np.abs(Hc))], " theory fn sqrt(1-2z^2)=", fn*np.sqrt(1-2*zeta**2))
# half-power bandwidth
ffine = np.linspace(4, 6, 200001); Hf = 1/(k - m*(2*np.pi*ffine)**2 + 1j*c*2*np.pi*ffine)
hp = ffine[np.abs(Hf) >= np.abs(Hf).max()/np.sqrt(2)]
print("half-power bandwidth:", hp[-1]-hp[0], " 2*zeta*fn =", 2*zeta*fn)
sig_out = 0.3*yc.std(); sig_in = 0.3*u.std()
def H12(x, y):
    f, Pxx, Pyy, Pxy = csd_est(x[None, :], y[None, :], nper, win="hann", nov=nper//2)
    return f, (Pxy/Pxx)[0], (Pyy/np.conj(Pxy))[0], (np.abs(Pxy)**2/(Pxx*Pyy))[0]
for label, xin, yout in [("output noise", u, yc + sig_out*rng.standard_normal(u.size)),
                         ("input noise", u + sig_in*rng.standard_normal(u.size), yc)]:
    f, H1, H2, g2 = H12(xin, yout)
    # compare at resonance and off-resonance
    for fq in [fn, 2.0, 8.0]:
        i = np.argmin(np.abs(f-fq))
        print(f"{label} f={fq}: |H1|/|H|={np.abs(H1[i])/np.abs(Hz[i]):.3f} |H2|/|H|={np.abs(H2[i])/np.abs(Hz[i]):.3f} g2={g2[i]:.3f}  g2 = H1/H2 check: {np.abs(H1[i]/H2[i]):.3f}")
# random error of |H| : eps = sqrt(1-g2)/(|g| sqrt(2 nd)), phase sd ~ same (rad)
M = 400; nd2 = 16; nper2 = 256
u = rng.standard_normal((M, nper2*nd2)); y = signal.lfilter(b, a, u, axis=-1)
y = y + 1.0*y.std()*rng.standard_normal(y.shape)
f, Pxx, Pyy, Pxy = csd_est(u, y, nper2, win="boxcar", nov=0)
H1 = Pxy/Pxx; g2 = np.abs(Pxy)**2/(Pxx*Pyy)
fr2, Hz2 = signal.freqz(b, a, worN=nper2, whole=True, fs=fs)
for fq in [fn, 3.0]:
    i = np.argmin(np.abs(fr2-fq)); g = g2[:, i].mean()
    eps_mc = (np.abs(H1[:, i])/np.abs(Hz2[i])).std(); ph_mc = np.angle(H1[:, i]/Hz2[i]).std()
    eps_th = np.sqrt(1-g)/(np.sqrt(g)*np.sqrt(2*nd2))
    print(f"|H1| random error at f={fq}: MC {eps_mc:.4f}, phase sd MC {ph_mc:.4f} rad, theory {eps_th:.4f} (mean g2={g:.3f})")
