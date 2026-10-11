function [f, G, out] = psdmt(x, fs, NW, K, adaptive, nfft)
% PSDMT Multitaper spectral estimate and harmonic F-test (Chapter 6).
%
% USE: >> [f, Gxx] = psdmt(x, fs);                       % NW = 4, K = 7, adaptive
%      >> [f, Gxx, S] = psdmt(x, fs, NW, K, adaptive, nfft);
%
% INPUT:
%   x        -- record (vector), sampled at fs (Hz); its mean is removed
%   NW       -- time-bandwidth product, default 4; half-bandwidth W = NW*fs/N
%   K        -- number of Slepian tapers, default 2*NW - 1
%   adaptive -- true (default): Thomson's adaptive weights; false: plain average
%   nfft     -- transform length >= N (zero padding), default N
%
% OUTPUT: one-sided density on f >= 0 in x^2/Hz, so sum(Gxx)*f(2) ~ var(x).
%   S.F   -- harmonic F statistic, F(2, 2K-2) under the no-line hypothesis
%   S.mu  -- complex line amplitude: a line A*cos(2*pi*f0*t + phi) has
%            |mu(f0)| = A/2, with phase at t = 0
%   S.nu  -- equivalent degrees of freedom of the estimate at each frequency
%   S.Gavg, S.lam, S.K -- plain-average estimate, taper concentrations, K
%
% Needs dpss (Signal Processing Toolbox).
% Written by David Chelidze.

x = x(:) - mean(x); N = numel(x);
if nargin < 3 || isempty(NW), NW = 4; end
if nargin < 4 || isempty(K), K = floor(2*NW) - 1; end
if nargin < 5 || isempty(adaptive), adaptive = true; end
if nargin < 6 || isempty(nfft), nfft = N; end
nfft = max(nfft, N);
[v, lam] = dpss(N, NW, K);              % N x K tapers, unit energy; lam = concentrations
lam = lam(:).';                         % 1 x K
dt = 1/fs; nf = floor(nfft/2) + 1;
f = (0:nf-1).'*fs/nfft;
Y = fft(v.*x, nfft, 1); Y = Y(1:nf, :);  % eigencoefficients, nf x K
Sk = dt*abs(Y).^2;                      % eigenspectra (two-sided densities)
Savg = mean(Sk, 2);
if adaptive
    sig2 = var(x, 1)*dt;                % white level with the record's variance
    Sa = mean(Sk(:, 1:2), 2);           % start from the two best-concentrated tapers
    for it = 1:100
        b = Sa./(Sa*lam + sig2*(1 - lam));      % nf x K
        d2 = lam.*b.^2;                         % squared weights d_k^2
        Snew = sum(d2.*Sk, 2)./sum(d2, 2);
        done = max(abs(Snew - Sa)./Sa) < 1e-6;
        Sa = Snew;
        if done, break; end
    end
    nu = 2*sum(d2, 2).^2./sum(d2.^2, 2);
else
    Sa = Savg; nu = 2*K*ones(nf, 1);
end
one = 2*ones(nf, 1); one(1) = 1;
if mod(nfft, 2) == 0, one(end) = 1; end  % Nyquist bin is not doubled
G = one.*Sa;
if nargout < 3, return; end
V0 = sum(v, 1);                         % taper sums; zero for the odd tapers
mu = (Y*V0.')/sum(V0.^2);               % least-squares line amplitude
resid = sum(abs(Y - mu*V0).^2, 2);
out.F = (K - 1)*abs(mu).^2*sum(V0.^2)./resid;
out.mu = mu; out.nu = nu; out.Gavg = one.*Savg; out.lam = lam; out.K = K;
end
