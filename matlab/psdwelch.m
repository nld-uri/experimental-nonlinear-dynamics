function [f, S] = psdwelch(x, fs, L, D, win, y)
% PSDWELCH Welch spectral estimates from one or two records (Chapter 6).
%
% USE: >> [f, Gxx] = psdwelch(x, fs, L, D, win);      % one-sided auto-spectrum
%      >> [f, S]   = psdwelch(x, fs, L, D, win, y);   % struct: Gxx Gyy Gxy coh H1 H2
%
% INPUT:
%   x, y -- records (vectors); x is the input when an FRF is wanted
%   fs   -- sampling frequency (Hz)
%   L    -- segment length (samples);  D -- overlap (samples), default L/2
%   win  -- window: a vector of length L, or a name ('hann', 'rectwin', ...)
%
% OUTPUT: one-sided densities on f >= 0, units x^2/Hz, so sum(Gxx)*f(2) ~ var(x).
%   Gxy = one-sided cross-spectrum, coh = |Gxy|^2/(Gxx Gyy),
%   H1 = Gxy./Gxx,  H2 = Gyy./conj(Gxy).
%
% Written by David Chelidze.

if nargin < 4 || isempty(D), D = floor(L/2); end
if nargin < 5 || isempty(win), win = 'hann'; end
if ischar(win) || isstring(win), w = window(str2func(char(win)), L, 'periodic'); ...
else, w = win(:); end
U = sum(w.^2);                          % window power: E[G] = S for white noise
nf = floor(L/2) + 1;
f = (0:nf-1)'*fs/L;
scale = 2/(fs*U)*ones(nf, 1); scale(1) = scale(1)/2;
if mod(L, 2) == 0, scale(end) = scale(end)/2; end   % Nyquist bin not doubled
X = segfft(x, L, D, w, nf);
Gxx = scale.*mean(abs(X).^2, 2);
if nargin < 6, S = Gxx; return; end
Y = segfft(y, L, D, w, nf);
S.Gxx = Gxx;
S.Gyy = scale.*mean(abs(Y).^2, 2);
S.Gxy = scale.*mean(conj(X).*Y, 2);
S.coh = abs(S.Gxy).^2./(S.Gxx.*S.Gyy);
S.H1 = S.Gxy./S.Gxx;
S.H2 = S.Gyy./conj(S.Gxy);
S.K = size(X, 2);
end

function X = segfft(x, L, D, w, nf)
% overlapped, mean-removed, tapered segments in columns, transformed
x = x(:); K = floor((numel(x) - L)/(L - D)) + 1;
idx = (1:L)' + (L - D)*(0:K-1);
seg = x(idx); seg = seg - mean(seg, 1);
X = fft(seg.*w, [], 1); X = X(1:nf, :);
end
