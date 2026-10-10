function [xc, L] = mtlines(x, fs, NW, K, p, pad)
% MTLINES Detect lines by the harmonic F-test and remove them (Chapter 6).
%
% USE: >> [xc, L] = mtlines(x, fs);                 % NW = 4, K = 7, p = 1/N
%      >> [f, Gc] = psdmt(xc, fs);                  % continuous part
%
% A line is a local maximum of the F statistic above the 1 - p point of
% F(2, 2K-2), on a grid zero-padded by pad (default 8). Each detected
% sinusoid is subtracted from the record. With p = 1/N and pad = 8 the
% measured false-line rate for white noise (N = 1024) is 1.8 per record.
%
% OUTPUT: xc -- record with the lines removed (mean removed)
%         L  -- one row per line: [f0 (Hz), amplitude A, phase (rad), F]
%
% Needs finv (Statistics and Machine Learning Toolbox).
% Written by David Chelidze for the second edition.

x = x(:) - mean(x); N = numel(x);
if nargin < 3 || isempty(NW), NW = 4; end
if nargin < 4 || isempty(K), K = floor(2*NW) - 1; end
if nargin < 5 || isempty(p), p = 1/N; end
if nargin < 6 || isempty(pad), pad = 8; end
[f, ~, S] = psdmt(x, fs, NW, K, false, pad*N);
F = S.F;
thr = finv(1 - p, 2, 2*K - 2);
pk = F > thr & F >= [0; F(1:end-1)] & F >= [F(2:end); 0];
pk([1 end]) = false;                    % no lines at 0 or Nyquist
idx = find(pk); t = (0:N-1).'/fs;
xc = x; L = zeros(numel(idx), 4);
for i = 1:numel(idx)
    j = idx(i); mu = S.mu(j);
    xc = xc - 2*real(mu*exp(2i*pi*f(j)*t));
    L(i, :) = [f(j), 2*abs(mu), angle(mu), F(j)];
end
end
