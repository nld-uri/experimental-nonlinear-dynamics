function f = fnnfrac(eps, dlt, r, s, a)
% FNNFRAC the false nearest neighbor fractions from the output of FNN.
%
% CALL: f = fnnfrac(eps, dlt, r, s, a);
%
% INPUT:
%   eps, dlt -- cell arrays from FNN (data dlt, or dsur for surrogates)
%   r, s, a  -- ratio threshold, size threshold in units of sigma_x, and
%               Kennel's A_tol (defaults 10, 0.5, 2)
%
% OUTPUT: structure of row vectors against d = 1:dmax
%   f.orig   -- original fraction f_nn(d;r), Eq. (fnnFrac)
%   f.kennel -- Kennel fraction with both criteria, Eq. (OrKennel)
%   f.tisean -- TISEAN fraction, Eq. (tisean) (zero if no admissible pair)
%   f.rel    -- reliable fraction, Eq. (K_Rel)
%   f.comp   -- composite fraction R_nn(d;r,s), Eq. (new)
%
% copyright by David Chelidze 2026
if nargin < 5, a = 2; end
if nargin < 4, s = 0.5; end
if nargin < 3, r = 10; end
dmax = numel(eps);
f = struct('orig', zeros(1, dmax), 'kennel', zeros(1, dmax), ...
           'tisean', zeros(1, dmax), 'rel', zeros(1, dmax), 'comp', zeros(1, dmax));
for d = 1:dmax
    e = eps{d}; dl = dlt{d};
    ratio = dl > r*e; adm = r*e < 1;
    f.orig(d) = mean(ratio);
    f.kennel(d) = mean(ratio | (dl.^2 + e.^2 > a^2));
    f.tisean(d) = sum(ratio & adm)/max(sum(adm), 1);
    f.rel(d) = mean(dl > s);
    f.comp(d) = mean(ratio | (dl > s));
end
end
