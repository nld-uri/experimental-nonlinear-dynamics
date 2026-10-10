function [E1, E2] = cao(x, tau, dmax, w)
% CAO Cao's statistics E1(d) and E2(d) for the minimum embedding dimension.
%
% CALL: [E1, E2] = cao(x, tau, dmax, w);
%
% INPUT:
%   x    -- scalar time series (normalized inside)
%   tau  -- delay
%   dmax -- largest embedding dimension
%   w    -- Theiler window in samples (0 for none)
%
% OUTPUT:
%   E1, E2 -- row vectors for d = 1:dmax, Eqs. (eq:ed_caoE1), (eq:ed_caoE2);
%             maximum norm, nearest neighbor outside the Theiler window
%
% copyright by David Chelidze 2026
x = x(:)'; x = (x - mean(x))/std(x); N = numel(x);
E = zeros(1, dmax + 1); Es = E;
for d = 1:dmax + 1
    n = N - d*tau;
    Y = zeros(n, d);
    for q = 1:d, Y(:, q) = x((1:n) + (q-1)*tau); end
    [J, D] = nnsearch(Y, w, 1, 'chebychev');
    i = (1:n)'; j = J(:, 1); e = D(:, 1);
    dl = abs(x(i + d*tau) - x(j + d*tau))';
    ok = e > 0;
    E(d) = mean(max(e(ok), dl(ok))./e(ok));
    Es(d) = mean(dl);
end
E1 = E(2:end)./E(1:end-1);
E2 = Es(2:end)./Es(1:end-1);
end
