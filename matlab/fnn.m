function [eps, dlt, dsur] = fnn(x, tau, dmax, w, k, l)
% FNN nearest-neighbor separations for the false nearest neighbor test
%  and the composite fraction of Chapter "Selecting Embedding Dimension".
%
% CALL: [eps, dlt, dsur] = fnn(x, tau, dmax, w, k, l);
%
% INPUT:
%   x    -- scalar time series (normalized inside, so sigma_x = 1)
%   tau  -- delay
%   dmax -- largest embedding dimension
%   w    -- Theiler window in samples (0 keeps all pairs), Eq. (rnns)
%   k, l -- (optional) candidate neighbors and strand half-length of the
%           composite-fraction selection, Eq. (CFrac); default 1, 0
%
% OUTPUT (cell arrays, entry d holds the N - d*tau pairs of dimension d):
%   eps  -- d-dimensional separation eps_i to the selected neighbor
%   dlt  -- appended separation delta_i = |x(i+d*tau) - x(j+d*tau)|
%   dsur -- surrogate delta_i, neighbor's appended coordinate drawn at
%           random from the record
% Use FNNFRAC to turn these into the fractions, and CAO for E1, E2.
%
% copyright by David Chelidze 2026
if nargin < 6, l = 0; end
if nargin < 5, k = 1; end
x = x(:)'; x = (x - mean(x))/std(x); N = numel(x);
eps = cell(dmax, 1); dlt = eps; dsur = eps;
for d = 1:dmax
    n = N - d*tau;
    Y = zeros(n, d);
    for q = 1:d, Y(:, q) = x((1:n) + (q-1)*tau); end
    [J, D] = nnsearch(Y, w, k, 'euclidean');
    i = (1:n)';
    if k > 1 && l > 0                          % strand selection, Eq. (CFrac)
        S = zeros(n, k);
        for q = -l:l
            iq = min(max(i + q, 1), n); jq = min(max(J + q, 1), n);
            for m = 1:k
                S(:, m) = S(:, m) + sqrt(sum((Y(jq(:, m), :) - Y(iq, :)).^2, 2));
            end
        end
        [~, m] = min(S./max(D, realmin), [], 2);
        lin = sub2ind([n, k], i, m); j = J(lin); e = D(lin);
    else
        j = J(:, 1); e = D(:, 1);
    end
    eps{d} = e;
    dlt{d} = abs(x(i + d*tau) - x(j + d*tau))';
    dsur{d} = abs(x(i + d*tau) - x(randi(N, n, 1)))';
end
end
