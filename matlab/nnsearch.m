function [Jk, Dk] = nnsearch(Y, w, k, metric)
% NNSEARCH k nearest neighbors of every row of Y outside a Theiler window.
%
% CALL: [J, D] = nnsearch(Y, w, k, metric);
%
% INPUT:
%   Y      -- n x d matrix of reconstructed points (rows)
%   w      -- Theiler window in samples: neighbors j of i need |i-j| > w
%   k      -- number of neighbors
%   metric -- 'euclidean' or 'chebychev' (maximum norm), as in knnsearch
%
% OUTPUT:
%   J, D   -- n x k indices and distances, nearest first
%
% copyright by David Chelidze 2026
n = size(Y, 1); K = min(n, k + 2*w + 1);
Jk = zeros(n, k); Dk = zeros(n, k); todo = (1:n)';
[J, D] = knnsearch(Y, Y, 'K', K, 'Distance', metric);
while ~isempty(todo)
    ok = abs(J - todo) > w; cnt = cumsum(ok, 2);
    done = cnt(:, end) >= k;
    for r = find(done)'
        sel = ok(r, :) & cnt(r, :) <= k;
        Jk(todo(r), :) = J(r, sel); Dk(todo(r), :) = D(r, sel);
    end
    todo = todo(~done);
    if ~isempty(todo)                          % too many close-in-time points
        K = min(n, 2*K);
        [J, D] = knnsearch(Y, Y(todo, :), 'K', K, 'Distance', metric);
    end
end
end