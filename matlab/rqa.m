function m = rqa(X, eps, w, lmin, vmin)
% RQA  Recurrence matrix and recurrence quantification measures (Chapter 18).
%
%   m = rqa(X, eps, w, lmin, vmin)
%
%   X     (M x d) delay vectors, one per row (Euclidean distances)
%   eps   recurrence threshold (same units as X)
%   w     Theiler window: pairs with |i-j| <= w are excluded (default 0)
%   lmin  minimum diagonal line length counted (default 2)
%   vmin  minimum vertical line length counted (default 2)
%
%   m     struct with fields R (logical M x M), RR, DET, L, Lmax, ENTR,
%         LAM, TT, P (diagonal line histogram, P(l+1) = count of length l),
%         Pv (vertical line histogram), eps
%
%   The definitions follow Marwan et al. (2007): RR is the fraction of
%   admissible pairs that recur, DET the fraction of recurrent points on
%   diagonal lines of length >= lmin, L and Lmax the mean and maximum
%   diagonal line length, ENTR the Shannon entropy of the diagonal
%   line-length distribution, LAM and TT the vertical-line analogues.
%   To set eps from a target recurrence rate rr, take the rr-quantile of
%   the admissible pairwise distances:
%       D = squareform(pdist(X)); A = triu(true(size(D)), w+1);
%       eps = quantile(D(A), rr);
%   Memory is O(M^2).
if nargin < 3 || isempty(w),    w = 0;    end
if nargin < 4 || isempty(lmin), lmin = 2; end
if nargin < 5 || isempty(vmin), vmin = 2; end
X = double(X); if size(X,1) < size(X,2), X = X.'; end
M = size(X,1);
D = sqrt(max(sum(X.^2,2) + sum(X.^2,2).' - 2*(X*X.'), 0));
R = D < eps;
[I, J] = ndgrid(1:M, 1:M);
R(abs(I - J) <= w) = false;
% diagonal line histogram (upper triangle, doubled by symmetry)
P = zeros(M+1, 1);
for k = w+1:M-1
    L = runs(diag(R, k));
    for q = 1:numel(L), P(L(q)+1) = P(L(q)+1) + 1; end
end
P = 2*P;
Pv = zeros(M+1, 1);
for j = 1:M
    L = runs(R(:, j));
    for q = 1:numel(L), Pv(L(q)+1) = Pv(L(q)+1) + 1; end
end
ell = (0:M).';
nadm = (M-w)*(M-w-1);
m.R = R; m.eps = eps;
m.RR = nnz(R)/nadm;
tot = sum(ell.*P);
sel = ell >= lmin;
m.DET = sum(ell(sel).*P(sel))/tot;
nl = sum(P(sel));
m.L = sum(ell(sel).*P(sel))/nl;
m.Lmax = max(ell(P > 0));
p = P(sel)/nl; p = p(p > 0);
m.ENTR = -sum(p.*log(p));
totv = sum(ell.*Pv); selv = ell >= vmin;
m.LAM = sum(ell(selv).*Pv(selv))/totv;
m.TT = sum(ell(selv).*Pv(selv))/sum(Pv(selv));
m.P = P; m.Pv = Pv;
end

function L = runs(v)
% lengths of runs of true in a logical vector
v = [false; v(:); false];
d = diff(double(v));
L = find(d == -1) - find(d == 1);
end
