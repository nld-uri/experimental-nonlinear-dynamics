function [wp, zp, U, H] = mdofmodes(M, C, K, w)
% MDOFMODES: modal parameters and FRF matrix of a linear M-C-K system.
% CALL:
%   [wp, zp, U, H] = mdofmodes(M, C, K, w);
% INPUT:
%   M, C, K -- n x n mass, damping and stiffness matrices (C may be zero)
%   w       -- frequency grid in rad/s (any length nw), may be empty
% OUTPUT:
%   wp -- undamped natural frequencies, ascending (rad/s)
%   zp -- modal damping ratios
%   U  -- n x n mode matrix. If C is proportional (U'*C*U diagonal to
%         1e-8 relative), U holds the real mass-normalized modes,
%         U'*M*U = I. Otherwise U holds the complex modes, one per
%         conjugate pair, scaled so the largest entry of each is 1.
%   H  -- n x n x nw receptance matrix, H(:,:,k) = inv(K - w^2 M + i w C),
%         computed from the modal/residue expansion, not by inversion.
%
% Chapter 19. Written by David Chelidze; Python twin: mdofmodes.py
if nargin < 4, w = []; end
n = size(M, 1); w = w(:).'; nw = numel(w);
[U, L2] = eig(K, M, 'chol');                 % K U = M U Omega^2
[wp, ix] = sort(sqrt(diag(L2))); U = U(:, ix);
U = U ./ sqrt(diag(U'*M*U)).';               % mass normalize: U'*M*U = I
U = U .* sign(U(1, :) + (U(1, :) == 0));     % first entry positive
Cm = U'*C*U; offd = max(abs(Cm - diag(diag(Cm))), [], 'all');
H = zeros(n, n, nw);
if offd <= 1e-8*max(max(abs(Cm(:))), realmin)  % proportional damping
    zp = diag(Cm) ./ (2*wp);
    for p = 1:n
        den = wp(p)^2 - w.^2 + 2i*zp(p)*wp(p)*w;
        H = H + reshape(U(:, p)*U(:, p).', n, n, 1) ./ reshape(den, 1, 1, nw);
    end
    return
end
% general viscous damping: 2n first-order state space
A = [zeros(n) eye(n); -M\K -M\C];
[V, D] = eig(A); lam = diag(D);
for r = 1:2*n                                % residue form of the FRF
    phi = V(1:n, r); lr = lam(r);
    ar = phi.'*(2*lr*M + C)*phi;             % modal 'a' constant
    H = H + reshape(phi*phi.', n, n, 1) ./ reshape(ar*(1i*w - lr), 1, 1, nw);
end
keep = imag(lam) > 0;                        % one of each conjugate pair
lam = lam(keep); Phi = V(1:n, keep);
[~, ix] = sort(abs(lam)); lam = lam(ix); Phi = Phi(:, ix);
wp = abs(lam); zp = -real(lam) ./ abs(lam);
[~, j] = max(abs(Phi), [], 1);               % largest entry -> real, = 1
U = Phi ./ Phi(sub2ind(size(Phi), j, 1:n));
end
