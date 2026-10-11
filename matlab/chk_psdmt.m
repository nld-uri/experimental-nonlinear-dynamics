% CHK_PSDMT  Checks of psdmt.m and mtlines.m with MATLAB's own dpss/finv.
% Run from matlab/:  >> chk_psdmt
% Reference values from python/psdmt.py are in ../reports/mt_test/.
%
R = fullfile('..', 'reports', 'mt_test');
x = load(fullfile(R, 'x.txt')); fs = 50;

% 1. tapers: concentrations against SciPy
[~, lam] = dpss(1000, 4, 7); lr = load(fullfile(R, 'lam_1000_4_7.txt'));
fprintf('1. 1-lambda MATLAB: %s\n', sprintf('%.2e ', 1 - lam));
fprintf('   1-lambda SciPy : %s\n', sprintf('%.2e ', 1 - lr));

% 2. same record, MATLAB tapers, against the Python outputs
[f, G, S] = psdmt(x, fs, 4, 7, true); P = load(fullfile(R, 'ref_ad.txt'));
fprintf('2. adaptive: max rel diff G %.1e, F %.1e, |mu| %.1e; max |nu diff| %.1e\n', ...
    max(abs(G - P(:,2))./P(:,2)), max(abs(S.F - P(:,3))./P(:,3)), ...
    max(abs(abs(S.mu) - P(:,4))./P(:,4)), max(abs(S.nu - P(:,5))));
[~, G, S] = psdmt(x, fs, 3, 5, false); P = load(fullfile(R, 'ref_av.txt'));
fprintf('   average : max rel diff G %.1e, F %.1e\n', ...
    max(abs(G - P(:,2))./P(:,2)), max(abs(S.F - P(:,3))./P(:,3)));
[xc, L] = mtlines(x, fs); PL = load(fullfile(R, 'ref_lines.txt'));
fprintf('   mtlines: %d lines (Python %d); max diff in [f A phase F] %.1e; xc %.1e\n', ...
    size(L,1), size(PL,1), max(abs(L(:) - PL(:))), max(abs(xc - load(fullfile(R, 'ref_xc.txt')))));

% 3. white noise: mean, dof, Parseval, F null distribution
rng(1); N = 1024; M = 2000; K = 7; Gw = zeros(N/2+1, M); Fw = Gw; pv = zeros(M,1);
for r = 1:M
    w = randn(N, 1); [f, g, s] = psdmt(w, 1, 4, K); Gw(:,r) = g; Fw(:,r) = s.F;
    pv(r) = sum(g)*f(2)/var(w, 1);
end
in = 21:N/2-19; m = mean(Gw(in,:), 'all'); v = var(Gw(in,:), 1, 'all');
fprintf('3. white noise: mean/2 = %.3f, nu = %.1f (2K = %d), Parseval %.4f, P(F > F99) = %.4f\n', ...
    m/2, 2*m^2/v, 2*K, mean(pv), mean(Fw(in,:) > finv(0.99, 2, 2*K-2), 'all'));

% 4. two-well record: harmonics against the Python values of chk_multitaper.out
xt = load(fullfile('..', 'data', 'twowell_2002-06-02.csv')); xt = xt - mean(xt);
[f, ~, S] = psdmt(xt, 160, 4, 7);
fprintf('4. two-well: ');
for h = 1:4, [~, j] = min(abs(f - 10*h)); fprintf('%d Hz F=%.1f |mu|=%.4e;  ', 10*h, S.F(j), abs(S.mu(j))); end
fprintf('\n');
