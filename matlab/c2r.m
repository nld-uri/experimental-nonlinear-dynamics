function [r, c] = c2r( x, d, m, n, r, p, q )
% C2R function calculates the correlation sum from a scalar time
%  series after reconstructing its phase space using delay coordinates
%  It uses Euclidean distances!
%
% CALL: [r, c] = c2r( x, d, m, n, r, p, q );
%       If no output is requested it will just plot the results
%
% INPUT:
%   x -- scalar time series
%   d -- embedding dimension
%   m -- delay time
%   n -- number of points to use in the reconstructed phase space
%   r -- number of bins (ball length-scales) to use if it's just a number, 
%        or the array of ball sizes (bins) to consider if it's a vector
%   p -- (optional) supply 0 to NOT remove temporal correlations
%        or number of points to remove. The default is p = d*m.
%        NOTE: only need to remove these correlations for the continuous 
%        flow data!
%   q -- (optional) only supply if instead of removing temporal
%        correlations you wish to randomly populate attractor with q sample
%        points (thus, this will ignore p input if provided, & use max
%        number of points possible in the phase space reconstruction)
%
% OUTPUT:
%   r -- ln of ball sizes
%   c -- ln of correlation sum
%
% copyright by David Chelidze 3/27/2002
% revised by DC 5/31/2002 & 6/12/2002 & 10/2/2002 & 10/30/2002
% & 11/04/2005 & 3/3/3009 & 11/4-21/2013 & 12/12/2013 & 10/20/17 & 12/12/17
 
if nargin < 5 % not enough input variables
    error('Error: Needs at least five input variables.')
end
if min(size(x)) ~= 1
   error('Error: The first input should be a vector of scalars.')
else % normalize the input time series
    x = (x(:)' - mean(x))/std(x); % make it zero mean and unit variance
end
nm = length(x) - (d-1)*m; % maximum number of reconstructed points possible
if isempty(n) % use max
    n = nm; 
elseif n > nm % use max possible and notify user too
    fprintf(['WARNING: Max number of embedded vectors exceeded.\n'...
        'ACTION:  Continuing with max number of vectors possible.\n'])
    n = nm;
end
if nargin < 6 || isempty(p) % remove temporal correlations of this length
    p = d*m; % still open for debate, may want to use different length
end
 
%% reconstruct the phase space to analyze interpoint distance distributions
if nargin == 7 && isempty(q) < 1 % randomly populate attractor w/ points 
    indx = randperm(n, q); % pick q random points from the reconstruction
    n = q; % new number of phase space points to use
    p = 0; % do not need to remove temporal correlations 
else % use the whole phase space
    indx = 1:n;
end
y = zeros(d,n); % initialize
for i = 0:d-1 % reconstructed phase space
    y(i+1,:) = x(indx+i*m);
end
 
%% calculate the correlation sum
np = n - p;
if length(r) == 1 % determine the needed length scales or bins to use
    c = zeros(1,r);
else % use the provided length scale or bins
    r = r.^2;
    c = zeros(size(r));
end
for i = 1:np-1 % for every point in the reconstructed phase space
    indx = i+p+1:n; % index of points to calculate distance to
    dist = 0; % initialize distances
    for k = 1:d % calculate distances squared to all other points
        dist = dist + (y(k, indx) - y(k, i)).^2;
    end
    if i == 1 && length(r) == 1 % find the suitable bins or length scales
        r = exp( linspace( ...
            log(min(dist)/np),log(max(dist) + min(dist) ), r) );
    end
    c = c + cumsum( histcounts(dist, [0, r]) ); % correlation sum
end
c = log(2*c/np/(np-1)); % ln of correlation sum
r = log(r)/2; % ln of the corresponding ball sizes
 
if nargout == 0 % plot the results
    figure
    set(gcf,'PaperSize',[10.5 3.5],'PaperPosition', [-.75 -.25 12 4]);
    subplot(121), plot(r,c,'-','linewidth',1)
    set(gca,'fontsize',12)
    grid on, pbaspect([1.618, 1, 1])
    xlabel('$\ln(\epsilon)$','Interpreter','Latex','FontSize',14)
    ylabel('$\ln \hat C(\epsilon)$','Interpreter','Latex','FontSize',14)
    subplot(122), plot(r(1:end-1),diff(c)./diff(r),'.-','linewidth',1)
    set(gca,'fontsize',12)
    grid on, pbaspect([1.618, 1, 1])
    xlabel('$\ln(\epsilon)$','Interpreter','Latex','FontSize',14)
    ylabel(['$D_2 \approx \frac{\mathrm{d}\ln \hat C(\epsilon)}',...
        '{\mathrm{d}\ln(\epsilon)}$'],'Interpreter','Latex','FontSize',14)
end
