function [ ns, lnd ] = lyapunov( x, d, m, N, r, n, w )
% This function calculates metrics for the largest short-time Lyapunov
% exponent from a scalar time series after reconstructing its phase space
% using delay coordinates
%
% CALL: [ ns, lnd ] = lyapunov( x, d, m, N, r, n, w );
%   If no output is requested it will just plot the results
%
% INPUT:
%   x -- scalar time series
%   d -- embedding dimension
%   m -- delay parameter
%   N -- number of initial (fiducial) points to use (should be large)
%   r -- number of nearest neighbors to use (r = 1 should work). For
%        r > 1 the neighbor trajectories are averaged with the weight
%        w(delta) = a^4/(delta^4 + a^4), Eq. (loclinw) of Chapter 15,
%        delta the distance from the fiducial point to the neighbor and
%        a the distance to its r-th (farthest) neighbor.
%   n -- maximum number of steps ahead to proceed
%   w -- (optional) supply 0 to not remove temporal correlations
%        or number of points to remove. The default is p=d*m.
%        NOTE: only need to remove these correlations for flow data!!
%
% OUTPUT:
%   ns -- the time steps used
%   lnd -- ln of averaged distances
%   If not output is requested it plots the results. rescale results using 
%   appropriate sampling time interval to get actual short-time maximal LE.
%
% (c) by David Chelidze 11/04/2005
%     modified by DC on 11/16/2005, 4/11/2011, 11/26/2012, 11/11/2013,
%     11/2/17, 12/11/17, 1/5/18

if nargin < 6
    error('Error: Needs at least six input variables!')
end
if min(size(x)) ~= 1 % check data and make a raw vector of scalar data
   error('Error: First input should be a vector of scalar data points.')
else % make row vector with zero mean and unit variance
    x = (x(:)' - mean(x))/std(x);
end
nm = length(x) - (d-1)*m; % max number of points possible in phase space
if nargin < 7 || isempty(w)
    w = d*m; % remove temporal correlations of this length, open for debate
end
if N == 0 || isempty(N) % set the default
    N = min(2^12, nm);
else % use the provided or max possible points
    N = min(N, nm);
end
% reconstruct the phase space
y = zeros(d, nm); 
for i = 0:d-1
    y(i+1,:) = x( (1:nm) + i*m );
end

% define N random initial conditions for `fiducial' points
indx = randperm(nm-n,N);
lnd = zeros(1,n); % initialize log of average distance array

% partition all suitable phase space points for fast searching
ns = createns(y(:,1:end-n)','NSMethod','kdtree','Distance','euclidean'); 
% search for the k points nearest to the test points (+ 2*w if removing
% temporal correlations)
np = r + 2*w + 1; % total nearest neighbors to look up
[idx, dis] = knnsearch(ns,y(:,indx)','Distance','euclidean','k',np);
if w == 0 % do not remove temporarily correlated nearest neighbor distances
    dis = dis(:,2:end); % only drop the first zero distance
    idx = idx(:,2:end); 
else % remove temporal correlations
    [dis, idx] = cellfun(@removecorrs, mat2cell(dis,ones(N,1),np),...
                 mat2cell(idx,ones(N,1),np),'UniformOutput',false);
    dis = cell2mat(dis);
    idx = cell2mat(idx);
end

for k = 1:N % for each fiducial trajectory
    a = dis(k,end); % scale: the r-th neighbor distance
    if a > 0
        W = a^4 ./ ( dis(k,:).^4 + a^4 ); % weighting function
    else
        W = ones(1,r);
    end
    dist = zeros(1, n); % initialize local distances
    for inn = 1:r % all nearest neighbor trajectories
        % find the difference between nn and fiducial trajectories
        tmp = y(:, idx(k,inn)+(1:n)) - y(:, indx(k)+(1:n)); 
        % normalized distances between trajectories:
        dist = dist + W(inn)*sqrt(sum(tmp.*tmp)); 
    end
    lnd = lnd + log(dist/sum(W)); % log of average distance
end % k loop
lnd = lnd/N; % normalize by the number of reference/fiducial points
ns = 1:n; % just how far we went

if nargout == 0 % plot the results
    figure
    set(gcf,'PaperSize',[10.5 3.5],'PaperPosition', [-.75 -.25 12 4]);
    subplot(121)
    set(gca,'fontsize',12)
    plot(ns,lnd,'linewidth',1), axis tight
    grid on, pbaspect([1.618, 1, 1])
    xlabel('$n$','Interpreter','Latex','FontSize',14)
    ylabel('$\Delta_n = \langle \ln \delta_n - \ln \delta_0 \rangle$',...
        'Interpreter','Latex','FontSize',14)   
    subplot(122)
    set(gca,'fontsize',12)
    plot(diff(ns).\diff(lnd),'linewidth',1), axis tight
    grid on, pbaspect([1.618, 1, 1])
    xlabel('$n$','Interpreter','Latex','FontSize',14)
    ylabel('$\Delta t \hat\lambda_1=\mathrm{d}\Delta_n/\mathrm{d}n$',...
        'Interpreter','Latex','FontSize',14)   
end
%% this removes temporally correlated nearest neighbors
function [d, id] = removecorrs(d, id)
    cond =  abs(id - id(1)) > w;
    d = d(cond);
    id = id(cond);
    d = d(1:r);
    id = id(1:r);
end
end
