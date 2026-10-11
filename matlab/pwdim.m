 function [lnD, lnN] = pwdim( x, m, d, k, N, w )
% PWDIM: estimates average pointwise dimension using fixed distance and
%        fixed mass methods
% NOTE: York claims it is equivalent to information dimension, 
%       Moon claims it is like correlation dimension
%
%   CALL [lnD, lnN] = pwdim( x, m, d, k, N, w );
% 
% INPUT:
%   x - scalar time series
%   m - delay parameter (time delay for embedding)
%   d - embedding dimensions to use (it can be an array!)
%   k - number of distance partitions (for fixed distance) or points inside 
%       a ball (for fixed mass)
%   N - maximum number of averages to do (optional -- 2^12 is default)
%   w - (optional) Theiler window to remove the correlations, by default
%       not temporal correlations are removed. This does not seem to alter
%       the results significantly, but considerably increase the
%       computational time.
%
% OUTPUT: The first two columns are fixed distance and the second are fixed 
%         mass. If no output is requested, then it plots the results.
%   lnN(:,1)  - <ln(fraction of points inside the eps-ball)>
%   lnD(:,1)  - ln(eps)--i.e., log of the epsilon-ball
%   lnN(:,2)  - ln(n/N)--i.e., faction of points or mass
%   lnD(:,2)  - <ln(ball size containing n points)>
%
%     copyright by David Chelidze, 1998-2013
%     modified 10/26/17 by DC
 
%% Initialization and preprocessing
if nargin < 3
   error('Requires at least three input arguments!')   
else
    x = x(:)'; % put 'em in correct form and get parameters 
    n = length(x)- m*(d-1); % <- that's the maximum size possible!!!; 
    if nargin < 6 % do not remove temporal correlations
        w = 0;
    end
    if nargin < 5 || N == 0 || isempty(N) % set the default
        N = min(2^12, n);
    else % use the provided or max possible points
        N = min(N, n);
    end
    if nargin < 4 || k == 0 || isempty(k) % use the default # of NNs
        k = 500;
    end
end
lnN = zeros(N,k); % initialize the log of measure array
y = zeros(d, n); % initialize array for embedded vectors
for i = 1:d % reconstruct the phase space
    y(i,:) = x((1:n) + (i-1)*m);
end % embedding
 
%% create the kd-partition search for nearest distances
ns = createns(y','NSMethod','kdtree','Distance','euclidean'); 
% search for the k points nearest to the test points (+ 2*w if removing
% temporal correlations)
np = k + 1 + 2*w; % total nearest neighbors to look up
[idx, dis] = knnsearch(ns,y(:,randperm(n,N))',...
    'Distance','euclidean','k',np);
if w == 0 % do not remove temporarily correlated nearest neighbor distances
    dis = dis(:,2:end); % only drop the first zero distance
else % remove temporal correlations
    dis = cell2mat( cellfun( @removecorrs,mat2cell(dis,ones(N,1),np), ...
          mat2cell(idx,ones(N,1),np),'UniformOutput',false ) );
end
 
%% the fixed mass based estimates:
lnDm = mean(log(dis)); % average log of the eps-ball
% lnNm = log(1:k) - log(N); % measure inside the eps-ball or use:
lnNm = psi(1:k) - log(N); % Grassberger (1985) measure inside the ball
 
%% the fixed distance based estimates
dm = max(dis(:)); % maximum distance
edgs = [0 exp(linspace(lnDm(1), mean([lnDm(end) log(dm)]), k))];
for ii = 1:N
    ln = log( cumsum(histcounts(dis(ii,:), edgs)) );
    ln( ln==-Inf ) = 0; % if count is zero assign zero
    lnN(ii,:) = ln;
end
lnN = [sum(lnN)./sum(lnN~=0) - log(N); lnNm]';
lnD = [log(edgs(2:end)); lnDm]';
 
%% plotting the results
if nargout == 0 
    figure
    set(gcf,'PaperSize',[10.5 3.5],'PaperPosition', [-.75 -.25 12 4]);
    subplot(121) 
    set(gca,'fontsize',12)
    plot(lnD,lnN,'linewidth',1), axis tight
    ylabel('$\langle \ln \hat \mu(\epsilon)\rangle,\; \ln \left(n/N\right)$',...
        'Interpreter','Latex','FontSize',14)
    xlabel('$\ln(\epsilon),\; \langle \ln(\epsilon) \rangle$','Interpreter',...
        'Latex','FontSize',14)
    legend('fixed distance', 'fixed mass','location','southeast')
    grid on, pbaspect([1.618, 1, 1])
    subplot(122)
    plot(lnD(1:end-1,:),diff(lnD).\diff(lnN),'linewidth',1)
    set(gca,'fontsize',12)
    grid on, pbaspect([1.618, 1, 1])
    xlabel('$\ln(\epsilon),\; \langle \ln(\epsilon) \rangle$','Interpreter',...
        'Latex','FontSize',14)
    ylabel('$\hat D_\mathrm{P}, \bar D_\mathrm{P}$','Interpreter','Latex','FontSize',14)
    legend('fixed distance', 'fixed mass','location','south')
    axis([min(lnD(1,2)) max(lnD(end,2)) 0 ...
        ceil(max(max(diff(lnD).\diff(lnN))))])
end

%% this removes distances to temporally correlated nearest neighbors
function d = removecorrs(d, id)
    d = d( abs(id - id(1)) > w );
    d = d(1:k);
end

end
