function [ xf ] = sodfilt( x, l, m, d, q, w, r, tw )
% SODFILTER: SOD based noise reduction in nonlinear time series. 
% CALL:
%   [ xf ] = sodfilt( x, l, m, d, q, w );
% REQUIRED INPUT:
%   x -- scalar time series (data) in a column or row form 
%   l -- number of points in each strand (use odd number, or it will add 1)
%   m -- delay time for the phase space reconstruction (see ami.m/cds.m)
%   d -- embedding dimension (see fnns.m) (d < l!)
%   q -- local embedding dimension (q < d!!)
%  OPTIONAL INPUT PARAMETERS:
%   w -- weighting used in filtering (0--comb def, 1--tent, 2--exponential)
%   r -- strands per bundle (default 1, i.e. no bundle). With r > 1 the
%        r-1 nearest temporally uncorrelated strands join the base one,
%        each is decomposed separately, and the smoothed strands are
%        averaged with weights w(delta) = a^4/(delta^4 + a^4), Eq.
%        (loclinw) of Chapter 15, delta the distance from the base point
%        to a strand centre and a the distance to the farthest (r-th)
%        strand centre of the bundle.
%        Costs about r times as much and pays only on noisy records:
%        r = 4 brought the correlation dimension within 3% of the clean
%        value at 20% noise (r = 8 at 40%) where r = 1 left it 44% high
%        (see the table in Sec. 13.2).
%   tw -- Theiler window excluding temporally correlated strands from the
%        bundle (default (d-1)*m + 2*l + 1, one strand span)
% OUTPUT:
%   xf -- filtered time series
%
% This function uses smooth orthogonal decomposition to filter noisy
% nonlinear time series. The input time series x is used to reconstruct
% phase space via delay coordinate embedding (generating a multivariate
% time series). To avoid edge effects, the reconstruction is continued on 
% both ends by finding the nearest neighbor to the end points in the 
% reconstruction and following their trajectories. An l-length segment of
% this vector-valued time series is called a strand. Each strand is
% decomposed using SOD and only the first q smooth modes are used for noise
% reduction. The filtering is done considering all the corrections to each
% individual sample.
%
% copyright and written by David Chelidze 03/16/2012 04/06/2012 11/16/17
% contact email: chelidze@uri.edu
 
%% input check and initialization
if nargin < 5
   error('Requires at least five input arguments!\n')   
elseif min(size(x)) ~= 1
   error('Error: data should be a vector of scalar data points.\n')
else % normalize the input time series
    x = x(:)';
    nx = m*(d-1); % shrinkage in reconstruction
    n = length(x)- nx; % <- the maximum size possible! 
    if nargin < 6 || isempty(w)
        w = 0;
    end
    if nargin < 7 || isempty(r)
        r = 1; % no bundle
    end
end
y = zeros(d, n); % initialize array for embedded vectors
%% check if input variables are feasible
if l <= d || isempty(l)
    error('make string length l larger than the embedding dimension d\n')
end
if d <= q
    error('make subspace dimension q smaller than embedding dimension d\n')
end
l = floor(l/2); % distance from first point to mid-point
sl = 2*l + 1; % strand length
if nargin < 8 || isempty(tw)
    tw = nx + sl; % one strand span, so no neighbor overlaps the base
end
%% embedding and partitioning for nearest neighbor (nn) search
for i = 1:d % reconstruct the phase space
    y(i,:) = x((1:n) + (i-1)*m);
end % embedding
tree = createns(y','NSMethod','kdtree','Distance','euclidean'); 
%% padding (extending edges for filtering and shrinkage mitigation)
np = l + nx; % padding required to deal with edge effects
[pqi, ~] = knnsearch(tree,y(:,1)','Distance','euclidean','k',np);
pqi = pqi( pqi >  np ); % find suitable nearest neighbor
ya = y(:,pqi(1)-np:pqi(1)-1); % pad the left side
[pqi, ~] = knnsearch(tree,y(:,end)','Distance','euclidean','k',np);
pqi = pqi( pqi <  n - np ); % find suitable neighbor
yb = y(:,pqi(1)+1:pqi(1)+np); % pad the right side
y = [ ya y yb ]; % padded phase space
npr = n+2*np; % new number of padded reconstructed points
%% bundle setup (only needed for r > 1)
if r > 1
    cen = (l+1):(npr-l); % strand centers available in the padded data
    btree = createns(y(:,cen)','NSMethod','kdtree',...
                     'Distance','euclidean');
    nq = min(numel(cen), r + 4*tw); % enough to survive the Theiler cut
end
%% weighting function for filtering
if w == 1 % use tent weighting function W (good results)
    wd = [0:l, l-1:-1:0];
elseif w == 2 % use exponential weighting function W (try if you want)
    wd = exp(-([l:-1:0, 1:l]));
else % use comb weighting function W (no averaging)
    wd = [zeros(1,l) 1 zeros(1,l)];
end
wd = wd/sum(wd); % normalize (not required: see the division below)
W = diag(wd); % weighting applied along the strand
%% SOD filtering part
Yh = zeros(d, npr); % stores adjusted trajectory points
wa = zeros(1, npr); % stores the weight accumulated at each point
for k = 1:npr-sl+1 % each reconstructed point
    if r == 1 % the base strand on its own
        cn = k + l; % its center
        Wn = 1;
    else % gather the bundle: base strand plus r-1 nearest neighbors
        [ci, dis] = knnsearch(btree, y(:,k+l)', 'k', nq);
        ci = cen(ci); % back to indices in the padded data
        ok = abs(ci - (k+l)) > tw; % temporally uncorrelated only
        ci = ci(ok); dis = dis(ok);
        nb = min(r-1, numel(ci));
        cn = [k+l, ci(1:nb)];
        dis = [0, dis(1:nb)];
        ok = (cn-l >= 1) & (cn+l <= npr); % full strands only
        cn = cn(ok); dis = dis(ok);
        a = dis(end); % scale: the farthest strand of the bundle
        if a > 0
            Wn = a^4 ./ ( dis.^4 + a^4 ); % falls off with distance
        else
            Wn = ones(size(dis));
        end
    end
    Ya = zeros(d, sl); % accumulates the smoothed strands of the bundle
    for j = 1:numel(cn)
        Y = y(:, cn(j)-l : cn(j)+l); % strand j of the bundle
        b = mean(Y, 2); % its bias vector
        Y = Y - b*ones(1,sl); % unbiased
        [U,~,X,C,~] = gsvd( Y', diff(Y'), 0 ); % SOD of this strand
        X = fliplr(X); % SOMs
        U = fliplr(U); % SOCs
        C = flipud(diag(C)); % SOVs
        % project onto the smooth subspace, add back the bias, weight it
        Ya = Ya + Wn(j)*( X(:,1:q)*diag(C(1:q))*U(:,1:q)' ...
                          + b*ones(1,sl) );
    end
    Ya = Ya/sum(Wn); % the weighted average over the bundle
    Yh(:, k:k+sl-1) = Yh(:, k:k+sl-1) + Ya*W;
    wa(k:k+sl-1) = wa(k:k+sl-1) + wd; % weight received by these points
end
%% final averaging to get filtered time series
% trim the padded edges and normalize each point by the weight it got:
% interior points are covered by all (2l+1) strands, the first and last l
% by fewer, so without this they would come out attenuated
Yh = Yh(:,l+1:npr-l)./( ones(d,1)*wa(l+1:npr-l) );
%% shifting coordinates so they align in columns
xf = zeros(d,n+nx);
for k = 1:d
    xf(k,:) = Yh(k,m*(d-k)+1:end-m*(k-1));
end
xf = mean(xf, 1); % average for filtered time series
