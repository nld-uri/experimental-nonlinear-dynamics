function [ xf ] = podfilt( x, l, m, d, q, f, u )
% PODFILT: Projective nonlinear noise reduction using SVD. 
% CALL:
%   [ xf ] = podfilt( x, l, m, d, q, f, u );
% REQUIRED INPUT:
%   x -- scalar time series (data) in a column or row form 
%   l -- number of points in a cloud of nearest neighbors or strand length
%   m -- delay time for the phase space reconstruction (see ami.m)
%   d -- embedding dimension (see fnns.m) (d < l)
%   q -- local embedding dimension (q < d)
% OPTIONAL INPUT:
%   f -- f = 1 (default) for cloud based or f = 0 strand based POD
%   u -- u = 1 (default) do not adjust for curvature, and u = 0 adjust 
% OUTPUT:
%   xf -- filtered time series in matrix form
%
% This function uses proper orthogonal decomposition to filter noisy
% nonlinear time series. The input time series is used to reconstruct phase
% space using delay coordinate embedding (generating a multivariate time
% series). To avoid edge effects, the reconstruction is continued on both
% ends by finding the nearest neighbor (nn) to the end points in the 
% reconstruction and following their trajectories. Point cloud based
% projective noise reduction is based on standard published procedure (see
% Kantz/Schreiber). For strand based projection, an l-length segment of
% this vector-valued time series is called a strand. Each strand is
% decomposed using SVD and only the first q dominant modes are used for
% noise reduction. The filtering is done considering all the corrections to
% each individual sample.
%
% copyright and written by David Chelidze 03/16/2012-04/06/2012, 11/16/17
% contact email: chelidze@uri.edu
 
%% input check and initialization
if nargin < 5
   error('Requires at least five input arguments!\n')   
elseif min(size(x)) ~= 1
   error('Error: data should be a vector of scalar data points.\n')
elseif d <= q
    error('make subspace dimension q smaller than embedding dimension d\n')
else % normalize the input time series
    x = x(:)';
    nx = m*(d-1); % shrinkage in reconstruction
    if nargin < 6 || isempty(f) % default: cloud based projection
        f = 1; 
    end
    if nargin < 7 || isempty(u) % default: do not adjust for curvature
        u = 1; % NOTE: cloud based projection with u = 0 diverges under
    end        % repeated application; the strand branch is fine with it
end
%% Embedding and Partitioning for the Nearest Neighbor Search
n = length(x); % original length of time series
nrp = n - nx; % number of reconstructed points
%% embedding and partitioning for nearest neighbor (nn) search
y = zeros(d, nrp); % initialize array for embedded vectors
for i = 1:d % reconstruct the phase space
    y(i,:) = x( (1:nrp) + (i-1)*m );
end % embedding
tree = createns(y','NSMethod','kdtree','Distance','euclidean'); 
R = eye(d,d); R(1,1) = 1000; R(d,d) = 1000; % weighting function
%% Padding (extending the reconstruction to deal with edge effects)
if f == 1 % Cloud Based Projection
    np = 1 + nx; % padding needed for the cloud based scheme
elseif l <= d
        error('Error: make string length (l) larger (l > d!)')
else
    l = floor(l/2) + 1; % strand middle point
    sl = 2*l - 1; % strand length
    np = l + nx; % padding needed for the strand based scheme
end    
[pqi, ~] = knnsearch(tree,y(:,1)','k',np); % find analogue to the first pnt
pqi = pqi( pqi >  np ); % choose the acceptable nns
ya = y(:, pqi(1) - np : pqi(1) - 1); % padding from the left
[pqi, ~] = knnsearch(tree,y(:,end)','k',np); % find analogue to the last p
pqi = pqi( pqi <  nrp - np ); % choose the acceptable nns
yb = y(:, pqi(1) + 1 : pqi(1) + np); % padding from the right
yp = [ ya y yb ]; % padded phase space
npr = nrp + 2*np; % new size of the phase space
%% Projecting the Noise Out
Yh = zeros(d, npr); % stores adjusted phase space points
if f == 1 % do nearest neighbor projection
    b = zeros(d, npr); % stores the bias
    [pqi, ~] = knnsearch(tree,yp','Distance','euclidean','k',l);
    for k = 1:npr % for each reconstructed point
        Y = y(:, pqi(k,:)); % the nearest neighbors
        b(:, k) = mean(Y, 2); % bias mean vector (crude mean filter)
        [U, S, V] = svd( ( Y - b(:,k)*ones(1,l) )'*R, 0 ); % POD
        C = V(:,1:q)*S(1:q,1:q)*U(:,1:q)'; % projection
        Yh(:,k) = R\C(:,1) + b(:,k); % adjustment
    end
    if u == 0
        Yh = Yh(:,2:npr-1) + b(:,2:end-1) - (b(:,1:end-2) + b(:,3:end))/2;
    else
        Yh = Yh(:,2:npr-1);
    end
else % do the strand projection
    b = zeros(d,npr-sl+1); % stores bias
    wd = [0:l-1, l-2:-1:0]; % use tent weighting for adjusting neighbors
    wd = wd/sum(wd); % normalize (not required: see the division below)
    W = diag(wd); % weighting applied along the strand
    wa = zeros(1, npr); % stores the weight accumulated at each point
    for k = 1:npr-sl+1 % for each reconstructed point
        Y = yp(:,k+(0:sl-1)); % get the corresponding strand
        b(:,k) = mean(Y,2); % bias mean vector (crude mean filter)
        [U,S,V] = svd((Y-b(:,k)*ones(1,sl))'*R,0); % POD
        C = V(:,1:q)*S(1:q,1:q)*U(:,1:q)'; % projection
        Yh(:, k+(0:sl-1) ) = Yh(:, k+(0:sl-1) )...
                + ( R\C + b(:,k)*ones(1,sl) )*W; % adjustment with weight
        wa(k+(0:sl-1)) = wa(k+(0:sl-1)) + wd; % weight these points got
    end % filtered phase space, now adjust for curvature
    % trim the padded edges and normalize each point by the weight it
    % got, so the first and last l points are weighted averages too
    Yh = Yh(:,l+1:npr-l)./( ones(d,1)*wa(l+1:npr-l) );
    if u == 0 % now adjust for curvature
        Yh = Yh + b(:,2:end-1) - (b(:,1:end-2)+b(:,3:end))/2;
    end 
end
%% Shifting coordinates so they align in columns
xf = zeros(d,n);
for k = 1:d
    xf(k,:) = Yh(k,(m*(d-k)+1):end-m*(k-1));
end
xf = mean(xf, 1);
