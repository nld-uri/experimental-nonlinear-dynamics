function [ Ixy ] = ami_x( x, y, k, nbx, nby )
% AMI_X calculates average-cross-mutual information for the input scalar
%       time series x and y. If no output requested, plots the AMI curve.
%
% CALL: Ixy = ami_x( x, y, m, nbx, nby );
%
% INPUT:
%   x  - first input scalar time series
%   y  - second inpit scalar time series
%   k  - maximum delay to consider
%   nbx - number of bins to use in calculation for x (default: nbx = 32)
%   nby - number of bins to use in calculation for y (default: nby = nbx)
% OUTPUT:
%   Ixy - average mutual information in bits for - k to k delay
%
% Copyright and Revised by David Chelidze 8/29/18, 9/5/18, 12/14/21
% chelidze@uri.edu
 
if nargin < 3 % not enough inputs provided
   error('Requires at least three input arguments!')   
elseif nargin == 3 % set the number of bins to default
   nbx = 32; 
   nby = 32;
elseif nargin == 4 % set bins for y the same as for x
    nby = nbx;
end

% check the input needed data's shape
if min(size(x)) ~= 1 || min(size(y)) ~= 1
   error('First two inputs should be vectors of scalar data points!')
else % make them into column vectors and get their sizes
    x = x(:); y = y(:);
    n = size(x,1); 
    m = size(y,1);
end

% get the data domain boundaries and the uniform bin sizes
minx = min(x); maxx = max(x);
binx = (maxx - minx)/nbx;
miny = min(y); maxy = max(y);
biny = (maxy - miny)/nby;
% get the corresponding bin boundaries (edges)
xx = minx + binx*(0:nbx);
xx(1) = -inf; xx(end) = inf;
yy = miny + biny*(0:nby);
yy(1) = -inf; yy(end) = inf;

% obtain Px and Py, probability density functions
Px = nonzeros( histcounts(x, xx, 'Normalization', 'probability') );
Py = nonzeros( histcounts(y, yy, 'Normalization', 'probability') );

Ixy = zeros(2*k+1, 1); % initialize Ixy
% evaluate the AMI at zero delay by first finding the joint pdf
indx = 1:min(m, n);
Pxy =  nonzeros( histcounts2(x(indx), y(indx), xx, yy, ...
                                 'Normalization', 'probability') );
Ixy(k+1) = sum(Pxy.*log2(Pxy)) - sum(Px.*log2(Px)) ...
                               - sum(Py.*log2(Py));
% now evaluate AMI for the other delays
for ii = 1:k % calculate the average mutual information
    indx = 1:min(m,n-ii); % for the negative delays
    Pxy =  nonzeros( histcounts2(x(indx + ii), y(indx), xx, yy, ...
                                 'Normalization', 'probability') );
    Ixy(k+1-ii) = sum(Pxy.*log2(Pxy)) - sum(Px.*log2(Px)) ...
                                   - sum(Py.*log2(Py));
    indx = 1:min(n,m-ii); % for the positive delays
    Pxy =  nonzeros( histcounts2(x(indx), y(indx + ii), xx, yy, ...
                                 'Normalization', 'probability') );
    Ixy(k+1+ii) = sum(Pxy.*log2(Pxy)) - sum(Px.*log2(Px)) ...
                                   - sum(Py.*log2(Py));
end
 
if nargout == 0 % plot the results
    plot(-k:k, Ixy, 'o-', 'markerfacecolor', 'w')
    grid on, pbaspect([1.618 1 1])
    xlabel('Delay (samples)')
    ylabel('Average Mutual Information (bits)')
end
