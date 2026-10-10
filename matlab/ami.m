function [ Ixx ] = ami( x, k, nb )
% AMI calculates average mutual information for the input scalar time
%     series x. If no output is requested, it plots the AMI curve vs delay.
%
% CALL: Ixx = ami( x, k, nb );
%
% INPUT:
%   x  - first input scalar time series
%   k  - maximum delay to consider
%   nb - number of bins to use in calculation for x (default: nb = 32)
% OUTPUT:
%   Ixx - average mutual information in bits for 0 to k delay
%
% Copyright and Revised by David Chelidze 8/29/18, 9/5/18, 12/14/21
% chelidze@uri.edu
 
if nargin < 2 % not enough inputs
   error('Requires at least two input arguments!')   
elseif nargin == 2 % set the number of bins to the default
   nb = 32; 
end

% check the data and make a column vector of data
if min(size(x)) ~= 1
   error('First input should be a vector of scalar data points!')
else % make it into column vector and get its size
    x = x(:);
    n = size(x,1); 
end

% get the domain boundaries and each bin size
minx = min(x); maxx = max(x); % domain
binx = (maxx - minx)/nb; % bin size
xx = minx + binx*(0:nb); % get the bin boundaries (edges)
xx(1) = -inf; xx(end) = inf; % extend the rage of the end points

% obtain the binned probability density for the data
Px = nonzeros( histcounts(x, xx, 'Normalization', 'probability') );
 
Ixx = zeros(k+1,1); % initialize Ixy
for ii = 0:k % calculate the average mutual information for each delay
    indx = 1:n-ii; % for the positive delays
    Pxx =  nonzeros( histcounts2(x(indx), x(indx + ii), xx, xx, ...
                                    'Normalization', 'probability') );
    Ixx(1+ii) = sum(Pxx.*log2(Pxx)) - 2*sum(Px.*log2(Px));
end
 
if nargout == 0 % plot the results
    plot(0:k, Ixx, 'o-', 'markerfacecolor', 'w')
    grid on, pbaspect([1.618 1 1])
    xlabel('Delay (samples)')
    ylabel('Average Mutual Information (bits)')
end
