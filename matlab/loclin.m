function yp = loclin( ym, yq, n, N, w, tw, iq )
% LOCLIN local linear modeling for prediction
%
% INPUT:
%   ym -- reconstructed phase space points (trajectory) used for modeling
%         should be d times M, where M is the number of points
%   yq -- initial point to start the prediction from
%         should be d times 1
%   n -- number of nearest neighbors to use for prediction
%   N -- how far ahead to do the prediction (number of time steps)
%   w -- optional, to use weighting w = 1, otherwise use no weighting.
%        The weight is w(delta) = a^4/(delta^4 + a^4), Eq. (loclinw) of
%        Chapter 15, with a the distance from the query point to its n-th
%        (last requested) neighbor, so the scale follows the local
%        density and the n-th neighbor gets half the weight of a
%        coincident point.
%   tw -- optional Theiler window (default 0: no exclusion); used with iq
%   iq -- optional column of ym that yq is (in-sample query); at step k
%         the columns j with |j - (iq+k-1)| <= tw are not used as neighbors
% OUTPUT:
%   yp -- predicted phase space trajectory starting from yq point
%
% EXAMPLE:
%   x = x(:)'; % experimental time series or generated from some model
%   np = size(x,2) - (d-1)*m; % maximum number of reconstructed points
%   y = zeros(d,np); % reconstructed phase space trajectory
%   for k = 1:d
%       y(k,:) = x((1:np) + (k-1)*m);
%   end
%   % use half points for modeling and other half for testing:
%   yp = loclin( y(:,1:np/2), y(:,np/2 + 10), 8, 1000 );
%   plot(yp(1,:)), hold on, plot(y(1,n/2+10:n/2+1000))
%
% written and copyrighted by David Chelidze on 12/4/17

[d, M] = size(ym);
ns = createns(ym(:,1:end-1)','NSMethod','kdtree','Distance','euclidean');
yp = [yq zeros(d,N)];
if nargin < 5 || w ~= 1 % use no weighting
    w = 0;
end
if nargin < 7 % no in-sample query index: no Theiler exclusion
    tw = 0; iq = 0;
end
excl = tw > 0;
if excl
    kq = min(n + 2*tw + 1, M - 1); % extra neighbors to discard
else
    kq = n;
end

for k = 1:N
    [indx, dis] = knnsearch(ns,yq','Distance','euclidean','k',kq);
    if excl % Theiler exclusion around the current trajectory point
        keep = abs(indx - (iq + k - 1)) > tw;
        indx = indx(keep); dis = dis(keep);
        indx = indx(1:n); dis = dis(1:n);
    end
    Yn = [ym(:,indx); ones(1,n)];   % (d+1) x n design X~
    Ynp1 = ym(:,indx+1);            % d x n images
    a = dis(end);                   % distance to the n-th neighbor
    if w == 1 && a > 0
        sw = sqrt(a^4 ./ ( dis.^4 + a^4 ))'; % square roots of weights, n x 1
    else
        sw = ones(n,1);
    end
    % weighted least squares on sqrt(W)*X~' directly (QR), no Gram matrix
    B = ( (sw.*Yn') \ (sw.*Ynp1') )';     % d x (d+1) = [A, b]
    yp(:,k+1) = B*[yq; 1];
    yq = yp(:,k+1);
end
