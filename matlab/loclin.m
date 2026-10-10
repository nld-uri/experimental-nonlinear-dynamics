function yp = loclin( ym, yq, n, N, w )
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

d = size(ym,1);
ns = createns(ym(:,1:end-1)','NSMethod','kdtree','Distance','euclidean');
yp = [yq zeros(d,N)];
if nargin < 5 || w ~= 1 % use no weighting
    w = 0;
end

for k = 1:N
    [indx, dis] = knnsearch(ns,yq','Distance','euclidean','k',n);
    Yn = [ym(:,indx); ones(1,n)];
    Ynp1 = ym(:,indx+1);
    if w == 1 % use weighting, scale a = distance to the n-th neighbor
        a = dis(end);
        if a > 0
            W = diag(a^4 ./ ( dis.^4 + a^4 )); % weighting fun (can make your own)
        else
            W = eye(n);
        end
        yp(:,k+1) = ((Ynp1*W*Yn')/(Yn*W*Yn'))*[yq; 1];
    else
        yp(:,k+1) = ((Ynp1*Yn')/(Yn*Yn'))*[yq; 1];
    end
    yq = yp(:,k+1);
end
