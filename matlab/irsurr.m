function [rb, sb] = irsurr(s, n) 
% IRSURR generates the iteratively refined surrogates of the original time
% series.
%
% USE: >> [rb, sb] = irsurr(s, n);
%
% INPUT:
%   x -- original scalar time series
%   n -- number of iterations to do
%
% OUTPUT:
%   rb -- surrogates that exactly match in histogram (PDF)
%   sb -- surrogates that exactly match in FFT magnitudes (PSD)
%
% Written by David Chelidze on 7/12/18 according to the paper:
%
%   Schreiber, T., Schmitz, A., 2000. Surrogate time series. 
%   Physica D 142 (3-4), 346-382.
 
s = s(:); % make a column vector
N = size(s,1); % get the length of the time series
 
c = sort(s); % sort and store the magnitudes in ascending order
S = abs( fft(s) ); % store the FFT magnitudes
 
% initialize the iterations by phase randomizing the original signal
rb = real(ifft(fft(s).*exp(1i*2*pi*rand(N,1))));
 
for k = 1:n
    sb = real(ifft( S.*exp(1i*angle(fft(rb))) )); % first match the PSD
    [~, indx] = sort(sb);  % now match the PDF
    rb(indx) = c;          % by substituting the original magnitudes
end