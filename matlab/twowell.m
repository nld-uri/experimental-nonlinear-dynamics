function [x, fs, npp] = twowell(n)
% TWOWELL  The book's measured record (Chapter 1, Section sec:twowelldata).
%
%   [x, fs, npp] = twowell()   returns the whole record
%   [x, fs, npp] = twowell(n)  returns its first n samples
%
%   x    strain-gauge voltage of the notched cantilever beam, V
%   fs   160 Hz, the sampling frequency
%   npp  16, the number of samples per forcing period (10 Hz forcing)
%
%   Two-well magneto-elastic oscillator, run of 2 June 2002, the rig of
%   Chelidze and Liu, J. Sound Vib. 281 (2004) 887-904.
%   96,000 samples = 600 s = 6000 forcing periods, 16-bit A/D over
%   +/-5 V (quantization step 10/65536 V). Forcing at 10 Hz.
p = fileparts(mfilename('fullpath'));
x = load(fullfile(p, '..', 'data', 'twowell_2002-06-02.csv'));
if nargin > 0 && ~isempty(n), x = x(1:n); end
fs = 160; npp = 16;
end
