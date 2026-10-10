# Measured records

## `twowell_2002-06-02.csv`

Strain-gauge output of the two-well magneto-elastic oscillator, run of
2 June 2002. Notched cantilever beam between two rare-earth magnets on a
shaker; strain gauge close to the clamped end, above the notch. The rig
and the run are those of

> D. Chelidze and M. Liu, Dynamical systems approach to fatigue damage
> identification, *J. Sound Vib.* **281** (2004) 887-904.
> doi:10.1016/j.jsv.2004.02.017

One column, no header, 96,000 samples, volts.

| quantity | value | how it is known |
|---|---|---|
| sampling frequency | 160 Hz | recorded at the time |
| forcing frequency | 10 Hz | recorded at the time |
| samples per forcing period | 16 | measured: the drive line and its harmonics fall at exactly f_s/16, f_s/8, 3f_s/16 |
| A/D | 16 bit over +/-5 V | measured: quantization step is exactly 10/65536 V |
| record length | 96,000 samples = 600 s = 6000 forcing periods | measured |
| mean, standard deviation | -0.187 V, 1.582 V | measured |
| range | -3.343 to 3.232 V | measured |

Note that the source paper states a forcing frequency of 8 Hz for the
experimental campaign. This particular run was driven at 10 Hz, which is
what the record shows: 16 samples per forcing period, with the drive line
and its harmonics at f_s/16, f_s/8 and 3f_s/16 and nothing at 8 or 16 Hz.

Loaders: `python/_twowell.py`, `matlab/twowell.m`.

## `psw_fatigue_2005.mat`

The finished phase-space-warping tracking matrix of the fatigue experiment in
variable operating conditions of

> D. Chelidze and M. Liu, Reconstructing slow-time dynamics from fast-time
> measurements, *Phil. Trans. R. Soc. A* **366** (2008) 729-745.
> doi:10.1098/rsta.2007.2124

Same rig as the two-well record above, with a notch that grows a fatigue crack;
the beam fractures at about 12.5 h. Used in Chapter 17, Section 17.4.

| variable | shape | what it is |
|---|---|---|
| `ss_psw` | 405 x 1728 | the tracking matrix, TRANSPOSED (rows are features, columns are records) |
| `t` | 1 x 1728 | time in hours, 0 to 12.281, one record every 25.6 s |
| `damage1`, `damage2` | 1 x 1728 | **the two electromagnet supply voltages, not damage** |

**The names `damage1` and `damage2` do not describe their contents.** They hold
the two operating-condition signals: sinusoids at a 2:1 frequency ratio (1.517
and 0.759 rad/h), mean 5.0 and amplitude 4.5 in the file's units, whose
Lissajous figure is the figure-eight of the source paper's figure 9(a). The
damage variable is not in the file; recovering it from the complement of the
subspace these two occupy is the point of the example.

The frequencies also differ from the paper's stated `v1 = 10 + 9 sin(t)` and
`v2 = 10 + 9 cos(0.5 t)`: the ratio is 2:1 as stated, but the rates are about
1.5 times faster and the amplitudes half. Only the 2:1 ratio matters to the
analysis.

### The raw strain record

It survives, as `cd.mat` (one variable, `strn`): 7,080,000 samples at 160 Hz,
12.29 h, 54 MB, created 21 Nov 2005. The first 1728 x 2^12 = 7,077,888 samples
are exactly the 1728 windows of the tracking matrix; 2,112 samples (13.2 s) are
left over at the end. Verified by `python/chk_rawmatch.py`, which rebuilds the
two-stage analysis from the raw record and reproduces every number of the
supplied matrix.

**It is not in this folder.** At 54 MB it is too large for the Overleaf tree, so
it lives beside the project. Put it at any of the paths
`python/_psw2005.py:RAW_CANDIDATES` lists — the simplest is one level above the
book tree, named `cd_raw_strain_2005.mat` — or set the environment variable
`PSW_RAW` to wherever you keep it. Three scripts need it:
`chk_rawmatch.py`, `chk_farfield_exp.py`, `mkfig_psw_farfield.py`. Nothing in
the book's own build does.

The recipe in the source paper is complete enough to rebuild the tracking matrix
from it — d = 5, tau = 6, local linear model on 16 neighbours, reference model
from the first 2^15 points, 1728 records of 2^12 points, 81 equally populated
hypercuboids, five PSW components averaged in each.

Loader: `python/_psw2005.py`.
