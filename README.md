# Experimental Nonlinear Dynamics: code and data

Companion code for the book *Experimental Nonlinear Dynamics* by David Chelidze and Joseph P. Cusumano.
The code is written by David Chelidze, Nonlinear Dynamics Laboratory, University of Rhode Island
([web.uri.edu/nld](https://web.uri.edu/nld/)).

## Contents

| Folder | What it holds |
| --- | --- |
| `matlab/` | MATLAB functions used in the book (Appendix A listings) |
| `python/` | Python ports of the same functions, the scripts that draw the book's figures (`mkfig_*.py`), and numerical checks (`chk_*.py`, `test_python.py`) |
| `data/` | The two measured records used in the book; see `data/README.md` |
| `book/` | Draft PDF of the book (draft of 10 October 2026, 384 pages) |
| `figures/` | Output folder for the figure scripts |

## The book

`book/Chelidze-Cusumano_Nonlinear-Time-Series-Analysis_draft-2026-10-10.pdf` is a draft of the book, *Nonlinear Time Series Analysis: Experimental Nonlinear Dynamics from Measured Records*. Figure captions name the script in `python/` that draws each figure. The draft PDF is copyright the authors and is not covered by the MIT license, which applies to the code and data only.

## Functions

| MATLAB | Python | Purpose |
| --- | --- | --- |
| `ami.m` | `ami.py` | Average mutual information |
| `ami_x.m` | `ami_x.py` | Cross average mutual information |
| `c2r.m` | `c2r.py` | Correlation sum from a scalar time series |
| `fnn.m`, `fnnfrac.m`, `cao.m` | `fnn.py` | False nearest neighbors, composite fraction and Cao's statistics |
| `nnsearch.m` | | k nearest neighbors outside a Theiler window |
| `irsurr.m` | `irsurr.py` | Iteratively refined surrogates |
| `loclin.m` | `loclin.py` | Local linear model prediction |
| `lyapunov.m` | `lyapunov.py` | Largest short-time Lyapunov exponent metrics |
| `pwdim.m` | `pwdim.py` | Average pointwise dimension |
| `podfilt.m` | `podfilt.py` | POD-based projective noise reduction |
| `sodfilt.m` | `sodfilt.py`, `sodbundle.py` | SOD-based smooth projective noise reduction |
| | `gsvd.py`, `nlutil.py` | Generalized SVD and shared noise-reduction helpers |
| `psdwelch.m` | `psdwelch.py` | Welch spectral estimates (Chapter 6) |
| `psdmt.m`, `mtlines.m` | `psdmt.py` | Multitaper spectral estimate and harmonic F-test (Chapter 6) |
| `rqa.m` | `rqa.py` | Recurrence plot and recurrence quantification (Chapter 18) |
| `mdofmodes.m` | `mdofmodes.py` | Modal parameters and FRF matrix of a linear M-C-K system |
| `twowell.m` | `_twowell.py` | Loader for the measured two-well record |

## Getting started

Python 3 with NumPy, SciPy and Matplotlib:

```
pip install -r requirements.txt
cd python
python3 test_python.py          # checks of the Python ports
python3 mkfig_amiplots.py       # draws one book figure into ../figures/
```

MATLAB: add `matlab/` to the path. `chk_psdmt.m` checks `psdmt.m` and `mtlines.m`.

## Data

`data/twowell_2002-06-02.csv` and `data/psw_fatigue_2005.mat` are measurements from the
Nonlinear Dynamics Laboratory. Their provenance and units are in `data/README.md`.

## License

MIT License, copyright (c) 2026 David Chelidze. See `LICENSE`.

## Citing

If you use this code in published work, please cite the book and this repository.

## Known gaps

- `mkfig_position_exp.py` and `mkfig_psw_farfield.py` need the raw strain record of the 2005 fatigue
  experiment, which is not in this repository; `data/README.md` says how to point `PSW_RAW` at it.
- `mkfig_fitwindow.py` reads a precomputed curve (`/tmp/lnd.npy`) that no script here produces.
- Scripts that use `plotprefs.py` need a LaTeX installation with the `cm-super` fonts.
