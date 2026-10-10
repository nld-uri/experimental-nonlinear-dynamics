"""LaTeX text rendering for matplotlib figures.

use_latex_fonts() switches matplotlib to real LaTeX typesetting, so symbols and
equations on a figure match the surrounding document instead of being rendered
by matplotlib's Computer Modern lookalike (mathtext).

Needs a working TeX: latex + dvipng on PATH.  On macOS with MacTeX that is
/Library/TeX/texbin, which this module adds automatically.  If TeX is missing,
it falls back to mathtext's cm fontset -- visually close, never an exception --
and returns False so the caller can tell.

latexify(fig) rewrites every text element of a finished figure so that
characters LaTeX treats specially (%, &, _, #) and common unicode symbols
(sigma, omega, +/-, x, ~, en/em dashes) become valid LaTeX.  Spans already
inside $...$ are left untouched, so hand-written math survives.
"""
import os
import shutil

TEXBIN = "/Library/TeX/texbin"

PREAMBLE = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{type1cm}"

# unicode -> LaTeX, applied only OUTSIDE existing $...$ spans
UNICODE_MAP = {
    "\u03b1": r"$\alpha$", "\u03b2": r"$\beta$", "\u03b3": r"$\gamma$",
    "\u0394": r"$\Delta$", "\u03b4": r"$\delta$", "\u03b5": r"$\epsilon$",
    "\u03b6": r"$\zeta$",  "\u03b7": r"$\eta$",   "\u03b8": r"$\theta$",
    "\u03bb": r"$\lambda$","\u03bc": r"$\mu$",    "\u03bd": r"$\nu$",
    "\u03c0": r"$\pi$",    "\u03c1": r"$\rho$",   "\u03c3": r"$\sigma$",
    "\u03c4": r"$\tau$",   "\u03c6": r"$\phi$",   "\u03c7": r"$\chi$",
    "\u03c8": r"$\psi$",   "\u03c9": r"$\omega$", "\u03a9": r"$\Omega$",
    "\u00b1": r"$\pm$",    "\u2213": r"$\mp$",    "\u00d7": r"$\times$",
    "\u00b7": r"$\cdot$",  "\u2248": r"$\approx$","\u2260": r"$\neq$",
    "\u2264": r"$\leq$",   "\u2265": r"$\geq$",   "\u221e": r"$\infty$",
    "\u2192": r"$\rightarrow$", "\u2190": r"$\leftarrow$",
    "\u2265": r"$\geq$",   "\u223c": r"$\sim$",   "~": r"$\sim$",
    "\u2014": "---", "\u2013": "--", "\u2212": "$-$",
    "\u00b0": r"$^\circ$", "\u2032": r"$'$", "\u2033": r"$''$",
}
LATEX_ESCAPES = {"%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_"}


def use_latex_fonts(serif=True, preamble=None, size=None):
    """Switch matplotlib to LaTeX text rendering. Returns True if TeX was found."""
    import matplotlib as mpl
    if preamble is None:
        preamble = PREAMBLE
    if os.path.isdir(TEXBIN) and TEXBIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = TEXBIN + os.pathsep + os.environ.get("PATH", "")
    ok = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    if ok:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif" if serif else "sans-serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": preamble,
        })
    else:
        # closest available without a TeX installation
        mpl.rcParams.update({
            "text.usetex": False, "mathtext.fontset": "cm",
            "font.family": "serif" if serif else "sans-serif",
            "font.serif": ["CMU Serif", "DejaVu Serif"],
        })
    if size is not None:
        mpl.rcParams["font.size"] = size
    return ok


def tex_escape(s):
    """Make one string safe for LaTeX, preserving any existing $...$ math spans."""
    out, math = [], False
    for part in str(s).split("$"):
        if math:
            out.append("$" + part + "$")
        else:
            for k, v in LATEX_ESCAPES.items():
                part = part.replace(k, v)
            for k, v in UNICODE_MAP.items():
                part = part.replace(k, v)
            out.append(part)
        math = not math
    joined = "".join(out)
    # a trailing unmatched $ would break the run; drop it rather than fail
    return joined if joined.count("$") % 2 == 0 else joined.replace("$", "")


def escaping_formatter(inner):
    """Wrap a matplotlib Formatter so its output is LaTeX-safe.

    Two traps this avoids, both of which fail SILENTLY:
      * it must subclass Formatter -- set_major_formatter wraps a bare
        callable in a FuncFormatter, which never forwards set_locs, and
        ScalarFormatter then returns an empty string for every tick;
      * it must delegate format_ticks, the API matplotlib uses at draw time,
        so stateful formatters (ScalarFormatter's offset and order of
        magnitude, LogFormatter's minor-tick suppression) keep working.

    Defined inside a function so this module is valid as a skill sidecar.
    """
    from matplotlib.ticker import Formatter

    class _EscapingFormatter(Formatter):
        _escaped = True

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __call__(self, x, pos=None):
            return tex_escape(self._inner(x, pos))

        def format_ticks(self, values):
            return [tex_escape(v) for v in self._inner.format_ticks(values)]

        def set_axis(self, axis):
            self._inner.set_axis(axis)

        def get_offset(self):
            return tex_escape(self._inner.get_offset())

    return _EscapingFormatter(inner)


def latexify(fig):
    """Escape every text element of a finished figure. Call just before savefig."""
    import matplotlib as mpl
    if not mpl.rcParams.get("text.usetex", False):
        return fig
    # Tick labels are produced by a Formatter at DRAW time, so escaping the
    # Text objects is not enough.  Delegate through the formatter instead: an
    # unescaped %% makes LaTeX treat the rest of the label as a comment, and the
    # label silently loses its tail rather than raising.
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for get, set_ in ((axis.get_major_formatter, axis.set_major_formatter),
                              (axis.get_minor_formatter, axis.set_minor_formatter)):
                f = get()
                if not getattr(f, "_escaped", False):
                    set_(escaping_formatter(f))
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text()
        if not s or not s.strip():
            continue
        if "\\mathdefault" in s:     # matplotlib's own auto tick labels
            continue
        e = tex_escape(s)
        if e != s:
            t.set_text(e)
    return fig


# ---------------------------------------------------------------- MATLAB colors
# MATLAB's default axes ColorOrder since R2014b, in order.
MATLAB_ORDER = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E",
                "#77AC30", "#4DBEEE", "#A2142F"]

MATLAB = {"blue": "#0072BD", "orange": "#D95319", "yellow": "#EDB120",
          "purple": "#7E2F8E", "green": "#77AC30", "cyan": "#4DBEEE",
          "darkred": "#A2142F",
          # not part of ColorOrder but MATLAB's usual neutrals
          "black": "#000000", "gray": "#808080", "lightgray": "#BFBFBF",
          "darkyellow": "#9C7A0A",   # legible text version of the yellow
          # Two additions, chosen by maximising the minimum CIEDE2000 distance
          # under simulated deuteranopia against the rest of the palette.
          # MATLAB's ColorOrder is not colour-vision-safe: orange/green sit at
          # dE 12.4, yellow/green at 9.7, blue/purple at 9.9 and purple/gray at
          # 2.7.  Use these two wherever those pairs would share a panel.
          "darkgreen": "#055432",    # for green beside orange or yellow: dE 9.7 -> 31.9
          "plum": "#67355C"}         # MATLAB purple's hue, darkened: dE 2.7 -> 17.6

# Black reads as a hole between saturated bars.  Use it for lines only; in bar
# charts and solid fills substitute this (MATLAB's own dark red, dE 21.9 from
# blue / orange / darkgreen under deuteranopia).
BAR_BLACK_SUBSTITUTE = "#A2142F"

# sRGB -> XYZ (D65) and the Vienot 1999 deuteranopia LMS transform, as literals
# so this module stays valid as a skill sidecar.
XYZ_D65 = [[0.4124564, 0.3575761, 0.1804375],
          [0.2126729, 0.7151522, 0.0721750],
          [0.0193339, 0.1191920, 0.9503041]]
LMS_HPE = [[0.31399022, 0.63951294, 0.04649755],
          [0.15537241, 0.75789446, 0.08670142],
          [0.01775239, 0.10944209, 0.87256922]]


def use_matlab_colors():
    """Set the matplotlib property cycle to MATLAB's default ColorOrder.

    Applies to line plots, bars and any solid fill that takes its color from
    the cycle.  Colormaps (gradients) are unaffected -- MATLAB's parula has no
    matplotlib equivalent; use 'viridis' for a perceptually uniform ramp.
    """
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams["axes.prop_cycle"] = cycler(color=MATLAB_ORDER)
    return MATLAB_ORDER


def apply_preferences(sizes=None, serif=True):
    """LaTeX text + MATLAB colors in one call. Returns True if TeX was found."""
    ok = use_latex_fonts(serif=serif)
    use_matlab_colors()
    if sizes is not None:
        import matplotlib as mpl
        mpl.rcParams.update({"font.size": sizes[0], "axes.titlesize": sizes[0],
                             "axes.labelsize": sizes[0], "legend.fontsize": sizes[1],
                             "xtick.labelsize": sizes[2], "ytick.labelsize": sizes[2]})
    return ok


def bar_colors(colors):
    """Map black to BAR_BLACK_SUBSTITUTE. Use for bars and solid fills.

    Black is fine for a line -- it reads as a reference overlay -- but a black
    bar beside saturated ones reads as a gap rather than a category.
    """
    out = []
    for c in colors:
        c = MATLAB.get(c, c)
        out.append(BAR_BLACK_SUBSTITUTE
                   if str(c).lower() in ("k", "black", "#000000", "#000") else c)
    return out


def cvd_check(colors, threshold=15.0):
    """Pairwise CIEDE2000 under simulated deuteranopia.

    Returns [(dE, colour_a, colour_b), ...] for every pair closer than
    `threshold`, worst first -- empty list means the set is safe.  A dE below
    about 15 means two series are hard to tell apart for a red-green colour
    blind reader, which is roughly 8% of men.  Run it on any categorical set
    before shipping the figure.
    """
    import numpy as np
    from itertools import combinations
    xyz_m, lms_m = np.array(XYZ_D65), np.array(LMS_HPE)
    wp = xyz_m @ np.ones(3)

    def _lin(c):
        c = np.array([int(str(c).lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)])
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def _lab_of(lin):
        xyz = xyz_m @ lin / wp
        f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
        return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])

    def _deut(c):
        l = lms_m @ _lin(c)
        return np.linalg.inv(lms_m) @ np.array([l[0], 0.494207 * l[0] + 1.24827 * l[2], l[2]])

    def _de(p, q):
        (l1, a1, b1), (l2, a2, b2) = p, q
        c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
        cb = (c1 + c2) / 2
        g = 0.5 * (1 - np.sqrt(cb ** 7 / (cb ** 7 + 25.0 ** 7))) if cb > 0 else 0.5
        a1p, a2p = (1 + g) * a1, (1 + g) * a2
        c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
        h1 = np.degrees(np.arctan2(b1, a1p)) % 360
        h2 = np.degrees(np.arctan2(b2, a2p)) % 360
        dl, dc = l2 - l1, c2p - c1p
        dh = 0.0 if c1p * c2p == 0 else ((h2 - h1 + 180) % 360) - 180
        dhp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2)
        lb, cbp = (l1 + l2) / 2, (c1p + c2p) / 2
        if c1p * c2p == 0:
            hb = h1 + h2
        elif abs(h1 - h2) <= 180:
            hb = (h1 + h2) / 2
        else:
            hb = (h1 + h2 + 360) / 2 if h1 + h2 < 360 else (h1 + h2 - 360) / 2
        tt = (1 - 0.17 * np.cos(np.radians(hb - 30)) + 0.24 * np.cos(np.radians(2 * hb))
              + 0.32 * np.cos(np.radians(3 * hb + 6)) - 0.20 * np.cos(np.radians(4 * hb - 63)))
        sl = 1 + 0.015 * (lb - 50) ** 2 / np.sqrt(20 + (lb - 50) ** 2)
        sc, sh = 1 + 0.045 * cbp, 1 + 0.015 * cbp * tt
        rt = (-2 * np.sqrt(cbp ** 7 / (cbp ** 7 + 25.0 ** 7))
              * np.sin(np.radians(60 * np.exp(-(((hb - 275) / 25) ** 2)))))
        return np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dhp / sh) ** 2
                       + rt * (dc / sc) * (dhp / sh))

    cols = [MATLAB.get(c, c) for c in colors]
    labs = {c: _lab_of(_deut(c)) for c in cols}
    bad = [(round(float(_de(labs[a], labs[b])), 1), a, b)
           for a, b in combinations(cols, 2) if _de(labs[a], labs[b]) < threshold]
    return sorted(bad)
