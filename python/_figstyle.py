"""Shared plotting style for the regenerated figures of Chapters 2-5 (not a listing).

Every mkfig_*.py in this directory does `from _figstyle import *` and writes its
PDF to FIGDIR, which is ../figures/ relative to this file, so the scripts run from
the python/ folder (or anywhere else) and always land in the book tree.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# MATLAB default colour order (CONVENTIONS.md)
COL = [(0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.929, 0.694, 0.125),
       (0.494, 0.184, 0.556), (0.466, 0.674, 0.188), (0.301, 0.745, 0.933),
       (0.635, 0.078, 0.184)]
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "mathtext.fontset": "cm",
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "legend.frameon": False,
    "legend.fontsize": 8, "axes.labelsize": 9, "pdf.fonttype": 42})
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures") + os.sep


def panel(ax, letter, x=-0.12, y=1.02):
    """Panel letter '(a)' etc. in the top-left corner outside the axes."""
    ax.text(x, y, "(%s)" % letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)


def sketch_axes(ax, xlabel="$t$", ylabel="$x(t)$", yzero=True):
    """Bare 'sketch' axes: only the left spine and a horizontal axis line through zero,
    no ticks. Used for the conceptual figures that show a generic signal."""
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_position(("data", 0))
    ax.set_xticks([]); ax.set_yticks([])
    if yzero:
        ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.xaxis.set_label_coords(1.0, 0.5)
    ax.yaxis.set_label_coords(0.0, 1.02)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha("left")
