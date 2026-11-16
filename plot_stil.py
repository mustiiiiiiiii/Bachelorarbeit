"""Gemeinsamer Abbildungsstil: Satz aller Beschriftungen mit LaTeX (Computer Modern wie im Dokument) und
deutsches Dezimalkomma auf den Achsen.

Verwendung:  import plot_stil; plot_stil.anwenden(); ax.yaxis.set_major_formatter(plot_stil.komma(2))
Hinweis: Mit usetex werden Beschriftungen von LaTeX gesetzt; Prozentzeichen daher als r"\\%" schreiben.
"""

import shutil

import matplotlib
from matplotlib.ticker import FuncFormatter

USETEX = shutil.which("latex") is not None


def anwenden(**extra) -> None:
    """Setzt die gemeinsamen rcParams (Schriftgrößen wie bisher, Serifenschrift, LaTeX-Satz)."""
    params = {
        "text.usetex": USETEX,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    }
    params.update(extra)
    matplotlib.rcParams.update(params)


def zahl(v: float, stellen: int = 2, kuerzen: bool = False) -> str:
    """Zahl mit Dezimalkomma im Mathematikmodus, z. B. -1,5 -> $-1{,}5$."""
    s = f"{v:.{stellen}f}"
    if kuerzen and "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", "-0.0", "-0.00"):
        s = s[1:]
    s = s.replace(".", "{,}")
    return f"${s}$" if USETEX else s.replace("{,}", ",")


def komma(stellen: int = 2, kuerzen: bool = False) -> FuncFormatter:
    """Achsenformatierer mit Dezimalkomma."""
    return FuncFormatter(lambda v, _: zahl(v, stellen, kuerzen))
