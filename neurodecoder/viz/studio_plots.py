"""Studio figures, drawn from analysis/ outputs only. Returns PNG bytes.

Colours are the dataviz reference palette's roles, stepped separately for light and
dark: ink, gridlines, one series blue, a single-hue blue ramp for magnitude and a
blue-grey-red ramp for signed values. The event line is muted ink, not a status
colour. Figures have a transparent background and a fixed layout, so the page can
map clicks on heatmap rows back to units.

Uses matplotlib's object API (no pyplot state), so figures are independent.
"""

import io

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from neurodecoder.analysis.psth import PSTH

_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
THEMES = {
    "light": {
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": "#2a78d6",
        "sequential": _BLUE,
        "diverging": ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#c22f2f"],
    },
    "dark": {
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": "#3987e5",
        # Near zero recedes into the dark surface; high values are bright.
        "sequential": _BLUE[::-1],
        "diverging": ["#6da7ec", "#256abf", "#383835", "#c22f2f", "#ec8a89"],
    },
}
_DPI = 110


def _theme(name: str) -> dict:
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}")
    return THEMES[name]


def _style(ax, t: dict) -> None:
    ax.set_facecolor("none")
    ax.tick_params(colors=t["ink2"], labelsize=8, length=3)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])


def _png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, transparent=True)
    return buf.getvalue()


def unit_figure(trial: np.ndarray, rel: np.ndarray, p: PSTH, window, baseline: bool, theme: str):
    """Raster (top) and PSTH mean ± SEM (bottom) sharing the time axis."""
    t = _theme(theme)
    fig = Figure(figsize=(5.6, 4.6))
    ax_r = fig.add_axes((0.13, 0.47, 0.84, 0.50))
    ax_p = fig.add_axes((0.13, 0.11, 0.84, 0.31), sharex=ax_r)
    ax_r.plot(rel, trial, "|", color=t["ink"], markersize=2.2, markeredgewidth=0.6)
    ax_r.set_ylim(p.n_trials - 0.5, -0.5)
    ax_r.set_ylabel("trial")
    ax_r.tick_params(labelbottom=False)
    ax_p.fill_between(
        p.bin_centers, p.mean - p.sem, p.mean + p.sem, color=t["series"], alpha=0.2, lw=0
    )
    ax_p.plot(p.bin_centers, p.mean, color=t["series"], lw=2)
    ax_p.set_ylabel("Δ rate (Hz)" if baseline else "rate (Hz)")
    ax_p.set_xlabel("time from event (s)")
    ax_p.set_xlim(*window)
    ax_p.grid(axis="y", color=t["grid"], lw=0.6)
    ax_p.set_axisbelow(True)
    for ax in (ax_r, ax_p):
        _style(ax, t)
        ax.axvline(0, color=t["muted"], lw=1, ls=(0, (3, 3)))
    return _png(fig)


def population_figure(
    scaled: np.ndarray, centers: np.ndarray, mean: np.ndarray, sem: np.ndarray, window, theme: str
) -> tuple[bytes, tuple[float, float, float, float]]:
    """Heatmap of row-scaled PSTHs, (n_units, n_bins) already sorted, and the selection mean.

    Returns the PNG and the heatmap's box (left, top, right, bottom) as fractions of
    the image, top-down, so row i spans top + (bottom - top) * [i, i + 1] / n_units.
    """
    t = _theme(theme)
    signed = bool((scaled < 0).any())
    cmap = LinearSegmentedColormap.from_list(
        "studio", t["diverging"] if signed else t["sequential"]
    )
    fig = Figure(figsize=(5.6, 5.6))
    box = (0.13, 0.40, 0.72, 0.56)  # left, bottom, width, height
    ax_h = fig.add_axes(box)
    cax = fig.add_axes((0.87, 0.40, 0.025, 0.56))
    ax_m = fig.add_axes((0.13, 0.08, 0.72, 0.24), sharex=ax_h)
    im = ax_h.imshow(
        scaled,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-1 if signed else 0,
        vmax=1,
        extent=(window[0], window[1], scaled.shape[0], 0),
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("rate / unit's max |rate|", color=t["ink2"], size=8)
    cb.outline.set_visible(False)
    cax.tick_params(colors=t["ink2"], labelsize=7, length=2)
    ax_h.set_ylabel("unit (sorted by peak time)")
    ax_h.tick_params(labelbottom=False)
    ax_m.fill_between(centers, mean - sem, mean + sem, color=t["series"], alpha=0.2, lw=0)
    ax_m.plot(centers, mean, color=t["series"], lw=2)
    ax_m.set_ylabel("mean ± SEM (Hz)")
    ax_m.set_xlabel("time from event (s)")
    ax_m.set_xlim(*window)
    ax_m.grid(axis="y", color=t["grid"], lw=0.6)
    ax_m.set_axisbelow(True)
    for ax in (ax_h, ax_m):
        _style(ax, t)
        ax.axvline(0, color=t["muted"], lw=1, ls=(0, (3, 3)))
    left, bottom, width, height = box
    return _png(fig), (left, 1 - bottom - height, left + width, 1 - bottom)
