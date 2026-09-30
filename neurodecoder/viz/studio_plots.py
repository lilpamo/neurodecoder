"""Studio figures, drawn from analysis/ outputs only. Returns PNG bytes.

Uses matplotlib's object API (no pyplot state), so figures are independent.
"""

import io

import numpy as np
from matplotlib.figure import Figure

from neurodecoder.analysis.psth import PSTH


def _png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return buf.getvalue()


def unit_figure(
    trial: np.ndarray, rel: np.ndarray, p: PSTH, window, title: str, baseline: bool
) -> bytes:
    """Raster (top) and PSTH mean ± SEM (bottom) sharing the time axis."""
    fig = Figure(figsize=(6.4, 5.2))
    ax_r, ax_p = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    ax_r.plot(rel, trial, "|", color="#222", markersize=2.5, markeredgewidth=0.6)
    ax_r.set_ylim(p.n_trials - 0.5, -0.5)
    ax_r.set_ylabel("trial")
    ax_r.set_title(title, fontsize=10, loc="left")
    ax_p.fill_between(p.bin_centers, p.mean - p.sem, p.mean + p.sem, color="#4a7bd0", alpha=0.3)
    ax_p.plot(p.bin_centers, p.mean, color="#1f4fa0", lw=1.4)
    ax_p.set_ylabel("Δ rate from baseline (Hz)" if baseline else "rate (Hz)")
    ax_p.set_xlabel("time from event (s)")
    ax_p.set_xlim(*window)
    for ax in (ax_r, ax_p):
        ax.axvline(0, color="#c33", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def population_figure(
    scaled: np.ndarray, centers: np.ndarray, mean: np.ndarray, sem: np.ndarray, window, title: str
) -> bytes:
    """Heatmap of row-scaled PSTHs, (n_units, n_bins) already sorted, and the selection mean."""
    fig = Figure(figsize=(6.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[40, 1], height_ratios=[2.4, 1], wspace=0.04)
    ax_h, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, 0], sharex=ax_h)
    signed = bool((scaled < 0).any())
    im = ax_h.imshow(
        scaled,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r" if signed else "viridis",
        vmin=-1 if signed else 0,
        vmax=1,
        extent=(window[0], window[1], scaled.shape[0], 0),
    )
    fig.colorbar(im, cax=cax, label="rate / unit's max |rate|")
    ax_h.set_ylabel("unit (sorted by peak time)")
    ax_h.set_title(title, fontsize=10, loc="left")
    ax_m.fill_between(centers, mean - sem, mean + sem, color="#888", alpha=0.35)
    ax_m.plot(centers, mean, color="#222", lw=1.3)
    ax_m.set_ylabel("mean ± SEM\nacross units (Hz)")
    ax_m.set_xlabel("time from event (s)")
    ax_m.set_xlim(*window)
    for ax in (ax_h, ax_m):
        ax.axvline(0, color="#c33", lw=0.8)
    ax_m.spines[["top", "right"]].set_visible(False)
    return _png(fig)
