"""Studio figures, drawn from analysis/ outputs only. Returns PNG bytes.

Colours are the dataviz reference palette's roles, stepped separately for light and
dark: ink, gridlines, one series blue, a single-hue blue ramp for magnitude and a
blue-grey-red ramp for signed values. The event line is muted ink, not a status
colour. Figures have a transparent background and a fixed layout, so the page can
map clicks on heatmap rows back to units.

Uses matplotlib's object API (no pyplot state), so figures are independent.
"""

import io
import os
from dataclasses import dataclass

import numpy as np
from matplotlib import rc_context
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from neurodecoder.analysis.psth import PSTH

# The reference palette's categorical slots, in its fixed, CVD-validated order.
_CATEGORICAL_LIGHT = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
_CATEGORICAL_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]
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
        "categorical": _CATEGORICAL_LIGHT,
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
        "categorical": _CATEGORICAL_DARK,
    },
}
_DPI = 110


def _theme(name: str) -> dict:
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}")
    return THEMES[name]


def probe_colours(probes: list[str], theme: str) -> dict[str, str]:
    """probe -> colour by its place in the session's probe list, so filtering never
    repaints a probe. Past the eighth probe, the rest share the muted ink ("other")."""
    t = _theme(theme)
    slots = t["categorical"]
    return {p: slots[i] if i < len(slots) else t["muted"] for i, p in enumerate(probes)}


def lab_colours(labs: list[str], theme: str) -> dict[str, str]:
    """lab -> colour, in the given order (largest lab first): the eight categorical slots,
    then the muted ink shared by every other lab. Never a generated hue."""
    t = _theme(theme)
    slots = t["categorical"]
    return {lab: slots[i] if i < len(slots) else t["muted"] for i, lab in enumerate(labs)}


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


def save_vector(fig: Figure, path: str | os.PathLike) -> None:
    """SVG or PDF by suffix, white background, text kept as editable text."""
    with rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        fig.savefig(path, facecolor="white")


def _frame(fig: Figure, title: str | None, ink2: str):
    """An add_axes that leaves room for a title line at the top when there is one."""
    squeeze = 1.0
    if title:
        fig.text(0.02, 0.985, title, va="top", ha="left", size=8, color=ink2, wrap=True)
        squeeze = 0.91

    def add(box, **kwargs):
        left, bottom, width, height = box
        return fig.add_axes((left, bottom * squeeze, width, height * squeeze), **kwargs)

    return add, squeeze


@dataclass(frozen=True)
class TraceGroup:
    """One condition's raster rows and PSTH; colour None draws the unsplit style."""

    name: str
    colour: str | None
    psth: PSTH
    trial: np.ndarray  # (n_spikes,) trial index within this group
    rel: np.ndarray  # (n_spikes,) seconds from the event


def condition_colours(name: str, levels: tuple[float, ...], theme: str) -> list[str]:
    """One colour per level. Sided conditions use the diverging pair: left blue, right
    red, graded by rank for contrast; 0% and the unbiased block get the muted ink. Choice
    takes the side it reports on correct trials. Outcome uses the categorical slots."""
    t = _theme(theme)
    ramp = LinearSegmentedColormap.from_list("sided", t["diverging"])

    def sided(position: float) -> str:  # 0 = strongest left, 1 = strongest right
        return to_hex(ramp(position))

    if name == "outcome":
        return [t["categorical"][1] if v < 0 else t["categorical"][0] for v in levels]
    if name == "side":
        return [sided(0.0) if v < 0 else sided(1.0) for v in levels]
    if name == "choice":  # -1 reports right, +1 reports left
        return [sided(1.0) if v < 0 else sided(0.0) for v in levels]
    if name == "block":  # p(left) 0.8 is a left block
        return [t["muted"] if v == 0.5 else sided(0.0 if v > 0.5 else 1.0) for v in levels]
    if name == "contrast":
        left = sorted(-v for v in levels if v < 0)
        right = sorted(v for v in levels if v > 0)
        out = []
        for v in levels:
            if v == 0:
                out.append(t["muted"])
                continue
            mags = left if v < 0 else right
            reach = 0.2 + 0.3 * (mags.index(abs(v)) + 1) / len(mags)  # never near the midpoint
            out.append(sided(0.5 - reach if v < 0 else 0.5 + reach))
        return out
    return [t["categorical"][i % len(t["categorical"])] for i in range(len(levels))]


def build_unit_figure(
    groups: list[TraceGroup], window, baseline: bool, theme: str, title=None
) -> Figure:
    """Raster (top, trials grouped by condition) and PSTH mean ± SEM (bottom), with a
    legend naming each condition and its trial count when split."""
    t = _theme(theme)
    fig = Figure(figsize=(5.6, 4.6))
    add, squeeze = _frame(fig, title, t["ink2"])
    split = len(groups) > 1
    # A split leaves a band between the panels for the legend, off the traces.
    ax_r = add((0.13, 0.56, 0.84, 0.41) if split else (0.13, 0.47, 0.84, 0.50))
    ax_p = add((0.13, 0.11, 0.84, 0.29) if split else (0.13, 0.11, 0.84, 0.31), sharex=ax_r)
    offset = 0
    for g in groups:
        ink = g.colour or t["ink"]
        ax_r.plot(g.rel, g.trial + offset, "|", color=ink, markersize=2.2, markeredgewidth=0.6)
        offset += g.psth.n_trials
        if len(groups) > 1:
            ax_r.axhline(offset - 0.5, color=t["grid"], lw=0.6)
        line = g.colour or t["series"]
        p = g.psth
        ax_p.fill_between(
            p.bin_centers, p.mean - p.sem, p.mean + p.sem, color=line, alpha=0.18, lw=0
        )
        ax_p.plot(p.bin_centers, p.mean, color=line, lw=2, label=f"{g.name} (n = {p.n_trials})")
    ax_r.set_ylim(offset - 0.5, -0.5)
    ax_r.set_ylabel("trial, by condition" if len(groups) > 1 else "trial")
    ax_r.tick_params(labelbottom=False)
    ax_p.set_ylabel("Δ rate (Hz)" if baseline else "rate (Hz)")
    ax_p.set_xlabel("time from event (s)")
    ax_p.set_xlim(*window)
    ax_p.grid(axis="y", color=t["grid"], lw=0.6)
    ax_p.set_axisbelow(True)
    if split:
        fig.legend(
            *ax_p.get_legend_handles_labels(),
            loc="upper left",
            bbox_to_anchor=(0.1, 0.545 * squeeze),
            ncol=min(5, len(groups)),
            fontsize=6.5,
            frameon=False,
            labelcolor=t["ink2"],
            handlelength=1.2,
            columnspacing=0.7,
        )
    for ax in (ax_r, ax_p):
        _style(ax, t)
        ax.axvline(0, color=t["muted"], lw=1, ls=(0, (3, 3)))
    return fig


def unit_figure(groups: list[TraceGroup], window, baseline: bool, theme: str) -> bytes:
    """build_unit_figure as a PNG for the page."""
    return _png(build_unit_figure(groups, window, baseline, theme))


def build_tuning_figure(
    names, means, sems, ns, colours, ordinal: bool, theme: str, title=None
) -> Figure:
    """Mean ± SEM response rate per condition level, n under each level's label."""
    t = _theme(theme)
    fig = Figure(figsize=(5.6, 2.5))
    add, _ = _frame(fig, title, t["ink2"])
    ax = add((0.13, 0.26, 0.84, 0.66))
    x = np.arange(len(names))
    if ordinal:
        ax.plot(x, means, color=t["axis"], lw=1, zorder=1)
    for xi, m, e, c in zip(x, means, sems, colours):
        ax.errorbar(
            xi, m, yerr=0 if np.isnan(e) else e, fmt="o", color=c, ms=6, lw=1.5, capsize=0, zorder=2
        )
    ax.set_xticks(x, [f"{n}\nn = {k}" for n, k in zip(names, ns)])
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylabel("response rate (Hz)")
    ax.grid(axis="y", color=t["grid"], lw=0.6)
    ax.set_axisbelow(True)
    _style(ax, t)
    ax.tick_params(axis="x", labelsize=7)
    return fig


def tuning_figure(names, means, sems, ns, colours, ordinal: bool, theme: str) -> bytes:
    return _png(build_tuning_figure(names, means, sems, ns, colours, ordinal, theme))


def population_figure(scaled, centers, mean, sem, window, theme: str, **stripe):
    """build_population_figure as a PNG for the page, with the heatmap's box."""
    fig, box = build_population_figure(scaled, centers, mean, sem, window, theme, **stripe)
    return _png(fig), box


def build_population_figure(
    scaled: np.ndarray,
    centers: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    window,
    theme: str,
    title=None,
    row_groups: list[str] | None = None,
    group_colours: dict[str, str] | None = None,
) -> tuple[Figure, tuple[float, float, float, float]]:
    """Heatmap of row-scaled PSTHs, (n_units, n_bins) already sorted, and the selection mean.

    row_groups (n_units,) and group_colours draw a stripe beside the rows (each row's
    probe) with a legend naming every colour, so identity is never colour alone.

    Returns the figure and the heatmap's box (left, top, right, bottom) as fractions of
    the image, top-down, so row i spans top + (bottom - top) * [i, i + 1] / n_units.
    """
    t = _theme(theme)
    signed = bool((scaled < 0).any())
    cmap = LinearSegmentedColormap.from_list(
        "studio", t["diverging"] if signed else t["sequential"]
    )
    fig = Figure(figsize=(5.6, 5.6))
    add, squeeze = _frame(fig, title, t["ink2"])
    box = (0.13, 0.40, 0.72, 0.56)  # left, bottom, width, height
    ax_h = add(box)
    cax = add((0.885, 0.40, 0.022, 0.56))
    ax_m = add((0.13, 0.08, 0.72, 0.24), sharex=ax_h)
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
    if row_groups is not None:
        assert len(row_groups) == scaled.shape[0]
        ax_s = add((0.855, 0.40, 0.014, 0.56), sharey=ax_h)
        rgb = np.array([to_rgb(group_colours[g]) for g in row_groups])[:, None, :]
        ax_s.imshow(rgb, aspect="auto", interpolation="nearest", extent=(0, 1, len(rgb), 0))
        ax_s.axis("off")
        shown = [g for g in group_colours if g in set(row_groups)]
        fig.legend(
            handles=[Patch(color=group_colours[g], label=g) for g in shown],
            loc="lower left",
            bbox_to_anchor=(0.13, 0.335 * squeeze),
            ncol=len(shown),
            frameon=False,
            fontsize=7,
            labelcolor=t["ink2"],
            handlelength=1,
            handleheight=0.8,
            columnspacing=1.2,
        )
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
    bottom, height = bottom * squeeze, height * squeeze
    return fig, (left, 1 - bottom - height, left + width, 1 - bottom)
