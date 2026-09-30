"""Movement controls: separating rate changes around an event from the movement after it.

At stimulus onset most units change rate (step 3), but the mouse moves within a few
hundred ms. These tools separate the two:

- **reaction_times:** first movement - stimulus onset, per trial (NaN kept).
- **movement_free:** trials whose first movement comes after a given time, e.g. the
  end of the response window, so a stimulus-onset test on them sees no movement.
- **wheel_speed_psth:** |d wheel position / dt| aligned to events, mean ± SEM across
  trials, on the same bins as the neural PSTH.
- **movement_locking:** is a unit locked to its own trial's first movement?
  - Statistic: the mean over trials of (rate in `post_window` - rate in `pre_window`)
    around each trial's first movement.
  - Null: the same with reaction times permuted within signed-contrast strata (zero
    split by side), so every trial keeps its stimulus and a reaction time drawn from
    its own contrast.
  - Why it discriminates: a stimulus-locked unit blurs the same way under true and
    permuted reaction times, so it is not labelled; a movement-locked unit is sharper
    at its true movement times.
  - p = (1 + #|null| >= |observed|) / (1 + n); BH across the units tested; seeded (R7).
  - Limitation: trials are exchangeable within contrast. If the reaction time tracks
    the unit's state (engagement, say) for reasons other than locking, the label can
    over-call.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from neurodecoder.analysis.conditions import contrast_strata
from neurodecoder.analysis.psth import PSTH, bin_edges
from neurodecoder.analysis.responsiveness import benjamini_hochberg, load_response_config
from neurodecoder.analysis.tuning import stratified_permutations
from neurodecoder.data.session import TimeSeries

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "movement.yaml"
_KEYS = {"pre_window", "post_window", "n_permutations", "seed"}
_NULL = "reaction times permuted within signed contrast"


@dataclass(frozen=True)
class MovementConfig:
    pre_window: tuple[float, float]
    post_window: tuple[float, float]
    n_permutations: int
    seed: int


def load_movement_config(path: str | os.PathLike = DEFAULT_CONFIG) -> MovementConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return MovementConfig(
        tuple(float(t) for t in raw["pre_window"]),
        tuple(float(t) for t in raw["post_window"]),
        int(raw["n_permutations"]),
        int(raw["seed"]),
    )


def _need(trials: pd.DataFrame, *columns: str) -> None:
    missing = [c for c in columns if c not in trials]
    if missing:
        raise ValueError(f"trials have no {', '.join(missing)} column, needed here")


def reaction_times(trials: pd.DataFrame) -> np.ndarray:
    """(n_trials,) first movement - stimulus onset, seconds; NaN where either is missing."""
    _need(trials, "stimOn_times", "firstMovement_times")
    rt = trials["firstMovement_times"].to_numpy(np.float64) - trials["stimOn_times"].to_numpy(
        np.float64
    )
    return rt


def movement_free(trials: pd.DataFrame, until_s: float) -> np.ndarray:
    """(n_trials,) bool: the first movement comes strictly after stimulus onset + until_s.

    A trial without a movement time is not known to be movement-free, so it is False.
    """
    rt = reaction_times(trials)
    return np.isfinite(rt) & (rt > until_s)


def wheel_speed_psth(wheel: TimeSeries, events: np.ndarray, window, bin_width: float) -> PSTH:
    """Mean ± SEM wheel speed (rad/s) per bin around each event, across trials.

    Speed is |d position / dt| at each wheel sample; a bin's value is the mean of the
    samples inside it. Events with no time are excluded and counted.
    """
    t = np.asarray(wheel.timestamps, np.float64)
    position = np.asarray(wheel.data, np.float64)
    assert position.ndim == 1 and t.shape == position.shape
    speed = np.abs(np.gradient(position, t))
    events = np.asarray(events, np.float64)
    valid = events[np.isfinite(events)]
    if valid.size == 0:
        raise ValueError("no trials with a time for this event")
    edges = bin_edges(window, bin_width)
    absolute = valid[:, None] + edges[None, :]  # (n_trials, n_bins + 1)
    if absolute.min() < t[0] or absolute.max() > t[-1]:
        raise ValueError("some event windows fall outside the wheel recording")
    idx = np.searchsorted(t, absolute, side="left")
    csum = np.concatenate([[0.0], np.cumsum(speed)])
    sums = np.diff(csum[idx], axis=1)
    counts = np.diff(idx, axis=1)
    per_trial = np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0)
    n = per_trial.shape[0]
    sem = (
        per_trial.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(per_trial.shape[1], np.nan)
    )
    centers = (edges[:-1] + edges[1:]) / 2
    return PSTH(centers, per_trial.mean(axis=0), sem, n, int(events.size - valid.size))


def _window_counts(spikes: np.ndarray, times: np.ndarray, window) -> np.ndarray:
    """Spike counts in [times + start, times + stop), same shape as times."""
    lo = np.searchsorted(spikes, times + window[0], side="left")
    hi = np.searchsorted(spikes, times + window[1], side="left")
    return hi - lo


def movement_locking(
    spikes, unit_ids, trials: pd.DataFrame, cfg: MovementConfig, alpha: float | None = None
) -> pd.DataFrame:
    """(n_units, ...) statistic_hz, p, q, locked, n_trials, n_null, n_tests, null, seed."""
    _need(trials, "stimOn_times", "firstMovement_times", "contrastLeft", "contrastRight")
    alpha = load_response_config().alpha if alpha is None else alpha
    unit_ids = list(unit_ids)
    stim = trials["stimOn_times"].to_numpy(np.float64)
    rt = reaction_times(trials)
    strata = contrast_strata(trials)
    keep = np.isfinite(stim) & np.isfinite(rt) & np.isfinite(strata)
    stim, rt, strata = stim[keep], rt[keep], strata[keep]
    n = stim.size
    if n < 2:
        raise ValueError("movement locking needs at least 2 trials with a first movement")
    rng = np.random.default_rng(cfg.seed)
    perms = stratified_permutations(np.arange(n), strata, cfg.n_permutations, rng)  # (n_perm, n)
    candidates = stim[:, None] + rt[None, :]  # (n, n): trial i aligned at reaction time j
    pre_len = cfg.pre_window[1] - cfg.pre_window[0]
    post_len = cfg.post_window[1] - cfg.post_window[0]
    rows = np.arange(n)[None, :]
    observed = np.empty(len(unit_ids))
    exceed = np.empty(len(unit_ids), np.int64)
    for k, unit in enumerate(unit_ids):
        s = np.asarray(spikes[unit], np.float64)
        change = (
            _window_counts(s, candidates, cfg.post_window) / post_len
            - _window_counts(s, candidates, cfg.pre_window) / pre_len
        )  # (n, n)
        observed[k] = np.diag(change).mean()
        null = change[rows, perms].mean(axis=1)  # (n_perm,)
        tol = 1e-12 * max(1.0, abs(observed[k]))
        exceed[k] = int((np.abs(null) >= abs(observed[k]) - tol).sum())
    p = (1 + exceed) / (1 + perms.shape[0])
    q = benjamini_hochberg(p)
    return pd.DataFrame(
        {
            "statistic_hz": observed,
            "p": p,
            "q": q,
            "locked": q < alpha,
            "n_trials": n,
            "n_null": perms.shape[0],
            "n_tests": len(unit_ids),
            "null": _NULL,
            "seed": cfg.seed,
        },
        index=pd.Index(unit_ids, name="unit_id"),
    )
