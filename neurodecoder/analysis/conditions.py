"""Task conditions to split trials by, read from a session's trials table.

IBL's conventions, checked on d23a44ef (docs/DECISIONS.md):
- **Contrast:** each trial sets exactly one of contrastLeft / contrastRight; the side
  without a stimulus is NaN. Zero contrast is 0 on the stimulus side. Signed contrast
  is contrastRight - contrastLeft with NaN read as 0: right positive, left negative,
  and 0% for both zero-contrast sides. A trial with both or neither side set breaks
  the convention and is excluded and counted, never guessed.
- **Choice:** -1 or +1, and 0 for no-go, which is excluded and counted. On correct
  trials a right stimulus has choice -1 and a left one +1, so the levels are named
  "right (-1)" and "left (+1)".
- **Outcome:** feedbackType, -1 error, +1 reward.
- **Block:** probabilityLeft, 0.2 / 0.5 / 0.8 (0.5 is the unbiased opening block).

Only conditions whose columns the table has are offered, so Phy sessions get what
their events CSV provides.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neurodecoder.analysis.events import trial_event_times

# name -> (label, trials columns it needs)
CONDITIONS = {
    "side": ("Stimulus side", ("contrastLeft", "contrastRight")),
    "contrast": ("Signed contrast", ("contrastLeft", "contrastRight")),
    "choice": ("Choice", ("choice",)),
    "outcome": ("Outcome", ("feedbackType",)),
    "block": ("Block", ("probabilityLeft",)),
}
_NAMES = {
    "side": {-1.0: "left", 1.0: "right"},
    "choice": {-1.0: "right (-1)", 1.0: "left (+1)"},
    "outcome": {-1.0: "error", 1.0: "reward"},
}


@dataclass(frozen=True)
class Condition:
    """values: (n_trials,) each trial's level, NaN where excluded (reason in `excluded`)."""

    name: str
    values: np.ndarray
    levels: tuple[float, ...]
    names: tuple[str, ...]
    n_excluded: int
    excluded: str


@dataclass(frozen=True)
class Part:
    """One level's event times, (n_level_trials,), NaN kept where the event is missing."""

    level: float
    name: str
    times: np.ndarray


def available_conditions(trials: pd.DataFrame) -> dict[str, str]:
    """name -> label, for the conditions this trials table has the columns for."""
    return {
        name: label
        for name, (label, columns) in CONDITIONS.items()
        if all(c in trials for c in columns)
    }


def _need(trials: pd.DataFrame, name: str) -> None:
    if name not in CONDITIONS:
        raise ValueError(f"unknown condition {name!r}; available: {sorted(CONDITIONS)}")
    missing = [c for c in CONDITIONS[name][1] if c not in trials]
    if missing:
        raise ValueError(f"trials have no {missing} column(s), needed for {name}")


def _one_side_set(trials: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = trials["contrastLeft"].to_numpy(np.float64)
    right = trials["contrastRight"].to_numpy(np.float64)
    valid = np.isnan(left) != np.isnan(right)
    return left, right, valid


def signed_contrast(trials: pd.DataFrame) -> np.ndarray:
    """(n_trials,) contrastRight - contrastLeft, NaN read as 0; NaN if not one side set."""
    _need(trials, "contrast")
    left, right, valid = _one_side_set(trials)
    out = np.nan_to_num(right) - np.nan_to_num(left) + 0.0  # + 0.0 turns -0.0 into 0.0
    out[~valid] = np.nan
    return out


def stimulus_side(trials: pd.DataFrame) -> np.ndarray:
    """(n_trials,) +1 right, -1 left; NaN if not exactly one side set."""
    _need(trials, "side")
    _, right, valid = _one_side_set(trials)
    return np.where(valid, np.where(np.isnan(right), -1.0, 1.0), np.nan)


def contrast_strata(trials: pd.DataFrame) -> np.ndarray:
    """(n_trials,) signed contrast with zero split by side (-0% and +0% differ).

    For stratified nulls: at 0% contrast the rewarded side still differs, and choice
    still tracks it, so merging the two zeros would leave choice confounded with side.
    """
    return signed_contrast(trials) + 1e-9 * stimulus_side(trials)


def _percent(value: float) -> str:
    text = f"{abs(value) * 100:g}%"
    return text if value == 0 else ("+" if value > 0 else "-") + text


def condition(trials: pd.DataFrame, name: str) -> Condition:
    _need(trials, name)
    excluded = ""
    if name == "side":
        values = stimulus_side(trials)
        excluded = "trials without exactly one of contrastLeft / contrastRight"
    elif name == "contrast":
        values = signed_contrast(trials)
        excluded = "trials without exactly one of contrastLeft / contrastRight"
    elif name == "choice":
        values = trials["choice"].to_numpy(np.float64).copy()
        values[values == 0] = np.nan
        excluded = "no-go trials (choice 0)"
    elif name == "outcome":
        values = trials["feedbackType"].to_numpy(np.float64).copy()
        excluded = "trials without feedback"
    else:
        values = trials["probabilityLeft"].to_numpy(np.float64).copy()
        excluded = "trials without a block"
    levels = tuple(float(v) for v in np.unique(values[np.isfinite(values)]))
    if name == "contrast":
        names = tuple(_percent(v) for v in levels)
    elif name == "block":
        names = tuple(f"p(left) {v:g}" for v in levels)
    else:
        names = tuple(_NAMES[name].get(v, f"{v:g}") for v in levels)
    n_excluded = int(np.isnan(values).sum())
    assert values.shape == (len(trials),)
    return Condition(name, values, levels, names, n_excluded, excluded if n_excluded else "")


def split_event_times(trials: pd.DataFrame, event: str, cond: Condition) -> list[Part]:
    """One Part per level: the event time of every trial at that level."""
    times = trial_event_times(trials, event)
    return [
        Part(level, name, times[cond.values == level])
        for level, name in zip(cond.levels, cond.names)
    ]
