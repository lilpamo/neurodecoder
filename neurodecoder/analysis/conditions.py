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

import json
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


# ---------- trial filters: which trials an analysis uses ----------

# name -> (label, trials columns it needs)
TRIAL_FILTERS = {
    "bwm_include": ("BWM trial inclusion", ("bwm_include",)),
    "exclude_nogo": ("Exclude no-go", ("choice",)),
    "contrasts": ("Contrasts", ("contrastLeft", "contrastRight")),
    "blocks": ("Blocks", ("probabilityLeft",)),
    "outcomes": ("Outcomes", ("feedbackType",)),
}


@dataclass(frozen=True)
class TrialFilter:
    """Which trials to keep. Empty tuples and False keep everything.

    bwm_include: keep only the release's bwm_include trials. On d23a44ef that is exactly
    a reaction time (first movement - stimulus onset) within [0.08, 2] s.
    contrasts: absolute contrasts to keep (0 is 0%); blocks: probabilityLeft values;
    outcomes: feedbackType values (-1 error, 1 reward).
    """

    bwm_include: bool = False
    exclude_nogo: bool = False
    contrasts: tuple[float, ...] = ()
    blocks: tuple[float, ...] = ()
    outcomes: tuple[float, ...] = ()

    def normalised(self) -> "TrialFilter":
        def values(v):
            return tuple(sorted({float(x) for x in v}))

        return TrialFilter(
            bool(self.bwm_include),
            bool(self.exclude_nogo),
            values(self.contrasts),
            values(self.blocks),
            values(self.outcomes),
        )

    def to_dict(self) -> dict:
        n = self.normalised()
        return {
            "bwm_include": n.bwm_include,
            "exclude_nogo": n.exclude_nogo,
            "contrasts": list(n.contrasts),
            "blocks": list(n.blocks),
            "outcomes": list(n.outcomes),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "TrialFilter":
        unknown = sorted(set(raw) - set(TRIAL_FILTERS))
        if unknown:
            raise ValueError(f"unknown trial filters {unknown}; known: {sorted(TRIAL_FILTERS)}")
        return cls(**raw).normalised()

    def key(self) -> str:
        """One canonical string per filter, for caches and records."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def active(self) -> list[str]:
        n = self.normalised()
        return [name for name in TRIAL_FILTERS if getattr(n, name)]


@dataclass(frozen=True)
class TrialSelection:
    """mask: (n_trials,) kept trials. excluded: reason -> trials failing it (a trial
    failing several filters is counted under each, and once in n_excluded)."""

    mask: np.ndarray
    n_total: int
    n_kept: int
    n_excluded: int
    excluded: dict[str, int]


def available_trial_filters(trials: pd.DataFrame) -> dict[str, dict]:
    """name -> {label, available, reason}: offered only when the table has the columns."""
    out = {}
    for name, (label, columns) in TRIAL_FILTERS.items():
        missing = [c for c in columns if c not in trials]
        out[name] = {
            "label": label,
            "available": not missing,
            "reason": f"trials have no {', '.join(missing)} column" if missing else "",
        }
    return out


def apply_trial_filter(trials: pd.DataFrame, f: TrialFilter) -> TrialSelection:
    """Which trials the filter keeps, with every exclusion counted by reason."""
    f = f.normalised()
    offered = available_trial_filters(trials)
    for name in f.active():
        if not offered[name]["available"]:
            raise ValueError(f"cannot filter by {name}: {offered[name]['reason']}")
    n = len(trials)
    reasons: dict[str, np.ndarray] = {}
    if f.bwm_include:
        reasons["not bwm_include"] = ~trials["bwm_include"].fillna(False).to_numpy(bool)
    if f.exclude_nogo:
        reasons["no-go"] = trials["choice"].to_numpy(np.float64) == 0
    if f.contrasts:
        level = np.abs(signed_contrast(trials))
        undefined = np.isnan(level)
        reasons["contrast undefined"] = undefined
        reasons["contrast not selected"] = ~undefined & ~np.isin(level, f.contrasts)
    if f.blocks:
        reasons["block not selected"] = ~np.isin(
            trials["probabilityLeft"].to_numpy(np.float64), f.blocks
        )
    if f.outcomes:
        reasons["outcome not selected"] = ~np.isin(
            trials["feedbackType"].to_numpy(np.float64), f.outcomes
        )
    dropped = np.zeros(n, bool)
    for failed in reasons.values():
        dropped |= failed
    excluded = {r: int(v.sum()) for r, v in reasons.items() if v.any()}
    return TrialSelection(~dropped, n, int((~dropped).sum()), int(dropped.sum()), excluded)
