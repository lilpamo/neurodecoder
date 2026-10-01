"""Tuning curves and selectivity between two conditions, each unit against its own null.

Response rate: spikes in a window after the event / its length, one value per trial.

Tuning curve: the mean ± SEM (across trials) response rate at each level of a
condition, with n per level.

Selectivity: AUROC = P(rate at level b > rate at level a), ties counted half; 0.5 means
no difference. It is tested two-sided against a null, with p = (1 + #null with
|AUROC - 0.5| >= observed) / (1 + n_null). Benjamini-Hochberg runs across the units
tested together, and n_tests is reported. Nulls (R4), by condition:

- **choice:** labels permuted within signed-contrast strata, zero split by side.
  Choice follows the stimulus on correct trials, so a plain shuffle would call a
  stimulus-driven unit choice-selective.
- **side:** labels permuted within choice strata, the mirror case: a choice-driven
  unit is not stimulus-selective.
- **outcome:** labels permuted within signed-contrast strata. Errors concentrate at
  low contrast, so a contrast-driven unit is not outcome-selective.
- **block:** pseudo-sessions from IBL's block generator (evaluation/nulls.py). Block
  labels are autocorrelated, so any label permutation is invalid for them. Block uses
  the pre-event (baseline) window: the block prior predicts the stimulus side, so a
  post-stimulus window would count stimulus responses as block selectivity.

Signed contrast has many levels: it gets a tuning curve, not a two-level AUROC.
Seeds and counts come from configs/selectivity.yaml (R7). Trials are treated as
exchangeable within strata; slow drift in a unit's rate is only accounted for by the
block null.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from unitwave.analysis.conditions import (
    condition,
    contrast_strata,
    split_event_times,
)
from unitwave.analysis.events import trial_event_times
from unitwave.analysis.psth import trial_counts
from unitwave.analysis.responsiveness import ResponseConfig, benjamini_hochberg
from unitwave.evaluation.nulls import generate_pseudo_blocks

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "selectivity.yaml"
_KEYS = {"n_permutations", "n_pseudo_sessions", "seed", "min_trials"}
# condition -> (level a, level b): AUROC > 0.5 means a higher rate at level b.
COMPARISONS = {
    "side": (-1.0, 1.0),
    "choice": (-1.0, 1.0),
    "outcome": (-1.0, 1.0),
    "block": (0.2, 0.8),
}
_STRATA = {
    "choice": "choice permuted within signed contrast",
    "side": "side permuted within choice",
    "outcome": "outcome permuted within signed contrast",
}
_PSEUDO = "pseudo-sessions from IBL's block generator"


@dataclass(frozen=True)
class SelectivityConfig:
    n_permutations: int
    n_pseudo_sessions: int
    seed: int
    min_trials: int


def load_selectivity_config(path: str | os.PathLike = DEFAULT_CONFIG) -> SelectivityConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return SelectivityConfig(**{k: int(raw[k]) for k in sorted(_KEYS)})


def window_rates(spikes: np.ndarray, events: np.ndarray, window) -> np.ndarray:
    """(n_trials,) Hz: spikes in [event + start, event + stop) / window length; NaN events
    stay NaN."""
    events = np.asarray(events, np.float64)
    assert events.ndim == 1
    width = float(window[1]) - float(window[0])
    out = np.full(events.shape, np.nan)
    valid = np.isfinite(events)
    if valid.any():
        out[valid] = trial_counts(spikes, events[valid], window, width)[:, 0] / width
    return out


def tuning_curve(spikes, trials: pd.DataFrame, event: str, name: str, window) -> pd.DataFrame:
    """(n_levels, 4): level, mean_hz, sem_hz, n per level of the condition, indexed by name.

    SEM is across trials (ddof=1), NaN with one trial.
    """
    rows = []
    for part in split_event_times(trials, event, condition(trials, name)):
        rates = window_rates(spikes, part.times, window)
        rates = rates[np.isfinite(rates)]
        n = rates.size
        sem = rates.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append((part.name, part.level, rates.mean() if n else np.nan, sem, n))
    table = pd.DataFrame(rows, columns=["name", "level", "mean_hz", "sem_hz", "n"])
    return table.set_index("name")


def auroc(rates_a: np.ndarray, rates_b: np.ndarray) -> float:
    """P(rate from b > rate from a), ties counted half."""
    ranks = rankdata(np.concatenate([rates_a, rates_b]))
    n_a, n_b = len(rates_a), len(rates_b)
    return float((ranks[n_a:].sum() - n_b * (n_b + 1) / 2) / (n_a * n_b))


def stratified_permutations(labels: np.ndarray, strata: np.ndarray, n: int, rng) -> np.ndarray:
    """(n, n_trials): labels shuffled independently within each stratum, n times."""
    labels, strata = np.asarray(labels), np.asarray(strata)
    assert labels.shape == strata.shape and labels.ndim == 1
    out = np.empty((n, labels.size), dtype=labels.dtype)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        order = rng.random((n, idx.size)).argsort(axis=1)
        out[:, idx] = labels[idx][order]
    return out


@lru_cache(maxsize=8)
def _pseudo_blocks(n_trials: int, n: int, seed: int) -> np.ndarray:
    """(n, n_trials) pseudo-session probabilityLeft sequences, seeds seed + 1 .. seed + n."""
    return np.vstack([generate_pseudo_blocks(n_trials, seed=seed + 1 + i) for i in range(n)])


def _strata(trials: pd.DataFrame, name: str) -> np.ndarray:
    if name == "side":
        if "choice" not in trials:
            raise ValueError("side selectivity needs the choice column to stratify by")
        values = trials["choice"].to_numpy(np.float64).copy()
        values[values == 0] = np.nan
        return values
    try:
        return contrast_strata(trials)
    except ValueError:
        raise ValueError(
            f"{name} selectivity needs contrastLeft and contrastRight to stratify by"
        ) from None


def selectivity(
    spikes,
    unit_ids,
    trials: pd.DataFrame,
    event: str,
    name: str,
    windows: ResponseConfig,
    cfg: SelectivityConfig,
    stratify: bool = True,
    trial_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """(n_units, ...) auroc, p, q, selective, n_a, n_b, n_null, n_tests, null, window.

    trial_mask: (n_trials,) trials to use (a trial filter). The full table is still
    passed, so block pseudo-sessions are generated over the whole session and then
    masked, keeping IBL's block structure intact.
    stratify=False replaces the null with a plain label permutation: only to show why
    the stratified and pseudo-session nulls are needed.
    """
    if name == "contrast":
        raise ValueError("signed contrast has many levels: use its tuning curve")
    if name not in COMPARISONS:
        raise ValueError(f"no selectivity test for {name!r}; available: {sorted(COMPARISONS)}")
    unit_ids = list(unit_ids)
    level_a, level_b = COMPARISONS[name]
    values = condition(trials, name).values
    events = trial_event_times(trials, event)
    keep = np.isfinite(events) & np.isin(values, (level_a, level_b))
    if trial_mask is not None:
        trial_mask = np.asarray(trial_mask, bool)
        assert trial_mask.shape == keep.shape, "trial_mask must be (n_trials,)"
        keep &= trial_mask
    strata = None
    if stratify and name != "block":
        strata = _strata(trials, name)
        keep &= np.isfinite(strata)
    y = (values[keep] == level_b).astype(np.int64)  # (n_keep,)
    n_b, n_a = int(y.sum()), int(y.size - y.sum())
    if min(n_a, n_b) < cfg.min_trials:
        raise ValueError(
            f"{name}: {n_a} and {n_b} trials; each condition needs at least {cfg.min_trials}"
        )

    window = windows.baseline_window if name == "block" else windows.response_window
    rates = np.vstack(
        [window_rates(spikes[u], events[keep], window) for u in unit_ids]
    )  # (n_units, n_keep)
    ranks = rankdata(rates, axis=1)
    rng = np.random.default_rng(cfg.seed)
    if name == "block" and stratify:
        pseudo = _pseudo_blocks(len(trials), cfg.n_pseudo_sessions, cfg.seed)[:, keep]
        if not np.isin(pseudo, (0.2, 0.8)).all():
            raise ValueError(
                "block: some tested trials fall in the pseudo-sessions' unbiased opening "
                "block; this session's block structure differs from IBL's generator"
            )
        labels, null_name = (pseudo == 0.8).astype(np.int64), _PSEUDO
    elif strata is not None:
        labels = stratified_permutations(y, strata[keep], cfg.n_permutations, rng)
        null_name = _STRATA[name]
    else:
        labels = stratified_permutations(y, np.zeros_like(y), cfg.n_permutations, rng)
        null_name = f"{name} permuted without strata"

    # AUROC from rank sums: ranks don't depend on labels, so one product gives every draw.
    nb = labels.sum(axis=1)  # (n_null,)
    na = labels.shape[1] - nb
    ok = (na > 0) & (nb > 0)
    labels, nb, na = labels[ok], nb[ok], na[ok]
    observed = (ranks @ y - n_b * (n_b + 1) / 2) / (n_a * n_b)  # (n_units,)
    null = (ranks @ labels.T - nb * (nb + 1) / 2) / (na * nb)  # (n_units, n_null)
    tol = 1e-12
    extreme = np.abs(null - 0.5) >= (np.abs(observed - 0.5) - tol)[:, None]
    p = (1 + extreme.sum(axis=1)) / (1 + null.shape[1])
    q = benjamini_hochberg(p)
    return pd.DataFrame(
        {
            "auroc": observed,
            "p": p,
            "q": q,
            "selective": q < windows.alpha,
            "n_a": n_a,
            "n_b": n_b,
            "n_null": int(null.shape[1]),
            "n_tests": len(unit_ids),
            "null": null_name,
            "window": "baseline" if name == "block" else "response",
            "seed": cfg.seed,
        },
        index=pd.Index(unit_ids, name="unit_id"),
    )
