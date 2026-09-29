"""The evaluation contract: every evaluation produces the six-row table.  [CLAUDE.md §5, R4]

`evaluate` is the only entry point that scores a model, and it always fits and scores
all six rows, in this order:

    null_shuffle      the model under test, refit on targets circularly shifted
                      within each session (per-session median over the shifts)
    null_trialstruct  a decoder given task-variable features only (never spikes)
    baseline_ridge    ridge / logistic, same split
    baseline_rrr      multi-session reduced-rank regression, same split
    model             the model under test
    ceiling_within    the model on a within-session split of the test sessions
                      (the model row itself when the split is within-session)

Every fit receives the train partition only; predictions are scored per test
session (evaluation.metrics). The result states plainly whether the model beats
null_trialstruct, null_shuffle and baseline_ridge: a one-sided Wilcoxon signed-rank
test over test sessions on the primary metric (R² or AUROC), alpha 0.05, with the
median difference and win count. Fewer than 5 sessions can never reach alpha, so
they never count as beating.

Data reaches decoders as SessionData from a DataProvider, loaded lazily, one session
at a time; decoders build their own inputs from it (e.g. windows via
preprocess.windows).
"""

import hashlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from neurodecoder.evaluation.metrics import (
    CLASSIFICATION,
    REGRESSION,
    EvalConfig,
    load_eval_config,
    per_session,
    summarise,
)
from neurodecoder.evaluation.nulls import NullConfig, load_null_config
from neurodecoder.splits.guards import assert_split_valid
from neurodecoder.splits.registry import Split

ROWS = (
    "null_shuffle",
    "null_trialstruct",
    "baseline_ridge",
    "baseline_rrr",
    "model",
    "ceiling_within",
)
PRIMARY = {"regression": "r2", "classification": "auroc"}
COMPARED = ("null_trialstruct", "null_shuffle", "baseline_ridge")
MIN_SHIFTS = 5
ALPHA = 0.05
# A one-sided exact Wilcoxon test over n sessions has p >= 2**-n: n < 5 can't reach 0.05.
MIN_SESSIONS = 5
_FAILURE = {
    "null_trialstruct": "this is not decoding",
    "null_shuffle": "no signal beyond the target's own autocorrelation",
    "baseline_ridge": "a more complex model is not justified",
}


@dataclass(frozen=True)
class SessionData:
    """One session's samples in one partition.

    z: (n_units, n_bins) float32 normalised counts (Normalizer.transform), bins
      first_bin + j; None for the trial-structure row, which never sees spikes.
    units: (n_units, ...) unit metadata in z's row order; None with z.
    ends: (n,) int64 absolute end bin of each sample's context_bins-long window.
    y: (n,) targets: values, or 0/1 labels for classification.
    task_features: (n, n_features) null_trialstruct features (evaluation.nulls).
    """

    eid: str
    partition: str
    z: np.ndarray | None
    first_bin: int
    bin_ms: int
    units: pd.DataFrame | None
    ends: np.ndarray
    context_bins: int
    y: np.ndarray
    task_features: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.ends)
        if self.y.shape != (n,) or self.task_features.ndim != 2 or len(self.task_features) != n:
            raise ValueError(f"{self.eid}: ends, y and task_features disagree on n = {n}")
        if self.z is not None and (self.units is None or len(self.units) != self.z.shape[0]):
            raise ValueError(f"{self.eid}: units must list z's rows")

    def without_spikes(self) -> "SessionData":
        return replace(self, z=None, units=None)


class Decoder(Protocol):
    kind: str  # "regression" or "classification"

    def fit(self, train: Sequence[SessionData], *, seed: int) -> None: ...

    def predict(self, data: SessionData) -> np.ndarray:
        """(n,) predictions: values, or P(class 1) for classification."""


class DataProvider(Protocol):
    split: Split
    kind: str
    context_bins: int

    def data(self, eid: str, partition: str, *, shift: int | None = None) -> SessionData:
        """shift: rotate the session's target by this many samples (null_shuffle)."""

    def shifts(self, eid: str, n_shifts: int, *, seed: int) -> np.ndarray:
        """Valid shifts for the session (evaluation.nulls.draw_shifts); may be empty."""


class _Sessions(Sequence):
    """The train sessions, loaded one at a time when a decoder reads them."""

    def __init__(self, provider, eids, shift_of, strip):
        self._provider, self._shift_of, self._strip = provider, shift_of, strip
        self._eids = [e for e in eids if shift_of(e) is not _SKIP]

    def __len__(self) -> int:
        return len(self._eids)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        eid = self._eids[i]
        data = self._provider.data(eid, "train", shift=self._shift_of(eid))
        return data.without_spikes() if self._strip else data

    def __iter__(self) -> Iterator[SessionData]:
        return (self[i] for i in range(len(self)))


_SKIP = object()  # a session with no shift at this draw sits the draw out


@dataclass(frozen=True)
class Verdict:
    row: str
    beats: bool
    p_value: float
    median_difference: float
    wins: int
    n_sessions: int

    def line(self, metric: str) -> str:
        if self.n_sessions == 0:
            return f"model vs {self.row}: undefined, no test session has a defined {self.row}."
        detail = (
            f"median Δ{metric} {self.median_difference:+.3f}, wins {self.wins}/{self.n_sessions} "
            f"sessions, Wilcoxon p = {self.p_value:.3g}"
        )
        if self.n_sessions < MIN_SESSIONS:
            detail = f"too few sessions ({self.n_sessions}) for the test; {detail}"
        if self.beats:
            return f"model beats {self.row} ({detail})."
        return f"model does NOT beat {self.row} ({detail}): {_FAILURE[self.row]}."


@dataclass(frozen=True)
class ContractResult:
    kind: str
    primary: str
    per_session: dict  # row -> (n_test_sessions, metrics) DataFrame, index eid
    shuffle: pd.DataFrame  # (n_test_sessions, n_shifts) primary metric per shift draw
    verdicts: tuple
    seed: int
    n_shifts: int
    split_hash: str
    ceiling_split_hash: str | None
    ceiling_is_model: bool

    def summary(self) -> pd.DataFrame:
        """Rows in contract order: the primary metric's distribution over test sessions,
        then the median of every other metric."""
        out = {}
        for row in ROWS:
            s = summarise(self.per_session[row])
            first = s.loc[self.primary, ["median", "q25", "q75", "n_sessions"]]
            others = s.loc[[m for m in s.index if m != self.primary], "median"]
            out[row] = {**first.to_dict(), **{f"median_{m}": v for m, v in others.items()}}
        table = pd.DataFrame.from_dict(out, orient="index")
        return table.astype({"n_sessions": int})

    def report(self) -> str:
        table = self.summary()
        lines = [
            f"Evaluation contract: {self.kind}, primary metric {self.primary} over test "
            f"sessions (split {self.split_hash[:12]}, seed {self.seed}, {self.n_shifts} shifts)",
            table.to_string(float_format=lambda v: f"{v:.3f}"),
        ]
        if self.ceiling_is_model:
            lines.append("ceiling_within = model: the split is itself within-session.")
        lines += [v.line(self.primary) for v in self.verdicts]
        return "\n".join(lines)

    __str__ = report


def _session_seed(seed: int, eid: str) -> int:
    return int(hashlib.sha256(f"{seed}:{eid}".encode()).hexdigest()[:8], 16)


def _check_predictions(data: SessionData, prediction, kind: str) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (len(data.ends),) or not np.all(np.isfinite(prediction)):
        raise ValueError(
            f"{data.eid}: predictions must be ({len(data.ends)},) finite, got {prediction.shape}"
        )
    if kind == "classification" and np.any((prediction < 0) | (prediction > 1)):
        raise ValueError(f"{data.eid}: classification predictions must be probabilities")
    return prediction


def _undefined(kind: str, eids) -> pd.DataFrame:
    metrics = CLASSIFICATION if kind == "classification" else REGRESSION
    table = pd.DataFrame(np.nan, index=pd.Index(list(eids), name="eid"), columns=list(metrics))
    return table.assign(n_samples=0)


def _run(factory, provider, kind, eval_config, seed, shift_of=None, strip=False) -> pd.DataFrame:
    shift_of = shift_of or (lambda eid: None)
    decoder = factory()
    if decoder.kind != kind:
        raise ValueError(f"a {decoder.kind} decoder cannot fit a {kind} target")
    split = provider.split
    train = _Sessions(provider, split.partitions["train"], shift_of, strip)
    tested = [e for e in split.partitions["test"] if shift_of(e) is not _SKIP]
    if not len(train) or not tested:
        # No session has this shift draw (e.g. all too short): the row is undefined.
        return _undefined(kind, split.partitions["test"])
    decoder.fit(train, seed=seed)
    ys, predictions, eids = [], [], []
    for eid in split.partitions["test"]:
        shift = shift_of(eid)
        if shift is _SKIP:
            continue
        data = provider.data(eid, "test", shift=shift)
        if strip:
            data = data.without_spikes()
        predictions.append(_check_predictions(data, decoder.predict(data), kind))
        ys.append(np.asarray(data.y, dtype=np.float64))
        eids.append(np.full(len(data.ends), eid, dtype=object))
    table = per_session(
        kind, np.concatenate(ys), np.concatenate(predictions), np.concatenate(eids), eval_config
    )
    return table.reindex(split.partitions["test"])


def _verdict(model: pd.Series, other: pd.Series, row: str) -> Verdict:
    diff = (model - other.reindex(model.index)).dropna()
    n, wins = len(diff), int((diff > 0).sum())
    median = float(diff.median()) if n else np.nan
    p = np.nan
    if n and np.any(diff != 0):
        p = float(wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue)
    beats = n >= MIN_SESSIONS and np.isfinite(p) and p < ALPHA and median > 0
    return Verdict(row, bool(beats), p, median, wins, n)


def evaluate(
    task: DataProvider,
    *,
    model: Callable[[], Decoder],
    baseline_ridge: Callable[[], Decoder],
    baseline_rrr: Callable[[], Decoder],
    trialstruct: Callable[[], Decoder],
    ceiling: DataProvider | None,
    seed: int,
    n_shifts: int | None = None,
    eval_config: EvalConfig | None = None,
    null_config: NullConfig | None = None,
) -> ContractResult:
    """Fit and score all six rows; decoders are factories, called once per fit.

    ceiling: a within-session provider over the task's test sessions, or None when the
    task's split is itself within-session. n_shifts: shift draws for null_shuffle
    (default configs/nulls.yaml), at least MIN_SHIFTS.
    """
    eval_config = eval_config or load_eval_config()
    null_config = null_config or load_null_config()
    n_shifts = null_config.n_shifts if n_shifts is None else n_shifts
    if n_shifts < MIN_SHIFTS:
        raise ValueError(f"null_shuffle needs at least {MIN_SHIFTS} shifts, got {n_shifts}")
    kind = task.kind
    if kind not in PRIMARY:
        raise ValueError(f"unknown target kind {kind!r}")

    assert_split_valid(task.split, context_bins=task.context_bins)
    ceiling_is_model = task.split.kind == "within_session"
    if ceiling_is_model:
        ceiling = None
    elif ceiling is None:
        raise ValueError("a cross-session split needs a within-session ceiling provider")
    else:
        if ceiling.split.kind != "within_session":
            raise ValueError("the ceiling provider's split must be within_session")
        if set(ceiling.split.partitions["test"]) != set(task.split.partitions["test"]):
            raise ValueError("the ceiling must cover exactly the task's test sessions")
        assert_split_valid(ceiling.split, context_bins=ceiling.context_bins)

    per: dict[str, pd.DataFrame] = {}
    per["model"] = _run(model, task, kind, eval_config, seed)
    per["baseline_ridge"] = _run(baseline_ridge, task, kind, eval_config, seed)
    per["baseline_rrr"] = _run(baseline_rrr, task, kind, eval_config, seed)
    per["null_trialstruct"] = _run(trialstruct, task, kind, eval_config, seed, strip=True)

    split = task.split
    eids = sorted(set(split.partitions["train"]) | set(split.partitions["test"]))
    draws = {e: np.asarray(task.shifts(e, n_shifts, seed=_session_seed(seed, e))) for e in eids}
    tables = []
    for k in range(n_shifts):

        def shift_of(eid, k=k):
            return int(draws[eid][k]) if k < len(draws[eid]) else _SKIP

        tables.append(_run(model, task, kind, eval_config, seed, shift_of=shift_of))
    stacked = pd.concat(tables, keys=range(n_shifts), names=["shift", "eid"])
    per["null_shuffle"] = stacked.groupby(level="eid").median().reindex(split.partitions["test"])
    shuffle = pd.concat([t[PRIMARY[kind]] for t in tables], axis=1, keys=range(n_shifts))

    per["ceiling_within"] = (
        per["model"] if ceiling_is_model else _run(model, ceiling, kind, eval_config, seed)
    )
    primary = PRIMARY[kind]
    verdicts = tuple(_verdict(per["model"][primary], per[row][primary], row) for row in COMPARED)
    return ContractResult(
        kind=kind,
        primary=primary,
        per_session={row: per[row] for row in ROWS},
        shuffle=shuffle,
        verdicts=verdicts,
        seed=seed,
        n_shifts=n_shifts,
        split_hash=split.hash,
        ceiling_split_hash=None if ceiling is None else ceiling.split.hash,
        ceiling_is_model=ceiling_is_model,
    )
