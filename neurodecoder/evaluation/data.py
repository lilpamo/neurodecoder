"""Real sessions as contract inputs: the DataProvider for one split and one target.

SplitData prepares every session of the split once: BWM session -> unit QC and bins
(preprocess_session) -> the target (targets/) -> a Normalizer fit on the split's
training data. It keeps the binned counts and targets, not the spike trains, and
normalises on each request. Every session is prepared up front, which suits the
within-session splits of Phase 3; cross-session runs over hundreds of sessions will
need a lazier provider (Phase 4).

Samples for one session and partition (contract.SessionData):
- per-bin targets (wheel_velocity, movement_state): every window end in the
  partition's span (preprocess.windows) whose bin lies in the task period and whose
  target is defined; with train_stride k > 1, training keeps every k-th bin only.
- trial targets (choice, block): the target's trials (bwm_include, label defined)
  that the split lists in the partition, at the target's own window. A trial whose
  window reaches outside the partition's span is dropped and counted in `dropped`.
With a shift (null_shuffle), the target is rotated first (evaluation.nulls) and the
same ends are sampled; a bin whose rotated target is undefined is dropped.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from neurodecoder.data.load import load_data_config, load_session
from neurodecoder.data.session import Session
from neurodecoder.evaluation.contract import SessionData
from neurodecoder.evaluation.nulls import (
    NullConfig,
    bin_trialstruct_features,
    draw_shifts,
    load_null_config,
    shift_bin_target,
    shift_trial_target,
    task_bins,
    trial_trialstruct_features,
)
from neurodecoder.preprocess.binning import (
    BinnedSpikes,
    PreprocConfig,
    load_preproc_config,
    preprocess_session,
)
from neurodecoder.preprocess.normalize import fit_normalizer
from neurodecoder.preprocess.windows import window_plan
from neurodecoder.qc.units import apply_unit_qc
from neurodecoder.splits.registry import Split
from neurodecoder.targets import bins as bin_targets
from neurodecoder.targets import trials as trial_targets
from neurodecoder.targets.config import TargetConfig, load_target_config

KINDS = {
    "wheel_velocity": "regression",
    "movement_state": "classification",
    "choice": "classification",
    "block": "classification",
}
PER_BIN = ("wheel_velocity", "movement_state")


@dataclass(frozen=True)
class _Trials:
    """A session's id and trials: all the null builders read (not its spike trains)."""

    eid: str
    trials: pd.DataFrame


@dataclass(frozen=True)
class _Prepared:
    trials: _Trials
    units: pd.DataFrame
    binned: BinnedSpikes
    target: object  # targets.bins.BinTarget or targets.trials.TrialTarget


class SplitData:
    """contract.DataProvider over real sessions, for one split and one target."""

    def __init__(
        self,
        split: Split,
        target: str,
        *,
        context_bins: int | None = None,
        train_stride: int = 1,
        load: Callable[[str], Session] | None = None,
        behaviour_root=None,
        preproc: PreprocConfig | None = None,
        targets: TargetConfig | None = None,
        nulls: NullConfig | None = None,
    ):
        if target not in KINDS:
            raise ValueError(f"unknown target {target!r}; one of {sorted(KINDS)}")
        if isinstance(train_stride, bool) or not isinstance(train_stride, int) or train_stride < 1:
            raise ValueError(f"train_stride must be a positive integer, got {train_stride!r}")
        self.split, self.target, self.kind = split, target, KINDS[target]
        self.train_stride = train_stride
        self.preproc = preproc or load_preproc_config()
        self.targets = targets or load_target_config()
        self.nulls = nulls or load_null_config()
        self.context_bins = self._context(context_bins)
        self.plan = window_plan(split, context_bins=self.context_bins, stride_bins=1)
        if target == "movement_state" and behaviour_root is None:
            behaviour_root = load_data_config().bwm_behavior_root
        self._behaviour_root = behaviour_root
        load = load or (lambda eid: load_session(eid, "bwm"))

        self._prepared: dict[str, _Prepared] = {}
        for eid in sorted(set().union(*split.partitions.values())):
            session = load(eid)
            binned = preprocess_session(session, self.preproc)
            units = apply_unit_qc(session, self.preproc.qc).units.loc[list(binned.unit_ids)]
            self._prepared[eid] = _Prepared(
                _Trials(eid, session.trials), units, binned, self._build_target(session, binned)
            )
        self.normalizer = fit_normalizer(
            split, {e: p.binned for e, p in self._prepared.items()}, self.preproc
        )
        self.dropped: dict[tuple[str, str], int] = {}

    def _context(self, context_bins: int | None) -> int:
        if self.target in PER_BIN:
            if context_bins is None:
                raise ValueError(f"{self.target} needs context_bins (the model's window)")
            return context_bins
        window = self.targets.windows[self.target].n_bins(self.preproc.bin_ms)
        if context_bins is not None and context_bins != window:
            raise ValueError(f"{self.target}'s window is {window} bins, not {context_bins}")
        return window

    def _build_target(self, session: Session, binned: BinnedSpikes):
        if self.target == "wheel_velocity":
            return bin_targets.wheel_velocity(session, binned, self.targets)
        if self.target == "movement_state":
            return bin_targets.movement_state(session, binned, self._behaviour_root, self.targets)
        return getattr(trial_targets, self.target)(session, binned, self.targets)

    def data(self, eid: str, partition: str, *, shift: int | None = None) -> SessionData:
        if eid not in self.split.partitions.get(partition, ()):
            raise ValueError(f"{eid} is not in {partition}")
        p = self._prepared[eid]
        if self.target in PER_BIN:
            ends, y, features = self._bin_samples(p, partition, shift)
        else:
            ends, y, features = self._trial_samples(p, partition, shift)
        return SessionData(
            eid=eid,
            partition=partition,
            z=self.normalizer.transform(p.binned),
            first_bin=p.binned.first_bin,
            bin_ms=p.binned.bin_ms,
            units=p.units,
            ends=ends,
            context_bins=self.context_bins,
            y=y,
            task_features=features,
        )

    def _bin_samples(self, p: _Prepared, partition: str, shift: int | None):
        b = p.binned
        ends = self.plan.ends(b, partition)
        rel = ends - b.first_bin
        task = task_bins(p.trials, b.first_bin, b.n_bins, b.bin_ms)
        keep = (rel >= task.start) & (rel < task.stop)
        if partition == "train" and self.train_stride > 1:
            keep &= ends % self.train_stride == 0
        target = p.target if shift is None else shift_bin_target(p.target, p.trials, shift)
        y = target.values[rel].astype(np.float64)
        keep &= np.isfinite(y)
        features, _ = bin_trialstruct_features(p.trials, b, self.nulls)
        return ends[keep], y[keep], features[rel[keep]]

    def _trial_samples(self, p: _Prepared, partition: str, shift: int | None):
        target = p.target if shift is None else shift_trial_target(p.target, shift)
        table = target.table
        if self.split.kind == "within_session":
            listed = self.split.sessions[p.trials.eid]["trials"][partition]
            table = table[table["trial"].isin(listed)]
        first, last = self.plan.span(p.binned, partition)
        inside = (table["end_bin"] - self.context_bins + 1 >= first) & (table["end_bin"] <= last)
        self.dropped[(p.trials.eid, partition)] = int((~inside).sum())
        table = table[inside].reset_index(drop=True)
        features, _ = trial_trialstruct_features(p.trials, replace(target, table=table), self.nulls)
        ends = table["end_bin"].to_numpy(np.int64)
        return ends, table["label"].to_numpy(np.float64), features

    def shifts(self, eid: str, n_shifts: int, *, seed: int) -> np.ndarray:
        p = self._prepared[eid]
        if self.target in PER_BIN:
            b = p.binned
            task = task_bins(p.trials, b.first_bin, b.n_bins, b.bin_ms)
            minimum = math.ceil(self.nulls.min_shift_s * 1000 / b.bin_ms)
            return draw_shifts(task.stop - task.start, minimum, n_shifts, seed=seed)
        return draw_shifts(len(p.target.table), self.nulls.min_shift_trials, n_shifts, seed=seed)
