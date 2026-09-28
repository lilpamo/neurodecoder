"""ibl-ai-agent's compressed Brain Wide Map dataset (`bwm_ephys`) -> Session.

Checked against the DANDI NWB backend for session d23a44ef-1402-4ed7-97f5-47e9a7a504d9
(see docs/DECISIONS.md): the same good units, identical spike counts, spike times
within 50 us (they are stored rounded to 100 us ticks), and identical trials except
firstMovement_times, which comes from a different trials revision.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numcodecs import Blosc

from neurodecoder.data.session import (
    BEHAVIOUR_FIELDS,
    TRIAL_FIELDS,
    TRIAL_TIME_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
)

DATASET_NAME = "bwm_ephys"
# Pinned: a new release can change counts or encodings, like ONE and DANDI revisions.
DATASET_VERSION = "1.2.1"

_SHARD_FORMAT = {
    "format": "ibl_agent_spike_shard_v2",
    "time_encoding": "delta_int_ticks",
    "cluster_encoding": "dense_local_indices",
}
_UNIT_EXTRAS = [
    "pid",
    "cluster_id",
    "beryl_acronym",
    "atlas_id",
    "axial_um",
    "lateral_um",
    "spike_count",
]
_BEHAVIOUR_REASON = "stored in the separate bwm_behavior dataset; not loaded by this backend yet"


def _read_array(shard_dir: Path, meta: dict, name: str) -> np.ndarray:
    spec = meta["arrays"][name]
    dtype = np.dtype(spec["dtype"])
    if int(np.prod(spec["shape"])) == 0:
        return np.empty(spec["shape"], dtype=dtype)
    raw = Blosc().decode((shard_dir / spec["entry"]).read_bytes())
    return np.frombuffer(raw, dtype=dtype).reshape(spec["shape"])


def _decode_spike_shard(shard_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one insertion's spikes.

    Returns spike times in seconds (n_spikes,), the shard's cluster ids (n_units,),
    and each spike's index into those ids (n_spikes,).
    """
    shard_dir = Path(shard_dir)
    meta = json.loads((shard_dir / "meta.json").read_text())
    for key, expected in _SHARD_FORMAT.items():
        if meta.get(key) != expected:
            raise ValueError(f"{shard_dir}: {key}={meta.get(key)!r}, expected {expected!r}")

    deltas = _read_array(shard_dir, meta, "spike_times_delta_ticks").astype(np.int64)
    ticks = np.cumsum(deltas) + int(meta["time_origin_ticks"])
    times = ticks * int(meta["time_quantization_us"]) / 1_000_000.0

    cluster_ids = _read_array(shard_dir, meta, "cluster_ids").astype(np.int64)
    local = _read_array(shard_dir, meta, "spike_clusters").astype(np.int64)
    counts = _read_array(shard_dir, meta, "cluster_spike_counts")
    if local.shape != times.shape:
        raise ValueError(f"{shard_dir}: spike_clusters and spike times differ in length")
    if not np.array_equal(np.bincount(local, minlength=len(cluster_ids)), counts):
        raise ValueError(f"{shard_dir}: spike_clusters disagree with cluster_spike_counts")
    return times, cluster_ids, local


def _split_by_unit(times: np.ndarray, local: np.ndarray, n_units: int) -> list[np.ndarray]:
    """Group (n_spikes,) times by unit index; a stable sort keeps each unit in time order."""
    # NumPy's stable sort is a linear-time radix sort for <=16-bit integers, ~10x
    # faster than for int64 on a 23M-spike probe; the resulting order is identical.
    fits_16_bits = n_units <= np.iinfo(np.uint16).max
    order = np.argsort(local.astype(np.uint16) if fits_16_bits else local, kind="stable")
    bounds = np.searchsorted(local[order], np.arange(n_units + 1))
    sorted_times = times[order]
    return [sorted_times[bounds[i] : bounds[i + 1]] for i in range(n_units)]


def _check_version(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    found = (manifest.get("dataset_name"), manifest.get("dataset_version"))
    if found != (DATASET_NAME, DATASET_VERSION):
        raise ValueError(
            f"{root}: found {found}, this backend reads {DATASET_NAME} {DATASET_VERSION}"
        )


def _units_and_spikes(root: Path, eid: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    units = pd.read_parquet(root / "metadata/units.parquet", filters=[("eid", "==", eid)])
    if units.empty:
        raise ValueError(f"eid {eid} has no units in {root}")
    units = units.sort_values(["probe_name", "cluster_id"])
    units.index = pd.Index(
        units["probe_name"] + "_" + units["cluster_id"].astype(str), name="unit_id"
    )

    spikes = {}
    for pid, probe_units in units.groupby("pid", sort=False):
        times, cluster_ids, local = _decode_spike_shard(root / "spikes" / pid)
        table_ids = probe_units["cluster_id"].to_numpy(np.int64)
        if set(cluster_ids) != set(table_ids):
            raise ValueError(f"insertion {pid}: units table and spike shard list different units")
        per_unit = _split_by_unit(times, local, len(cluster_ids))
        probe_name = probe_units["probe_name"].iloc[0]
        for cluster_id, unit_times in zip(cluster_ids, per_unit):
            spikes[f"{probe_name}_{cluster_id}"] = unit_times

    counts = pd.Series({u: len(t) for u, t in spikes.items()})
    if not (counts[units.index].to_numpy() == units["spike_count"].to_numpy()).all():
        raise ValueError(f"eid {eid}: units table spike_count disagrees with the spike shards")

    table = units[[*UNIT_FIELDS, *_UNIT_EXTRAS]].copy()
    numeric = [f for f in UNIT_FIELDS if f not in ("probe_name", "acronym")]
    table[numeric] = table[numeric].astype(np.float64)
    return table, spikes


def _trials(root: Path, eid: str) -> pd.DataFrame:
    trials = pd.read_parquet(root / "metadata/trials.parquet", filters=[("eid", "==", eid)])
    trials = trials.sort_values("trial_id").reset_index(drop=True)
    keep = [f for f in TRIAL_FIELDS if f in trials] + ["trial_id", "bwm_include"]
    return trials[keep]


def _time_bounds(spikes: dict[str, np.ndarray], trials: pd.DataFrame) -> tuple[float, float]:
    firsts = [t[0] for t in spikes.values() if t.size]
    lasts = [t[-1] for t in spikes.values() if t.size]
    trial_times = trials[[f for f in TRIAL_TIME_FIELDS if f in trials]].to_numpy(np.float64)
    trial_times = trial_times[np.isfinite(trial_times)]
    if trial_times.size:
        firsts.append(trial_times.min())
        lasts.append(trial_times.max())
    return float(min(firsts)), float(max(lasts))


def _declare_missing(table: pd.DataFrame, group: str, fields, missing: dict) -> pd.DataFrame:
    """Record canonical fields that are absent or all-NaN as missing, and drop the latter."""
    for field in fields:
        if field not in table:
            missing[f"{group}.{field}"] = f"not in the bwm_ephys {group} table"
        elif table[field].isna().all():
            missing[f"{group}.{field}"] = f"all values are NaN in the bwm_ephys {group} table"
    return table.drop(columns=[f for f in fields if f"{group}.{f}" in missing and f in table])


def load_session_bwm(eid: str, root: str | os.PathLike) -> Session:
    """Read one session from an extracted `bwm_ephys` release into a Session.

    root: the release directory, e.g. .../bwm_compressed/bwm_ephys/1.2.1.
    Units are IBL good units only (label == 1). Spike times are seconds, rounded to
    the dataset's 100 us ticks. Trials keep the release's `bwm_include` flag as an
    extra column; nothing is filtered here.
    """
    root = Path(root)
    _check_version(root)
    units, spikes = _units_and_spikes(root, eid)
    trials = _trials(root, eid)

    missing = {f"behaviour.{f}": _BEHAVIOUR_REASON for f in BEHAVIOUR_FIELDS}
    trials = _declare_missing(trials, "trials", TRIAL_FIELDS, missing)
    units = _declare_missing(units, "units", UNIT_FIELDS, missing)

    present = {f"trials.{f}" for f in TRIAL_FIELDS if f in trials}
    present |= {f"units.{f}" for f in UNIT_FIELDS if f in units}

    return Session(
        eid=eid,
        time_bounds=_time_bounds(spikes, trials),
        spikes=spikes,
        units=units,
        trials=trials,
        behaviour={},
        available=Capabilities(present=frozenset(present), missing=missing),
    )
