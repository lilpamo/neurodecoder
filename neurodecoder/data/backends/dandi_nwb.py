"""DANDI 000409 (IBL Brain Wide Map) processed NWB files -> Session.

Every mapping to IBL's ALF conventions below was checked value by value against
ONE for session d23a44ef-1402-4ed7-97f5-47e9a7a504d9 (see docs/DECISIONS.md).
Values outside the checked encodings raise instead of being guessed.
"""

import os

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from neurodecoder.data.session import (
    BEHAVIOUR_FIELDS,
    TRIAL_FIELDS,
    TRIAL_TIME_FIELDS,
    UNIT_FIELDS,
    Capabilities,
    Session,
    TimeSeries,
)

_TRIAL_TIMES = {
    "intervals_0": "start_time",
    "intervals_1": "stop_time",
    "stimOn_times": "gabor_stimulus_onset_time",
    "stimOff_times": "gabor_stimulus_offset_time",
    "goCue_times": "auditory_cue_time",
    "firstMovement_times": "wheel_movement_onset_time",
    "response_times": "choice_registration_time",
    "feedback_times": "feedback_time",
}
_TRIAL_EXTRAS = {
    "quiescencePeriod": "quiescence_period",
    "rewardVolume": "reward_volume_uL",
    "block_index": "block_index",
}
_CHOICE = {"clockwise": 1.0, "counter_clockwise": -1.0}
_SIDES = ("left", "right")
_CONTRAST_PERCENT = (0.0, 6.25, 12.5, 25.0, 50.0, 100.0)

_UNIT_COLUMNS = {
    "firing_rate": "firing_rate",
    "label": "ibl_quality_score",
    "depths": "distance_from_probe_tip_um",
}
_COORDS_REASON = (
    "NWB electrode coordinates are Allen CCF um; the canonical (BWM) convention is not verified"
)
_MISSING_UNITS = {
    "acronym": "NWB stores full Allen region names (kept as 'location'); "
    "name-to-acronym mapping not implemented",
    "x": _COORDS_REASON,
    "y": _COORDS_REASON,
    "z": _COORDS_REASON,
}
_MULTI_CAMERA_REASON = (
    "recorded per camera at different rates; the Session contract holds one TimeSeries per field"
)
_MISSING_BEHAVIOUR = {
    "pose": _MULTI_CAMERA_REASON,
    "pupil": _MULTI_CAMERA_REASON,
    "motion_energy": _MULTI_CAMERA_REASON,
    "lick": "lick times are events, not a sampled signal; not loaded",
}


def _map_choice(values: pd.Series) -> np.ndarray:
    unknown = sorted(set(values) - set(_CHOICE))
    if unknown:
        raise ValueError(f"unexpected mouse_wheel_choice values: {unknown}")
    return values.map(_CHOICE).to_numpy(dtype=np.float64)


def _map_feedback(values: pd.Series) -> np.ndarray:
    if values.dtype != bool:
        raise ValueError(f"is_mouse_rewarded must be bool, got {values.dtype}")
    return np.where(values.to_numpy(), 1.0, -1.0)


def _split_contrast(side: pd.Series, contrast_percent: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (contrastLeft, contrastRight), each (n_trials,), as fractions with NaN off-side."""
    unknown = sorted(set(side) - set(_SIDES))
    if unknown:
        raise ValueError(f"unexpected gabor_stimulus_side values: {unknown}")
    pct = contrast_percent.to_numpy(dtype=np.float64)
    odd = sorted(set(pct[~np.isin(pct, _CONTRAST_PERCENT)]))
    if odd:
        raise ValueError(
            f"gabor_stimulus_contrast values {odd} are not IBL percent contrasts {_CONTRAST_PERCENT}"
        )
    frac = pct / 100.0
    is_left = (side == "left").to_numpy()
    return np.where(is_left, frac, np.nan), np.where(is_left, np.nan, frac)


def _trials(nwb) -> pd.DataFrame:
    src = nwb.trials.to_dataframe().reset_index(drop=True)
    out = pd.DataFrame(
        {ours: src[theirs].to_numpy(np.float64) for ours, theirs in _TRIAL_TIMES.items()}
    )
    out["choice"] = _map_choice(src["mouse_wheel_choice"])
    out["feedbackType"] = _map_feedback(src["is_mouse_rewarded"])
    out["contrastLeft"], out["contrastRight"] = _split_contrast(
        src["gabor_stimulus_side"], src["gabor_stimulus_contrast"]
    )
    out["probabilityLeft"] = src["probability_left"].to_numpy(np.float64)
    for ours, theirs in _TRIAL_EXTRAS.items():
        out[ours] = src[theirs].to_numpy()
    return out


def _as_str(values) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def _units_and_spikes(nwb) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    units = nwb.units
    names = _as_str(units["unit_name"].data[:])
    probe_from_name, cluster_id = zip(*(n.rsplit("_", 1) for n in names))
    probe_col = np.char.lower(_as_str(units["probe_name"].data[:]))
    if not np.array_equal(np.array(probe_from_name), probe_col):
        raise ValueError("unit_name prefixes disagree with the probe_name column")

    table = pd.DataFrame(
        {
            "probe_name": probe_col,
            "cluster_id": np.array(cluster_id, dtype=np.int64),
            "cluster_uuid": _as_str(units["cluster_uuid"].data[:]),
        },
        index=pd.Index(names, name="unit_id"),
    )
    for ours, theirs in _UNIT_COLUMNS.items():
        table[ours] = np.asarray(units[theirs].data[:], dtype=np.float64)
    electrode_rows = np.asarray(units["max_electrode"].data[:])
    table["location"] = _as_str(nwb.electrodes["location"].data[:])[electrode_rows]

    # Spike times are stored flat with cumulative end offsets, one per unit.
    flat = np.asarray(units["spike_times"].target.data[:], dtype=np.float64)
    ends = np.asarray(units["spike_times"].data[:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    spikes = {uid: flat[s:e] for uid, s, e in zip(names, starts, ends)}
    return table, spikes


def _time_bounds(spikes, trials: pd.DataFrame, behaviour: dict[str, TimeSeries]) -> tuple:
    firsts = [t[0] for t in spikes.values() if t.size]
    lasts = [t[-1] for t in spikes.values() if t.size]
    trial_times = trials[[f for f in TRIAL_TIME_FIELDS if f in trials]].to_numpy(np.float64)
    trial_times = trial_times[np.isfinite(trial_times)]
    firsts += [trial_times.min()] if trial_times.size else []
    lasts += [trial_times.max()] if trial_times.size else []
    for series in behaviour.values():
        if series.timestamps.size:
            firsts.append(series.timestamps[0])
            lasts.append(series.timestamps[-1])
    return float(min(firsts)), float(max(lasts))


def _drop_all_nan(table: pd.DataFrame, group: str, fields, missing: dict) -> pd.DataFrame:
    empty = [f for f in fields if f in table and table[f].isna().all()]
    for f in empty:
        missing[f"{group}.{f}"] = "all values are NaN in the NWB file"
    return table.drop(columns=empty)


def load_session_nwb(path: str | os.PathLike) -> Session:
    """Read one DANDI 000409 `desc-processed` NWB file into a Session.

    Spike times and all other times are seconds on the session clock. The wheel is
    the raw encoder position (radians); smoothing belongs to preprocessing.
    """
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        if not nwb.session_id:
            raise ValueError(f"{path}: NWB session_id (the IBL eid) is missing")
        units, spikes = _units_and_spikes(nwb)
        trials = _trials(nwb)
        position = nwb.processing["wheel"]["WheelPosition"]
        behaviour = {"wheel": TimeSeries(position.timestamps[:], position.data[:])}
        eid = str(nwb.session_id)

    missing = {f"units.{f}": why for f, why in _MISSING_UNITS.items()}
    missing.update({f"behaviour.{f}": why for f, why in _MISSING_BEHAVIOUR.items()})
    trials = _drop_all_nan(trials, "trials", TRIAL_FIELDS, missing)
    units = _drop_all_nan(units, "units", UNIT_FIELDS, missing)

    present = {f"trials.{f}" for f in TRIAL_FIELDS if f in trials}
    present |= {f"units.{f}" for f in UNIT_FIELDS if f in units}
    present |= {f"behaviour.{f}" for f in BEHAVIOUR_FIELDS if f in behaviour}

    return Session(
        eid=eid,
        time_bounds=_time_bounds(spikes, trials, behaviour),
        spikes=spikes,
        units=units,
        trials=trials,
        behaviour=behaviour,
        available=Capabilities(present=frozenset(present), missing=missing),
    )
