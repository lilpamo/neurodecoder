import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurodecoder.data.backends.dandi_nwb import (
    _map_choice,
    _map_feedback,
    _split_contrast,
    load_session_nwb,
)

EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
DATA_ROOT = Path(os.environ.get("NEURODECODER_DATA_ROOT", "~/data/neurodecoder")).expanduser()
NWB_PATH = (
    DATA_ROOT
    / "dandi/000409/sub-DY-016"
    / f"sub-DY-016_ses-{EID}_desc-processed_behavior+ecephys.nwb"
)

needs_nwb = pytest.mark.skipif(not NWB_PATH.exists(), reason=f"real NWB file not at {NWB_PATH}")


def test_choice_mapping():
    got = _map_choice(pd.Series(["clockwise", "counter_clockwise", "clockwise"]))
    np.testing.assert_array_equal(got, [1.0, -1.0, 1.0])


def test_choice_mapping_rejects_unknown_values():
    with pytest.raises(ValueError, match="no_go"):
        _map_choice(pd.Series(["clockwise", "no_go"]))


def test_feedback_mapping():
    np.testing.assert_array_equal(_map_feedback(pd.Series([True, False])), [1.0, -1.0])


def test_feedback_mapping_rejects_non_bool():
    with pytest.raises(ValueError, match="bool"):
        _map_feedback(pd.Series([1.0, np.nan]))


def test_contrast_split():
    left, right = _split_contrast(pd.Series(["left", "right"]), pd.Series([6.25, 100.0]))
    np.testing.assert_array_equal(left, [0.0625, np.nan])
    np.testing.assert_array_equal(right, [np.nan, 1.0])


def test_contrast_split_rejects_unknown_side():
    with pytest.raises(ValueError, match="side"):
        _split_contrast(pd.Series(["centre"]), pd.Series([25.0]))


def test_contrast_split_rejects_fractions():
    # NWB stores percent; a 0-1 value here means the source changed convention.
    with pytest.raises(ValueError, match="percent"):
        _split_contrast(pd.Series(["left", "right"]), pd.Series([0.25, 0.5]))


@pytest.fixture(scope="module")
def session():
    return load_session_nwb(NWB_PATH)


# Expected counts below were checked to be identical to ONE for this session
# (spike sorting revision 2024-05-06, trials revision 2025-03-03).


@needs_nwb
def test_eid_read_from_file(session):
    assert session.eid == EID


@needs_nwb
def test_all_units_and_spikes(session):
    assert session.n_units == 1961
    assert session.units["probe_name"].value_counts().to_dict() == {"probe01": 1287, "probe00": 674}
    assert sum(len(t) for t in session.spikes.values()) == 61_981_600


@needs_nwb
def test_unit_ids_encode_probe_and_cluster(session):
    u = session.units
    assert u.index[0] == "probe00_0"
    assert (u.index == u["probe_name"] + "_" + u["cluster_id"].astype(str)).all()
    assert u["cluster_uuid"].is_unique


@needs_nwb
def test_unit_quality_fields(session):
    u = session.units
    assert set(np.round(u["label"].unique(), 4)) <= {0.0, 0.3333, 0.6667, 1.0}
    assert int((u["label"] == 1.0).sum()) == 398
    assert u["depths"].notna().all()
    assert u["firing_rate"].notna().all()


@needs_nwb
def test_trials_use_canonical_encodings(session):
    t = session.trials
    assert len(t) == 410
    assert t["choice"].value_counts().to_dict() == {-1.0: 292, 1.0: 118}
    assert t["feedbackType"].value_counts().to_dict() == {1.0: 304, -1.0: 106}
    assert (t["contrastLeft"].isna() ^ t["contrastRight"].isna()).all()
    shown = pd.concat([t["contrastLeft"], t["contrastRight"]]).dropna()
    assert set(shown.unique()) <= {0.0, 0.0625, 0.125, 0.25, 1.0}
    assert set(t["probabilityLeft"].unique()) == {0.2, 0.5, 0.8}


@needs_nwb
def test_wheel_is_raw_position(session):
    wheel = session.behaviour["wheel"]
    assert wheel.timestamps.shape == (755_552,)
    assert wheel.data.shape == (755_552,)


@needs_nwb
def test_time_bounds_span_all_loaded_data(session):
    t0, t1 = session.time_bounds
    assert t0 == pytest.approx(0.0008, abs=1e-3)
    assert t1 == pytest.approx(3668.940, abs=1e-3)


@needs_nwb
def test_capabilities_are_explicit(session):
    caps = session.available
    for field in ["trials.stimOn_times", "trials.choice", "units.label", "units.depths"]:
        assert field in caps.present
    for field in ["units.acronym", "units.x", "behaviour.pose", "behaviour.motion_energy"]:
        assert caps.missing[field]
