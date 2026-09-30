from urllib.parse import unquote

import pandas as pd
import pytest
from phy_folder import write_phy_folder

from neurodecoder.analysis.responsiveness import ResponseConfig
from neurodecoder.data.backends.phy import load_session_phy
from neurodecoder.qc.phy import PhyUnitQC
from neurodecoder.studio.server import Studio, _static

SAMPLES = [30, 60, 90, 150, 30000, 45000, 60000, 90000]
CLUSTERS = [3, 7, 3, 7, 3, 11, 9, 7]
EVENTS = pd.DataFrame(
    {"intervals_0": [0.5, 1.5], "intervals_1": [1.4, 2.9], "stimOn_times": [0.6, 1.6]}
)
PLOT = {"event": "stim_on", "t0": "-0.5", "t1": "0.5", "bin": "0.1", "all": "1"}


def _studio(tmp_path) -> Studio:
    folder = write_phy_folder(
        tmp_path / "imec0", SAMPLES, CLUSTERS, ks_label={3: "good", 9: "good"}
    )
    EVENTS.to_csv(tmp_path / "events.csv", index=False)
    session = load_session_phy(folder, tmp_path / "events.csv")
    return Studio(session, PhyUnitQC(groups=("good",), min_firing_rate_hz=0.1), tmp_path / "atlas")


def test_static_files_never_escape_the_static_folder():
    assert _static("/static/app.js") is not None
    assert _static("/static/vendor/three/LICENSE") is not None
    assert _static("/static/../server.py") is None
    assert _static("/static/../../data/load.py") is None
    assert _static("/static/nope.js") is None


def test_without_regions_or_positions_the_api_says_why(tmp_path):
    studio = _studio(tmp_path)
    session = studio.session_json({})
    assert session["missing"]["units.acronym"] == "Phy folders have no brain region"
    units = studio.units_json({"all": "1"})
    assert units["level"] is None and units["tree"] == []
    assert [u["region_level"] for u in units["units"]] == [None] * 4
    assert studio.geometry_json({}) == {"missing": "Phy folders have no 3-D position"}
    probe = studio.probe_json({"unit": "imec0_3", "all": "1"})
    assert probe["probe"] == "imec0" and probe["runs"] == []


def test_population_headers_map_heatmap_rows_to_units(tmp_path):
    png, headers = _studio(tmp_path).population_png(PLOT)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert sorted(headers["X-Rows"].split(",")) == ["imec0_11", "imec0_3", "imec0_7", "imec0_9"]
    left, top, right, bottom = map(float, headers["X-Box"].split(","))
    assert 0 <= left < right <= 1 and 0 <= top < bottom <= 1
    assert unquote(headers["X-Caption"]) == (
        "4 units (all regions) · Stimulus onset · sorted by peak time on odd trials (n = 1), "
        "showing even trials (n = 1)"
    )


def test_unit_caption_has_no_region_when_there_is_none(tmp_path):
    _, headers = _studio(tmp_path).unit_png({**PLOT, "unit": "imec0_3"})
    assert unquote(headers["X-Caption"]) == "imec0_3 · Stimulus onset · n = 2 trials"


def test_responsiveness_runs_once_per_event_and_feeds_the_table(tmp_path):
    studio = _studio(tmp_path)
    # The hand-built session is 3 s long: shrink the null's minimum shift to fit it.
    studio.response_cfg = ResponseConfig((-0.2, 0.0), (0.0, 0.3), 0.001, 0.5, 0.05)
    q = {"event": "stim_on", "all": "1"}
    with pytest.raises(ValueError, match="run the responsiveness test"):
        studio.units_json({**q, "responsive": "1"})
    summary = studio.test_json(q)
    assert summary["n_tests"] == 4 and summary["n_trials"] == 2
    rows = studio.units_json(q)["units"]
    assert {r["resp"] for r in rows} <= {"up", "down", "no"}
    assert all(0 < r["resp_p"] <= 1 for r in rows)
    assert studio.test_json(q) == summary  # cached, not rerun
