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
        "4 units (all regions) · probe imec0 · Stimulus onset · sorted by peak time on odd trials "
        "(n = 1), "
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


def test_posts_only_from_this_page_and_saving_writes_the_project(tmp_path):
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from neurodecoder.studio.project import DEFAULT_VIEW, Source
    from neurodecoder.studio.server import make_handler

    studio = _studio(tmp_path)
    studio.source = Source(
        kind="phy", folder=str(tmp_path / "imec0"), events=str(tmp_path / "events.csv")
    )
    studio.project_path = tmp_path / "p.ndstudio.json"
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(studio))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/api/project"
    body = json.dumps({**DEFAULT_VIEW, "event": "stim_on"}).encode()

    def post(headers):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            return urllib.request.urlopen(request).status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        assert post({"Content-Type": "text/plain"}) == 415
        assert post({"Content-Type": "application/json", "Origin": "http://evil.test"}) == 403
        assert not studio.project_path.exists()
        assert post({"Content-Type": "application/json"}) == 200
        assert json.loads(studio.project_path.read_text())["view"]["event"] == "stim_on"
    finally:
        server.shutdown()


def _two_probe_studio(tmp_path) -> Studio:
    """The hand-built session with clusters 9 and 11 relabelled as a second probe."""
    import dataclasses

    studio = _studio(tmp_path)
    session = studio.session
    units = session.units.assign(
        probe_name=["imec0", "imec0", "imec1", "imec1"]  # imec0_3, _7, _9, _11
    )
    session = dataclasses.replace(session, units=units)
    two = Studio(session, studio.qc, tmp_path / "atlas")
    two.response_cfg = ResponseConfig((-0.2, 0.0), (0.0, 0.3), 0.001, 0.5, 0.05)
    return two


def test_the_probe_filter_applies_to_table_and_population(tmp_path):
    studio = _two_probe_studio(tmp_path)
    assert studio.session_json({})["probes"] == ["imec0", "imec1"]
    rows = studio.units_json({"all": "1", "probe": "imec1"})["units"]
    assert [r["id"] for r in rows] == ["imec0_9", "imec0_11"]
    _, headers = studio.population_png({**PLOT, "probe": "imec1"})
    assert sorted(headers["X-Rows"].split(",")) == ["imec0_11", "imec0_9"]
    assert "· probe imec1 ·" in unquote(headers["X-Caption"])
    _, headers = studio.population_png(PLOT)
    assert "· probes imec0, imec1 ·" in unquote(headers["X-Caption"])
    data = studio.population_data(PLOT)
    assert data["probes"] == [studio.units.at[u, "probe"] for u in data["units"]]
    with pytest.raises(ValueError, match="no probe"):
        studio.units_json({"probe": "imec9"})


def test_responsiveness_is_corrected_over_the_units_tested_on_a_probe(tmp_path):
    studio = _two_probe_studio(tmp_path)
    q = {"event": "stim_on", "all": "1"}
    one = studio.test_json({**q, "probe": "imec1"})
    assert one["n_tests"] == 2 and one["probes"] == ["imec1"]
    # The all-probe result is a separate test over 4 units, not reused.
    with pytest.raises(ValueError, match="run the responsiveness test"):
        studio.units_json({**q, "responsive": "1"})
    both = studio.test_json(q)
    assert both["n_tests"] == 4 and both["probes"] == ["imec0", "imec1"]
    tested = studio._tested({**q, "probe": "imec1"})
    assert list(tested.index) == ["imec0_9", "imec0_11"] and (tested["n_tests"] == 2).all()


def test_the_3d_view_follows_the_probe_filter_on_real_data(tmp_path):
    from neurodecoder.data.load import load_data_config, load_session
    from neurodecoder.qc.units import load_qc_config

    eid = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
    try:
        session = load_session(eid, "bwm")
    except (OSError, ValueError) as e:
        pytest.skip(f"d23a44ef not available: {e}")
    studio = Studio(session, load_qc_config(), load_data_config().data_root / "atlas")
    assert studio.probes == ["probe00", "probe01"]
    both = studio.geometry_json({})
    one = studio.geometry_json({"probe": "probe00"})
    assert [t["probe"] for t in both["tracks"]] == ["probe00", "probe01"]
    assert [t["probe"] for t in one["tracks"]] == ["probe00"]
    assert {u["id"].split("_")[0] for u in one["units"]} == {"probe00"}
    assert len(one["units"]) < len(both["units"])
