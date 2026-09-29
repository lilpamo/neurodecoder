import dataclasses
import json

import pandas as pd
import pytest
from test_eval_data import EIDS, SESSIONS, _manifest

from neurodecoder.cli.evaluate import REPO, load_run_config, run, select_sessions
from neurodecoder.preprocess.normalize import Normalizer
from neurodecoder.splits.registry import load_split

FIRST_TABLE = REPO / "configs/runs/phase3_first_table.yaml"


def test_fixed_sessions_then_seeded_extras():
    chosen = select_sessions(_manifest(), ["e3", "e0"], 2, seed=0)
    assert chosen[:2] == ["e3", "e0"] and len(set(chosen)) == 4
    assert chosen == select_sessions(_manifest(), ["e3", "e0"], 2, seed=0)
    with pytest.raises(ValueError, match="not in the manifest"):
        select_sessions(_manifest(), ["nope"], 1, seed=0)


def test_the_first_table_config():
    cfg = load_run_config(FIRST_TABLE)
    assert len(cfg.fixed_sessions) == 8 and cfg.random_sessions == 2
    assert cfg.targets == ("choice", "block", "wheel_velocity", "movement_state")
    assert cfg.train_stride == {"wheel_velocity": 1, "movement_state": 5}
    assert (cfg.train_fraction, cfg.gap_s, cfg.model, cfg.n_shifts) == (0.8, 2.0, "ridge", 20)


def test_bad_configs_are_refused(tmp_path):
    text = FIRST_TABLE.read_text().replace("[choice, block", "[choice, pupil")
    (tmp_path / "bad.yaml").write_text(text)
    with pytest.raises(ValueError, match="pupil"):
        load_run_config(tmp_path / "bad.yaml")


def test_a_run_logs_everything(tmp_path):
    cfg = dataclasses.replace(
        load_run_config(FIRST_TABLE),
        name="synthetic",
        fixed_sessions=("e0", "e1", "e2"),
        random_sessions=2,
        targets=("choice", "wheel_velocity"),
        n_shifts=5,
    )
    out = run(
        cfg,
        config_path=FIRST_TABLE,
        runs_dir=tmp_path,
        manifest=_manifest(),
        load=SESSIONS.__getitem__,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["git"]["sha"]) == 40
    assert set(manifest["configs"]) == {
        "run",
        "qc",
        "preprocess",
        "targets",
        "nulls",
        "evaluation",
        "baselines",
    }
    assert all(len(c["sha256"]) == 64 for c in manifest["configs"].values())
    assert sorted(manifest["sessions"]) == sorted(EIDS)
    assert load_split(out / "split.json").hash == manifest["split_hash"]
    for target in ("choice", "wheel_velocity"):
        folder = out / target
        report = (folder / "report.txt").read_text()
        assert "vacuous" in report and "ceiling_within = model" in report
        metrics = json.loads((folder / "metrics.json").read_text())  # strict JSON: no NaN tokens
        assert set(metrics["rows"]) == set(metrics["summary"])
        assert set(metrics["rows"]["model"]) == set(EIDS)
        table = pd.read_parquet(folder / "per_session.parquet")
        assert table.index.names == ["row", "eid"]
        assert pd.read_parquet(folder / "shuffle.parquet").shape[1] == 5
        assert manifest["targets"][target]["seconds"] >= 0
        # R3: the training-only normalisation statistics are stored with the run.
        normalizer = Normalizer.from_dict(json.loads((folder / "normalizer.json").read_text()))
        assert (
            normalizer.hash
            == metrics["normalizer_hash"]
            == manifest["targets"][target]["normalizer_hash"]
        )
        assert normalizer.split_hash == manifest["split_hash"]
