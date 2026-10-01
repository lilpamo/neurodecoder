"""Studio project files (`*.ndstudio.json`): what was loaded and how it was viewed.

A project holds no results. It records:
- the data source: an IBL session, or a Phy folder plus events CSV;
- a sha256 of each source file (Phy) and a fingerprint of the loaded Session;
- the QC and analysis configs, by hash and content;
- the view: event, window, bin, baseline, region level and node, filters, unit,
  and the single-trial view's trial, alignment, pads and number of trials.

Opening a project reloads the data and recomputes everything, so it can never show
numbers that disagree with the data. Anything that changed since it was saved is
reported as a plain-language warning naming the file, not silently accepted.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from neurodecoder.analysis.movement import DEFAULT_CONFIG as MOVEMENT_CONFIG
from neurodecoder.analysis.responsiveness import DEFAULT_CONFIG as ANALYSIS_CONFIG
from neurodecoder.analysis.trial_view import DEFAULT_CONFIG as TRIAL_VIEW_CONFIG
from neurodecoder.analysis.trial_view import load_trial_view_config
from neurodecoder.analysis.tuning import DEFAULT_CONFIG as SELECTIVITY_CONFIG
from neurodecoder.data.backends.phy import load_session_phy
from neurodecoder.data.load import key_parts, load_session
from neurodecoder.data.session import Session
from neurodecoder.qc.phy import DEFAULT_CONFIG as PHY_QC_CONFIG
from neurodecoder.qc.phy import PhyUnitQC, load_phy_qc_config
from neurodecoder.qc.units import DEFAULT_CONFIG as QC_CONFIG
from neurodecoder.qc.units import load_qc_config

PROJECT_VERSION = 1
SUFFIX = ".ndstudio.json"
REPO = Path(__file__).resolve().parents[2]
# Files a Phy folder may hold that change what Studio shows.
PHY_FILES = (
    "params.py",
    "spike_times.npy",
    "spike_clusters.npy",
    "cluster_group.tsv",
    "cluster_KSLabel.tsv",
    "templates.npy",
    "spike_templates.npy",
    "channel_positions.npy",
)
_TRIAL_CFG = load_trial_view_config()
DEFAULT_VIEW = {
    "event": "stim_on",
    "t0": -0.5,
    "t1": 1.0,
    "bin": 0.02,
    "baseline": False,
    "b0": -0.5,
    "b1": 0.0,
    "level": "Beryl",
    "node": "",
    "all": False,
    "responsive": False,
    "unit": None,
    "probe": "",  # "" is every probe; added after version 1 files, which open on all probes
    "split": "",  # "" is no condition split; added later, older files open unsplit
    # analysis.conditions.TrialFilter as a dict; {} keeps every trial. Added later: older
    # files, computed on all trials, open on all trials.
    "trials": {},
    # Responsiveness on movement-free trials only (analysis.movement); added later.
    "movement_free": False,
    # The single-trial view (analysis.trial_view); added later. trial None: none chosen.
    "trial": None,
    "trial_align": "trial_start",
    "trial_pre_s": _TRIAL_CFG.pre_pad_s,
    "trial_post_s": _TRIAL_CFG.post_pad_s,
    "trial_n": 1,
    "trial_all": False,  # step through every trial, not only those passing the filters
    "trial_traces": [],  # optional behaviour traces (analysis.trial_view.TRACES)
}


@dataclass(frozen=True)
class Source:
    """kind "ibl" (eid, backend) or "phy" (folder, events)."""

    kind: str
    eid: str | None = None
    backend: str | None = None
    folder: str | None = None
    events: str | None = None

    def __post_init__(self):
        needs = {"ibl": ("eid", "backend"), "phy": ("folder", "events")}
        if self.kind not in needs:
            raise ValueError(f"unknown data source kind {self.kind!r}")
        if any(getattr(self, f) is None for f in needs[self.kind]):
            raise ValueError(f"a {self.kind} source needs {needs[self.kind]}")


def load_source(source: Source):
    """(Session, unit QC) for a source, with the QC that fits its data."""
    if source.kind == "phy":
        return load_session_phy(source.folder, source.events), load_phy_qc_config()
    return load_session(source.eid, source.backend), load_qc_config()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(source: Source) -> dict[str, str]:
    """name -> sha256 of each Phy file present, and of the events CSV. {} for IBL."""
    if source.kind != "phy":
        return {}
    folder = Path(source.folder)
    hashes = {name: _sha256(folder / name) for name in PHY_FILES if (folder / name).exists()}
    hashes[Path(source.events).name] = _sha256(Path(source.events))
    return hashes


def session_fingerprint(session: Session) -> str:
    """sha256 over every spike train, the units table and the trials table."""
    digest = hashlib.sha256()
    for unit in sorted(session.spikes):
        digest.update(unit.encode())
        digest.update(session.spikes[unit].tobytes())
    for table in (session.units, session.trials):
        digest.update(json.dumps(list(map(str, table.columns))).encode())
        digest.update(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _configs(qc) -> dict:
    path = PHY_QC_CONFIG if isinstance(qc, PhyUnitQC) else QC_CONFIG
    return {
        name: {"path": str(p.relative_to(REPO)), "sha256": _sha256(p)}
        for name, p in (
            ("qc", Path(path)),
            ("analysis", Path(ANALYSIS_CONFIG)),
            ("selectivity", Path(SELECTIVITY_CONFIG)),
            ("movement", Path(MOVEMENT_CONFIG)),
            ("trial_view", Path(TRIAL_VIEW_CONFIG)),
        )
    }


def _git_sha() -> str:
    run = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    )
    return run.stdout.strip()


def make_project(source: Source, session: Session, qc, view: dict) -> dict:
    unknown = sorted(set(view) - set(DEFAULT_VIEW))
    if unknown:
        raise ValueError(f"unknown view settings {unknown}; known: {sorted(DEFAULT_VIEW)}")
    project = {
        "version": PROJECT_VERSION,
        "source": asdict(source),
        "files": file_hashes(source),
        "fingerprint": session_fingerprint(session),
        "configs": _configs(qc),
        "view": {**DEFAULT_VIEW, **view},
        "saved_with": {"git_sha": _git_sha(), "at": datetime.now(UTC).isoformat()},
    }
    if source.kind == "ibl":
        project["source"]["release"] = key_parts(source.eid, source.backend)["source"]
    return project


def save_project(project: dict, path: str | os.PathLike) -> Path:
    """Write atomically: a crash mid-write never leaves half a project."""
    path = Path(path)
    if not path.name.endswith(SUFFIX):
        raise ValueError(f"project files end in {SUFFIX}, got {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(project, indent=1))
    tmp.replace(path)
    return path


def open_project(path: str | os.PathLike):
    """(project, Session, unit QC, warnings) with everything reloaded from the source."""
    project = json.loads(Path(path).read_text())
    if project.get("version") != PROJECT_VERSION:
        raise ValueError(
            f"{path} is project version {project.get('version')}; this Studio reads "
            f"version {PROJECT_VERSION}"
        )
    fields = {k: v for k, v in project["source"].items() if k != "release"}
    source = Source(**fields)
    session, qc = load_source(source)
    warnings = []
    saved, now = project["files"], file_hashes(source)
    for name in sorted(set(saved) | set(now)):
        if name not in now:
            warnings.append(f"{name} is gone since the project was saved")
        elif name not in saved:
            warnings.append(f"{name} is new since the project was saved")
        elif saved[name] != now[name]:
            warnings.append(f"{name} changed since the project was saved")
    if session_fingerprint(session) != project["fingerprint"]:
        where = source.folder if source.kind == "phy" else f"IBL session {source.eid}"
        warnings.append(f"the loaded data differ from when the project was saved ({where})")
    for name, config in _configs(qc).items():
        if name not in project["configs"]:  # files saved before this config was recorded
            warnings.append(f"{config['path']} was not recorded when the project was saved")
        elif config["sha256"] != project["configs"][name]["sha256"]:
            warnings.append(f"{config['path']} changed since the project was saved")
    return project, session, qc, warnings


def view_to_query(view: dict) -> dict[str, str]:
    """A view as the server's query strings: booleans "1"/"0", lists comma-joined, the
    trial filters as JSON under "tf" (as the page sends them), and no key for None."""
    unknown = sorted(set(view) - set(DEFAULT_VIEW))
    if unknown:
        raise ValueError(f"unknown view settings {unknown}")
    query = {}
    for key, value in {**DEFAULT_VIEW, **view}.items():
        if value is None:
            continue
        if key == "trials":
            query["tf"] = json.dumps(value)
        elif isinstance(value, bool):
            query[key] = "1" if value else "0"
        elif isinstance(value, list | tuple):
            query[key] = ",".join(map(str, value))
        else:
            query[key] = str(value)
    return query
