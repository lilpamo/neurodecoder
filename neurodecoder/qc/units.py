"""Unit QC: which units are usable, with a reason for every exclusion.

A unit passes when its IBL QC label is at least `min_label`, it is not located in an
excluded region, and it fires at least `min_firing_rate_hz`. A missing label, rate or
location fails the unit rather than passing it. Thresholds live in configs/qc.yaml.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from neurodecoder.data.session import Capabilities, Session

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "qc.yaml"
_KEYS = {"min_label", "exclude_regions", "min_firing_rate_hz"}
# IBL/Allen atlas ids for the two non-regions, for backends that give ids, not acronyms.
_ATLAS_ID_NAMES = {0: "void", 997: "root"}


@dataclass(frozen=True)
class UnitQC:
    min_label: float
    exclude_regions: tuple[str, ...]
    min_firing_rate_hz: float

    def hash(self) -> str:
        """sha256 of the thresholds; changes whenever any threshold does."""
        text = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_qc_config(path: str | os.PathLike = DEFAULT_CONFIG) -> UnitQC:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return UnitQC(
        min_label=float(raw["min_label"]),
        exclude_regions=tuple(str(r) for r in raw["exclude_regions"]),
        min_firing_rate_hz=float(raw["min_firing_rate_hz"]),
    )


def _regions(units: pd.DataFrame) -> pd.Series:
    """Each unit's region name, from whichever location field its backend provides.

    BWM gives `acronym`, ONE gives Allen `atlas_id`s, and NWB gives full region names in
    `location`; the non-regions void and root are spelled the same in all three.
    """
    if "acronym" in units:
        return units["acronym"]
    if "atlas_id" in units:
        ids = units["atlas_id"]
        return ids.map(lambda i: None if pd.isna(i) else _ATLAS_ID_NAMES.get(int(i), str(int(i))))
    if "location" in units:
        return units["location"]
    raise ValueError("units have no location field (acronym, atlas_id or location)")


def unit_qc(units: pd.DataFrame, qc: UnitQC) -> pd.DataFrame:
    """(n_units, 2): `passed` and `reason` ("" when passed, else every failed criterion)."""
    label = units["label"].to_numpy(np.float64)
    rate = units["firing_rate"].to_numpy(np.float64)
    region = _regions(units)
    excluded = {r.lower() for r in qc.exclude_regions}

    reasons = []
    for lab, fr, reg in zip(label, rate, region):
        why = []
        if np.isnan(lab):
            why.append("label missing")
        elif lab < qc.min_label:
            why.append(f"label {lab:.4g} < {qc.min_label:g}")
        if reg is None or (isinstance(reg, float) and np.isnan(reg)):
            why.append("location missing")
        elif str(reg).lower() in excluded:
            why.append(f"located in {reg}")
        if np.isnan(fr):
            why.append("firing rate missing")
        elif fr < qc.min_firing_rate_hz:
            why.append(f"firing rate {fr:.3g} Hz < {qc.min_firing_rate_hz:g} Hz")
        reasons.append("; ".join(why))
    result = pd.DataFrame({"reason": reasons}, index=units.index)
    result.insert(0, "passed", result["reason"] == "")
    return result


def apply_unit_qc(session: Session, qc: UnitQC) -> Session:
    """The same session with only the units that pass QC (and their spikes)."""
    passed = unit_qc(session.units, qc)["passed"]
    keep = passed.index[passed.to_numpy()]
    if len(keep) == 0:
        raise ValueError(f"{session.eid}: no units pass QC")
    return Session(
        eid=session.eid,
        time_bounds=session.time_bounds,
        spikes={u: session.spikes[u] for u in keep},
        units=session.units.loc[keep],
        trials=session.trials,
        behaviour=dict(session.behaviour),
        available=Capabilities(
            present=session.available.present, missing=dict(session.available.missing)
        ),
    )
