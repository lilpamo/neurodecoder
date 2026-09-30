"""Unit QC for Phy/Kilosort data, with a reason for every exclusion.

Phy folders have no IBL QC label and no brain region, so qc.units.unit_qc would fail
every unit. A Phy unit passes when its group (from cluster_group.tsv or
cluster_KSLabel.tsv, see data/backends/phy.py) is one of `groups`, and it fires at
least `min_firing_rate_hz` during the task, using qc.units' task-period definition.
A missing group or rate fails the unit. Thresholds live in configs/qc_phy.yaml.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from neurodecoder.qc.units import TASK_RATE

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "qc_phy.yaml"
_KEYS = {"groups", "min_firing_rate_hz"}


@dataclass(frozen=True)
class PhyUnitQC:
    groups: tuple[str, ...]
    min_firing_rate_hz: float

    def hash(self) -> str:
        """sha256 of the thresholds; changes whenever any threshold does."""
        text = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_phy_qc_config(path: str | os.PathLike = DEFAULT_CONFIG) -> PhyUnitQC:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return PhyUnitQC(
        groups=tuple(str(g) for g in raw["groups"]),
        min_firing_rate_hz=float(raw["min_firing_rate_hz"]),
    )


def phy_unit_qc(units: pd.DataFrame, qc: PhyUnitQC) -> pd.DataFrame:
    """(n_units, 2): `passed` and `reason` ("" when passed, else every failed criterion).

    units needs `phy_group` and `task_firing_rate` (qc.units.task_firing_rates).
    """
    for column in ("phy_group", TASK_RATE):
        if column not in units:
            raise ValueError(f"units have no {column} column")
    reasons = []
    for group, rate in zip(units["phy_group"], units[TASK_RATE].to_numpy(np.float64)):
        why = []
        if not isinstance(group, str):
            why.append("group missing")
        elif group not in qc.groups:
            why.append(f"group {group} not in {list(qc.groups)}")
        if np.isnan(rate):
            why.append("task firing rate missing")
        elif rate < qc.min_firing_rate_hz:
            why.append(f"task firing rate {rate:.3g} Hz < {qc.min_firing_rate_hz:g} Hz")
        reasons.append("; ".join(why))
    result = pd.DataFrame({"reason": reasons}, index=units.index)
    result.insert(0, "passed", result["reason"] == "")
    return result
