"""The unit table Studio shows: one row per unit, with the repo's own unit QC verdict."""

import numpy as np
import pandas as pd

from neurodecoder.data.session import Session
from neurodecoder.qc.phy import PhyUnitQC, phy_unit_qc
from neurodecoder.qc.units import TASK_RATE, UnitQC, task_firing_rates, unit_qc


def unit_table(session: Session, qc: UnitQC | PhyUnitQC) -> pd.DataFrame:
    """(n_units, 8): probe, region, depth_um, lateral_um, firing_rate_hz (task period),
    label, qc_passed, qc_reason.

    Indexed by unit_id. `label` is IBL's numeric QC label with UnitQC (configs/qc.yaml),
    or the Phy group with PhyUnitQC (configs/qc_phy.yaml). A field the session lacks
    is NaN, never filled.
    """
    units = session.units.assign(**{TASK_RATE: task_firing_rates(session)})
    if isinstance(qc, PhyUnitQC):
        verdict, label = phy_unit_qc(units, qc), units["phy_group"]
    else:
        verdict, label = unit_qc(units, qc), units["label"]
    table = pd.DataFrame(
        {
            "probe": units["probe_name"],
            "region": units.get("acronym", pd.Series(np.nan, units.index, dtype=object)),
            "depth_um": units.get("depths", pd.Series(np.nan, units.index)),
            "lateral_um": units.get("lateral_um", pd.Series(np.nan, units.index)),
            "firing_rate_hz": units[TASK_RATE],
            "label": label,
            "qc_passed": verdict["passed"],
            "qc_reason": verdict["reason"],
        },
        index=units.index,
    )
    assert len(table) == session.n_units
    return table
