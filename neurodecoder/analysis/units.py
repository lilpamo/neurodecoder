"""The unit table Studio shows: one row per unit, with the repo's own unit QC verdict."""

import pandas as pd

from neurodecoder.data.session import Session
from neurodecoder.qc.units import TASK_RATE, UnitQC, task_firing_rates, unit_qc


def unit_table(session: Session, qc: UnitQC) -> pd.DataFrame:
    """(n_units, 6): region, depth_um, firing_rate_hz (task period), label, qc_passed, qc_reason.

    Indexed by unit_id. QC is qc.units.unit_qc with configs/qc.yaml, unchanged.
    """
    units = session.units.assign(**{TASK_RATE: task_firing_rates(session)})
    verdict = unit_qc(units, qc)
    table = pd.DataFrame(
        {
            "region": units["acronym"],
            "depth_um": units["depths"],
            "firing_rate_hz": units[TASK_RATE],
            "label": units["label"],
            "qc_passed": verdict["passed"],
            "qc_reason": verdict["reason"],
        },
        index=units.index,
    )
    assert len(table) == session.n_units
    return table
