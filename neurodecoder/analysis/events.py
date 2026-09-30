"""Task events to align on, read from a session's canonical trials table.

Feedback is split by outcome (reward: feedbackType 1, error: -1), so the two are never
averaged together. A trial with no time for an event keeps NaN; psth() excludes and
counts it rather than filling it.
"""

import numpy as np
import pandas as pd

# event name -> (label, trials column, feedbackType the trial must have, or None for all)
EVENTS = {
    "stim_on": ("Stimulus onset", "stimOn_times", None),
    "first_movement": ("First movement", "firstMovement_times", None),
    "feedback_reward": ("Feedback: reward", "feedback_times", 1.0),
    "feedback_error": ("Feedback: error", "feedback_times", -1.0),
}


def event_times(trials: pd.DataFrame, event: str) -> np.ndarray:
    """(n_selected_trials,) seconds, trial order kept; NaN where the trial has no time."""
    if event not in EVENTS:
        raise ValueError(f"unknown event {event!r}; available: {sorted(EVENTS)}")
    _, column, outcome = EVENTS[event]
    if column not in trials:
        raise ValueError(f"trials have no {column} column, needed for {event}")
    times = trials[column].to_numpy(np.float64)
    if outcome is not None:
        if "feedbackType" not in trials:
            raise ValueError(f"trials have no feedbackType column, needed for {event}")
        times = times[trials["feedbackType"].to_numpy(np.float64) == outcome]
    assert times.ndim == 1
    return times
