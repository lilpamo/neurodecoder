import numpy as np
import pandas as pd
import pytest

from neurodecoder.analysis.conditions import (
    available_conditions,
    condition,
    signed_contrast,
    split_event_times,
    stimulus_side,
)
from neurodecoder.analysis.events import trial_event_times

NAN = np.nan
# IBL's convention: the side without a stimulus is NaN; zero contrast is 0 on the
# stimulus side. The last two rows break it and must be excluded, not guessed.
TRIALS = pd.DataFrame(
    {
        "contrastLeft": [NAN, 1.0, 0.0, NAN, NAN, 0.5],
        "contrastRight": [0.25, NAN, NAN, 0.0, NAN, 0.5],
        "choice": [-1.0, 1.0, 0.0, -1.0, 1.0, 1.0],
        "feedbackType": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
        "probabilityLeft": [0.5, 0.2, 0.8, 0.8, 0.2, 0.5],
        "stimOn_times": [1.0, 2.0, 3.0, 4.0, NAN, 6.0],
        "feedback_times": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    }
)


def test_signed_contrast_follows_ibls_convention():
    np.testing.assert_array_equal(signed_contrast(TRIALS), [0.25, -1.0, -0.0, 0.0, NAN, NAN])
    np.testing.assert_array_equal(stimulus_side(TRIALS), [1, -1, -1, 1, NAN, NAN])


def test_only_conditions_the_table_has_are_offered():
    assert set(available_conditions(TRIALS)) == {"side", "contrast", "choice", "outcome", "block"}
    only_choice = TRIALS[["choice", "stimOn_times"]]
    assert set(available_conditions(only_choice)) == {"choice"}
    with pytest.raises(ValueError, match="contrastLeft"):
        condition(only_choice, "side")


def test_no_go_choices_are_excluded_and_counted():
    c = condition(TRIALS, "choice")
    np.testing.assert_array_equal(c.values, [-1, 1, NAN, -1, 1, 1])
    assert c.levels == (-1.0, 1.0)
    assert c.names == ("right (-1)", "left (+1)")
    assert c.n_excluded == 1 and "no-go" in c.excluded


def test_side_and_contrast_exclude_trials_that_break_the_convention():
    side = condition(TRIALS, "side")
    assert side.levels == (-1.0, 1.0) and side.names == ("left", "right")
    assert side.n_excluded == 2
    contrast = condition(TRIALS, "contrast")
    assert contrast.levels == (-1.0, 0.0, 0.25)
    assert contrast.names == ("-100%", "0%", "+25%")


def test_outcome_and_block_levels():
    assert condition(TRIALS, "outcome").names == ("error", "reward")
    block = condition(TRIALS, "block")
    assert block.levels == (0.2, 0.5, 0.8)
    assert block.names == ("p(left) 0.2", "p(left) 0.5", "p(left) 0.8")


def test_split_event_times_keep_one_time_per_trial_of_each_level():
    parts = split_event_times(TRIALS, "stim_on", condition(TRIALS, "outcome"))
    assert [p.name for p in parts] == ["error", "reward"]
    np.testing.assert_array_equal(parts[0].times, [3.0, 4.0])
    np.testing.assert_array_equal(parts[1].times, [1.0, 2.0, NAN, 6.0])  # NaN stays: psth counts it


def test_trial_event_times_are_nan_where_the_event_does_not_apply():
    np.testing.assert_array_equal(
        trial_event_times(TRIALS, "feedback_error"), [NAN, NAN, 3.5, 4.5, NAN, NAN]
    )
    np.testing.assert_array_equal(
        trial_event_times(TRIALS, "stim_on"), TRIALS["stimOn_times"].to_numpy()
    )
