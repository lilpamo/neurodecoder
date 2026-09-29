import copy
import dataclasses

import numpy as np
import pandas as pd
import pytest

from neurodecoder.data.manifest import Manifest, manifest_versions
from neurodecoder.preprocess.binning import load_preproc_config
from neurodecoder.splits.guards import assert_split_valid
from neurodecoder.splits.registry import leave_one_block_out, load_split, save_split

PREPROC = load_preproc_config()
N_UNBIASED, BLOCK, N_BLOCKS = 20, 10, 5


def _manifest(eids=("e0",)) -> Manifest:
    rows = [{"eid": e, "subject": "s", "lab": "lab"} for e in eids]
    return Manifest(
        sessions=pd.DataFrame(rows), insertions=pd.DataFrame(), provenance=manifest_versions()
    )


def _trials(n_blocks=N_BLOCKS) -> pd.DataFrame:
    # Trial i spans [10 + 4i, 13 + 4i] s: 1 s between trials, less than the 2 s gap.
    prior = [0.5] * N_UNBIASED + [p for b in range(n_blocks) for p in [[0.8, 0.2][b % 2]] * BLOCK]
    start = 10.0 + 4.0 * np.arange(len(prior))
    return pd.DataFrame(
        {"intervals_0": start, "intervals_1": start + 3.0, "probabilityLeft": prior}
    )


def _split(n_blocks=N_BLOCKS):
    return leave_one_block_out(_manifest(), {"e0": _trials(n_blocks)}, PREPROC, gap_s=2.0)


def test_one_fold_per_biased_block():
    split = _split()
    folds = split.sessions["e0"]["folds"]
    assert split.kind == "leave_one_block_out" and len(folds) == N_BLOCKS
    for k, fold in enumerate(folds):
        first = N_UNBIASED + k * BLOCK
        assert fold["trials"]["test"] == list(range(first, first + BLOCK))
    assert_split_valid(split, context_bins=15)


def test_trials_within_the_gap_train_in_no_fold():
    fold = _split().sessions["e0"]["folds"][2]  # test trials 40-49
    train = fold["trials"]["train"]
    # Trials 39 and 50 end / start 1 s from the test block: inside the 2 s gap.
    assert 39 not in train and 50 not in train
    assert 38 in train and 51 in train
    assert len(fold["blocks"]["train"]) == 2  # before and after


def test_every_biased_trial_is_tested_exactly_once():
    folds = _split().sessions["e0"]["folds"]
    tested = [i for fold in folds for i in fold["trials"]["test"]]
    assert sorted(tested) == list(range(N_UNBIASED, N_UNBIASED + N_BLOCKS * BLOCK))


def test_guard_refuses_a_context_longer_than_the_gap():
    with pytest.raises(ValueError, match="gap"):
        assert_split_valid(_split(), context_bins=300)


def test_guard_catches_a_gap_trial_moved_into_training():
    split = _split()
    sessions = copy.deepcopy(split.sessions)
    sessions["e0"]["folds"][2]["trials"]["train"].append(39)
    with pytest.raises(ValueError, match="trial 39"):
        assert_split_valid(dataclasses.replace(split, sessions=sessions), context_bins=15)


def test_guard_catches_a_trial_tested_twice():
    split = _split()
    sessions = copy.deepcopy(split.sessions)
    sessions["e0"]["folds"][3]["trials"]["test"].insert(0, 49)
    sessions["e0"]["folds"][3]["blocks"]["test"][0] = sessions["e0"]["trial_intervals"][49][0]
    with pytest.raises(ValueError, match="more than one fold|shares bins|gap"):
        assert_split_valid(dataclasses.replace(split, sessions=sessions), context_bins=15)


def test_too_few_blocks_is_refused_not_dropped():
    with pytest.raises(ValueError, match="3 biased blocks"):
        _split(n_blocks=3)


def test_lobo_split_round_trips(tmp_path):
    split = _split()
    assert load_split(save_split(split, tmp_path / "lobo.json")) == split
