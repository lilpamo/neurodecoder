"""Cross-correlograms and putative connections. Spike trains here are test inputs,
never shown as data."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from neurodecoder.analysis.correlograms import (
    CorrelogramConfig,
    close_pairs,
    connections,
    cross_correlogram,
    jitter_expected_ccg,
    jitter_test,
    load_correlogram_config,
)

CFG = CorrelogramConfig(0.025, 0.0005, 0.005, (0.001, 0.004), 30, 50.0)


def test_default_config():
    assert load_correlogram_config() == CFG


def test_cross_correlogram_by_hand():
    # Lags b - a within +-5 ms: 2 ms (from a = 0), -1 and +1 ms (from a = 10 ms).
    a = np.array([0.0, 0.010])
    b = np.array([0.002, 0.009, 0.011, 0.040])
    lags, counts = cross_correlogram(a, b, 0.005, 0.001)
    np.testing.assert_allclose(lags, np.arange(-5, 6) * 0.001)
    assert counts.tolist() == [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    # Swapping the units mirrors the correlogram.
    assert cross_correlogram(b, a, 0.005, 0.001)[1].tolist() == counts[::-1].tolist()


def test_expected_correlogram_under_jitter_by_hand():
    # b's spike at 1.2 ms is resampled uniformly in its window [0, 5) ms, so its lag
    # from a's spike at 0 is uniform on [0, 5) ms: half a bin at 0 and 5 ms, whole
    # bins at 1-4 ms, each 1/5 of the spike.
    expected = jitter_expected_ccg(np.array([0.0]), np.array([0.0012]), 0.005, 0.001, 0.005)
    np.testing.assert_allclose(expected, [0, 0, 0, 0, 0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def _poisson(rng, rate, duration, modulation=None):
    """Spike times on [0, duration); `modulation(t)` in [0, 1] thins a 2x rate."""
    t = np.sort(rng.uniform(0, duration, rng.poisson(rate * duration * (2 if modulation else 1))))
    if modulation:
        t = t[rng.random(t.size) < modulation(t)]
    return t


def test_the_exact_jitter_null_matches_monte_carlo_jitter():
    rng = np.random.default_rng(0)
    a, b = _poisson(rng, 20, 60), _poisson(rng, 20, 60)
    b = np.sort(np.concatenate([b, a[rng.random(a.size) < 0.1] + 0.002]))  # some coupling
    window = (0.001, 0.004)
    r = jitter_test(a, b, window, 0.005)
    starts = np.floor(b / 0.005) * 0.005
    draws = []
    for _ in range(4000):
        jittered = np.sort(starts + rng.uniform(0, 0.005, b.size))
        lo = np.searchsorted(jittered, a + window[0], side="left")
        hi = np.searchsorted(jittered, a + window[1], side="left")
        draws.append(int((hi - lo).sum()))
    draws = np.array(draws)
    assert abs(r.expected - draws.mean()) < 4 * draws.std() / np.sqrt(draws.size)
    mc = (draws >= r.observed).mean()
    assert abs(r.p_high - mc) < 4 * np.sqrt(max(mc, 1e-3) / draws.size) + 1e-3


def test_jitter_p_values_are_uniform_on_co_modulated_uncoupled_pairs():
    # A shared slow (1 Hz) rate modulation but no millisecond coupling: a shuffle null
    # would call these connected; the jitter null keeps the slow part and should not.
    rng = np.random.default_rng(1)

    def slow(t):
        return 0.5 + 0.5 * np.sin(2 * np.pi * t)

    p = []
    for _ in range(300):
        a, b = _poisson(rng, 15, 60, slow), _poisson(rng, 15, 60, slow)
        p.append(jitter_test(a, b, (0.001, 0.004), 0.005).p_high)
    assert kstest(p, "uniform").pvalue > 0.01
    assert (np.array(p) < 0.05).mean() <= 0.05 + 2 * np.sqrt(0.05 * 0.95 / len(p))


def test_an_injected_2_ms_excitatory_coupling_is_detected_in_its_direction():
    rng = np.random.default_rng(2)
    a = _poisson(rng, 10, 300)
    b = _poisson(rng, 10, 300)
    follow = a[rng.random(a.size) < 0.3]
    b = np.sort(np.concatenate([b, follow + rng.normal(0.002, 0.0002, follow.size)]))
    units = pd.DataFrame(
        {"probe": ["p0", "p0"], "depth_um": [100.0, 400.0], "lateral_um": [0.0, 0.0]},
        index=["a", "b"],
    )
    result = connections({"a": a, "b": b}, ["a", "b"], units, CFG, alpha=0.05)
    assert result["n_tests"].iloc[0] == 2  # one pair, both directions
    ab = result.set_index(["pre", "post"])
    assert ab.loc[("a", "b"), "p"] < 1e-6 and ab.loc[("a", "b"), "connected"]
    # The jitter expectation smears a's real +2 ms peak into b -> a's window, so b -> a
    # shows a deficit. It is not labelled: only excess counts as a connection.
    ba = ab.loc[("b", "a")]
    assert ba["observed"] < ba["expected"] and ba["p"] > 0.5 and not ba["connected"]
    assert result["null"].iloc[0] == "interval jitter, 5 ms windows, exact"


def test_close_pairs_are_flagged_and_missing_positions_say_so():
    units = pd.DataFrame(
        {
            "probe": ["p0", "p0", "p0", "p1", "p0"],
            "depth_um": [100.0, 120.0, 600.0, 100.0, np.nan],
            "lateral_um": [16.0, 16.0, 16.0, 16.0, np.nan],
        },
        index=["a", "b", "c", "d", "e"],
    )
    close, why = close_pairs(units, [("a", "b"), ("a", "c"), ("a", "d"), ("a", "e")], 50.0)
    assert close == [True, False, False, None]  # d is on another probe: never close
    assert why[3] == "no site position for e"


def test_connections_refuse_too_many_units():
    units = pd.DataFrame(
        {"probe": ["p0"] * 31, "depth_um": 0.0, "lateral_um": 0.0},
        index=[f"u{i}" for i in range(31)],
    )
    spikes = {u: np.array([1.0, 2.0]) for u in units.index}
    with pytest.raises(ValueError, match="at most 30 units"):
        connections(spikes, list(units.index), units, CFG)


def test_late_spikes_give_the_same_test_as_early_ones():
    # Jitter windows sit on a grid from time 0, so shifting both trains by a whole
    # number of windows must not change anything. Late in a session, floor(t / D)
    # misplaces boundaries by rounding; the test must not depend on it.
    rng = np.random.default_rng(3)
    a, b = _poisson(rng, 20, 30), _poisson(rng, 20, 30)
    early = jitter_test(a, b, (0.001, 0.004), 0.005)
    late = jitter_test(a + 3000.0, b + 3000.0, (0.001, 0.004), 0.005)
    assert late.observed == early.observed
    assert late.expected == pytest.approx(early.expected, rel=1e-9)
    assert late.p_high == pytest.approx(early.p_high, rel=1e-6)
