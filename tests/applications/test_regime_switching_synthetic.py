import numpy as np
import pytest

from applications.time_series.synthetic import (
    RegimeParams,
    RegimeSwitchingVolatilityDataset,
    default_regimes,
    generate_regime_switching_garch_series,
)


def test_regime_params_rejects_non_stationary_configuration():
    with pytest.raises(ValueError):
        RegimeParams("explosive", omega=0.1, alpha=0.6, beta=0.6)  # alpha+beta > 1


def test_generator_shapes_and_finiteness():
    returns, rv, regimes = generate_regime_switching_garch_series(1000, seed=1)
    assert len(returns) == len(rv) == len(regimes) == 1000
    assert np.all(rv > 0)
    assert np.isfinite(returns).all()
    assert set(np.unique(regimes)).issubset(set(range(len(default_regimes()))))


def test_regime_switching_actually_switches_regimes():
    """A real sanity check on the generator: with the default persistent
    (p_stay=0.97) transition matrix over 1000 steps, more than one regime
    must actually occur -- otherwise this is not a regime-switching process
    at all, just a relabeled fixed-parameter one."""
    _, _, regimes = generate_regime_switching_garch_series(1000, seed=1)
    assert len(np.unique(regimes)) >= 2


def test_regimes_have_detectably_different_realized_vol_statistics():
    """The core sanity check this platform's instructions ask for: the
    regime-switching generator must actually produce statistically distinct
    behavior across regimes, not just noise with different labels attached.
    Uses a long series (so every regime is sampled many times) and checks
    both the mean level and the dispersion of realized vol differ sharply
    between the calmest and most turbulent regimes actually observed.
    """
    _, rv, regimes = generate_regime_switching_garch_series(6000, seed=11)
    present = np.unique(regimes)
    assert len(present) >= 2, "need at least two observed regimes to compare statistics"

    means = {r: rv[regimes == r].mean() for r in present}
    lowest, highest = min(means, key=means.get), max(means, key=means.get)

    # By construction (default_regimes: unconditional variance 0.1 vs 0.8
    # vs 2.0) the calmest and most turbulent observed regimes should differ
    # in mean realized vol by a large, unambiguous factor -- not a marginal
    # statistical-noise-sized difference.
    assert means[highest] / means[lowest] > 1.5, (
        f"regime realized-vol means too similar to be a real regime-switching "
        f"target: {means}"
    )

    # A basic two-sample check that the difference isn't a fluke of a few
    # points: both groups must have enough samples and non-overlapping IQRs.
    lowest_vals, highest_vals = rv[regimes == lowest], rv[regimes == highest]
    assert len(lowest_vals) > 20 and len(highest_vals) > 20
    assert np.percentile(lowest_vals, 75) < np.percentile(highest_vals, 25)


def test_regime_switching_dataset_no_lookahead():
    ds = RegimeSwitchingVolatilityDataset(n_steps=300, seed=4)
    assert ds.features().shape[0] == ds.targets().shape[0] == len(ds.regime_labels)
    for train_idx, test_idx in ds.split_walk_forward(n_splits=3):
        assert train_idx.max() < test_idx.min()


def test_regime_switching_dataset_descriptor():
    ds = RegimeSwitchingVolatilityDataset(n_steps=150, seed=2)
    d = ds.descriptor
    assert d.domain == "realized_volatility"
    assert d.n_samples == len(ds.targets())
    assert d.n_features == ds.features().shape[1]
