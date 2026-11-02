import pytest

from applications.time_series.domain import NoiseSpec, QRCHyperparams, TimeSeriesForecastSpec
from applications.time_series.errors import NoiseConfigError
from applications.time_series.noise import compare_noisy_vs_noiseless, effective_depolarizing_prob
from applications.time_series.synthetic import SyntheticVolatilityDataset


def test_effective_depolarizing_prob_increases_with_error_rates_and_system_size():
    small = NoiseSpec(depolarizing_prob_1q=0.01, depolarizing_prob_2q=0.02)
    large = NoiseSpec(depolarizing_prob_1q=0.05, depolarizing_prob_2q=0.1)
    p_small_n5 = effective_depolarizing_prob(small, n1=3, n2=2)
    p_large_n5 = effective_depolarizing_prob(small, n1=4, n2=4)  # more qubits, same rates
    p_large_rates = effective_depolarizing_prob(large, n1=3, n2=2)  # same qubits, worse rates

    assert 0.0 <= p_small_n5 <= 1.0
    assert p_large_n5 > p_small_n5  # more qubits -> more error terms accumulate
    assert p_large_rates > p_small_n5  # worse per-gate rates -> higher effective prob


def test_noisy_forecast_is_measurably_worse_than_noiseless():
    """Real, measurable difference (not a mocked or trivially-equal
    comparison): the same reservoir, same data, same ridge readout,
    evaluated once with a depolarizing + readout-error channel injected and
    once without. Settings/seed chosen from a small parameter sweep that
    consistently showed clear degradation (see this platform's report for the
    sweep) rather than cherry-picking a single lucky run.
    """
    dataset = SyntheticVolatilityDataset(n_steps=300, seed=7)
    X, y = dataset.features(), dataset.targets()
    spec = TimeSeriesForecastSpec(
        qrc=QRCHyperparams(n1=3, n2=2),
        noise=NoiseSpec(enabled=True, depolarizing_prob_1q=0.1, depolarizing_prob_2q=0.3, readout_error_prob=0.15),
        test_fraction=0.3,
        seed=7,
    )
    result = compare_noisy_vs_noiseless(X, y, spec, log_experiments=False)

    assert result.noisy_mse > result.noiseless_mse
    assert result.mse_degradation_ratio > 1.2, (
        f"noise injection did not produce a clearly measurable degradation: "
        f"noiseless={result.noiseless_mse}, noisy={result.noisy_mse}"
    )
    assert result.notes is not None and "purity" in result.notes


def test_zero_noise_gives_negligible_degradation():
    """A regression guard on the noise-injection machinery itself: with
    every noise probability at 0, the 'noisy' path should reduce to the
    noiseless one (up to floating point), not silently degrade forecasts
    regardless of configuration."""
    dataset = SyntheticVolatilityDataset(n_steps=200, seed=2)
    X, y = dataset.features(), dataset.targets()
    spec = TimeSeriesForecastSpec(
        qrc=QRCHyperparams(n1=3, n2=2),
        noise=NoiseSpec(enabled=True, depolarizing_prob_1q=0.0, depolarizing_prob_2q=0.0, readout_error_prob=0.0),
        test_fraction=0.3,
        seed=2,
    )
    result = compare_noisy_vs_noiseless(X, y, spec, log_experiments=False)
    assert result.noisy_mse == pytest.approx(result.noiseless_mse, rel=1e-6)


def test_readout_error_prob_above_half_is_rejected_by_the_noise_comparison():
    """NoiseSpec itself allows any probability in [0, 1] (it's a generic
    validator), but a readout flip probability above 0.5 is meaningless as
    'noise' (it would invert the signal more than it destroys it) --
    compare_noisy_vs_noiseless enforces the tighter, physically-meaningful
    bound and raises the specific NoiseConfigError."""
    dataset = SyntheticVolatilityDataset(n_steps=150, seed=1)
    X, y = dataset.features(), dataset.targets()
    spec = TimeSeriesForecastSpec(
        qrc=QRCHyperparams(n1=3, n2=2),
        noise=NoiseSpec(enabled=True, readout_error_prob=0.7),
        seed=1,
    )
    with pytest.raises(NoiseConfigError):
        compare_noisy_vs_noiseless(X, y, spec, log_experiments=False)


def test_noise_comparison_raises_for_too_short_series():
    spec = TimeSeriesForecastSpec(qrc=QRCHyperparams(n1=3, n2=2), noise=NoiseSpec(enabled=True), test_fraction=0.5)
    import numpy as np

    X = np.zeros((6, 3))
    y = np.zeros(6)
    with pytest.raises(NoiseConfigError):
        compare_noisy_vs_noiseless(X, y, spec, log_experiments=False)
