import numpy as np

from applications.time_series.domain import MethodChoice, QRCHyperparams, TargetKind, TimeSeriesForecastSpec
from applications.volatility.realized_volatility import compare_realized_volatility_methods


def test_compare_realized_volatility_methods_regime_switching_end_to_end():
    spec = TimeSeriesForecastSpec(
        target_kind=TargetKind.REALIZED_VOLATILITY,
        methods=[MethodChoice.QRC_GEN2, MethodChoice.ESN, MethodChoice.NAIVE_CONSTANT],
        n_splits=3,
        qrc=QRCHyperparams(n1=3, n2=2),
        seed=5,
    )
    result = compare_realized_volatility_methods(
        use_regime_switching=True, n_steps=250, spec=spec, log_experiments=False,
    )
    assert result.dataset_name == "synthetic_regime_switching_volatility"
    assert set(result.results.keys()) == {"qrc_gen2", "esn", "naive_constant"}
    for r in result.results.values():
        assert r.failed or np.isfinite(r.mean_mse)
    assert result.best_method in result.results


def test_compare_realized_volatility_methods_plain_synthetic_path():
    spec = TimeSeriesForecastSpec(
        target_kind=TargetKind.REALIZED_VOLATILITY,
        methods=[MethodChoice.ESN, MethodChoice.NAIVE_CONSTANT],
        n_splits=3,
        seed=5,
    )
    result = compare_realized_volatility_methods(
        use_regime_switching=False, n_steps=200, spec=spec, log_experiments=False,
    )
    assert result.dataset_name == "synthetic_garch_volatility"
    assert result.naive_floor_mse is not None
