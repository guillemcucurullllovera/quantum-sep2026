from applications.time_series.dispatcher import ARXIV_GATE_NOTE, run_dispatcher
from applications.time_series.domain import (
    QUANTUM_METHODS,
    ESNHyperparams,
    FeatureSpec,
    MambaHyperparams,
    MethodChoice,
    MethodResult,
    MultiMethodForecastResult,
    NoiseComparisonResult,
    NoiseSpec,
    QRCHyperparams,
    TargetKind,
    TimeSeriesForecastSpec,
    ValidationStrategy,
)
from applications.time_series.forecaster import ForecastComparison, TimeSeriesForecaster
from applications.time_series.noise import compare_noisy_vs_noiseless
from applications.time_series.report import format_report
from applications.time_series.synthetic import (
    RegimeParams,
    RegimeSwitchingVolatilityDataset,
    SyntheticVolatilityDataset,
    default_regimes,
    generate_garch_like_series,
    generate_regime_switching_garch_series,
)

__all__ = [
    "TimeSeriesForecaster",
    "ForecastComparison",
    "SyntheticVolatilityDataset",
    "generate_garch_like_series",
    "RegimeParams",
    "default_regimes",
    "generate_regime_switching_garch_series",
    "RegimeSwitchingVolatilityDataset",
    "TargetKind",
    "MethodChoice",
    "QUANTUM_METHODS",
    "ValidationStrategy",
    "FeatureSpec",
    "NoiseSpec",
    "QRCHyperparams",
    "ESNHyperparams",
    "MambaHyperparams",
    "TimeSeriesForecastSpec",
    "MethodResult",
    "NoiseComparisonResult",
    "MultiMethodForecastResult",
    "run_dispatcher",
    "ARXIV_GATE_NOTE",
    "compare_noisy_vs_noiseless",
    "format_report",
]
