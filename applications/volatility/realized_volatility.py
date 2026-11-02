"""Realized-volatility forecasting — the specific application the arXiv
replication (algorithms_extended.qrc_arxiv_replication) targets. Thin
specialization of applications.time_series for RV semantics, plus an
optional loader for the paper's real dataset when available locally.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from applications.time_series.dispatcher import run_dispatcher
from applications.time_series.domain import MultiMethodForecastResult, TargetKind, TimeSeriesForecastSpec
from applications.time_series.forecaster import ForecastComparison, TimeSeriesForecaster
from applications.time_series.synthetic import RegimeSwitchingVolatilityDataset, SyntheticVolatilityDataset
from core.dataset_interfaces import FinancialDataset
from core.registry import APPLICATION_REGISTRY
from core.schemas import DatasetDescriptor


class PaperRealizedVolatilityDataset(FinancialDataset):
    """Loads the arXiv:2505.13933 authors' Data.CSV if present at
    QRC_PAPER_DATA_DIR (not redistributed in this repo). Raises
    FileNotFoundError with instructions if absent — callers should catch
    this and fall back to SyntheticVolatilityDataset for CI/offline testing.
    """

    QR1_FEATURES = ["RV", "MKT", "DP", "IP", "RV_q", "STR", "DEF"]

    def __init__(self, data_dir: Path | None = None) -> None:
        p = (data_dir or Path(os.environ.get("QRC_PAPER_DATA_DIR", "./data/qrc_rv"))) / "Data.CSV"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. This dataset is not redistributed in this repo; "
                "set QRC_PAPER_DATA_DIR to a local copy, or use "
                "SyntheticVolatilityDataset for offline testing."
            )
        df = pd.read_csv(p)
        missing = [c for c in self.QR1_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Data.CSV missing expected columns: {missing}")
        self._X = df[self.QR1_FEATURES].to_numpy(dtype=float)
        self._y = df["RV"].to_numpy(dtype=float)

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name="qrc_paper_realized_volatility",
            n_samples=len(self._y),
            n_features=self._X.shape[1],
            domain="realized_volatility",
            source_path="Data.CSV (arXiv:2505.13933 authors' dataset)",
        )

    def features(self) -> np.ndarray:
        return self._X

    def targets(self) -> np.ndarray:
        return self._y


@APPLICATION_REGISTRY.register("realized_volatility_forecaster")
def forecast_realized_volatility(
    use_real_data: bool = False, **forecaster_kwargs
) -> ForecastComparison:
    """Run the QRC-vs-ESN-vs-naive comparison on either the real paper
    dataset (if available and use_real_data=True) or the synthetic stand-in.
    """
    if use_real_data:
        dataset: FinancialDataset = PaperRealizedVolatilityDataset()
    else:
        dataset = SyntheticVolatilityDataset()
    return TimeSeriesForecaster(**forecaster_kwargs).run(dataset)


@APPLICATION_REGISTRY.register("realized_volatility_multi_method_forecaster")
def compare_realized_volatility_methods(
    use_real_data: bool = False,
    use_regime_switching: bool = True,
    n_steps: int = 800,
    spec: TimeSeriesForecastSpec | None = None,
    log_experiments: bool = True,
) -> MultiMethodForecastResult:
    """Run the full unified dispatcher -- QRC (Gen1-informed / Gen2 /
    Gen2-original / arXiv-faithful) vs. ESN vs. Mamba SSM vs. naive
    baselines -- on realized volatility, through the same
    applications.time_series.dispatcher entry point every other method
    comparison currently uses.

    Defaults to the regime-switching synthetic generator
    (RegimeSwitchingVolatilityDataset) rather than the single
    fixed-parameter GARCH series, since a single-regime target makes every
    method's job artificially easy. Pass use_regime_switching=False for the
    original SyntheticVolatilityDataset, or use_real_data=True for the
    paper's own dataset (if present locally -- see
    PaperRealizedVolatilityDataset).
    """
    if use_real_data:
        dataset: FinancialDataset = PaperRealizedVolatilityDataset()
    elif use_regime_switching:
        dataset = RegimeSwitchingVolatilityDataset(n_steps=n_steps)
    else:
        dataset = SyntheticVolatilityDataset(n_steps=n_steps)

    spec = spec or TimeSeriesForecastSpec(target_kind=TargetKind.REALIZED_VOLATILITY)
    return run_dispatcher(dataset, spec, log_experiments=log_experiments)


__all__ = [
    "PaperRealizedVolatilityDataset",
    "forecast_realized_volatility",
    "compare_realized_volatility_methods",
]
