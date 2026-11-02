"""End-to-end example: forecast a synthetic volatility series with QRC vs.
an ESN baseline vs. a naive-constant floor. Run:
    python -m examples.qrc_time_series_example
"""
from __future__ import annotations

from applications.time_series.forecaster import TimeSeriesForecaster
from applications.time_series.synthetic import SyntheticVolatilityDataset


def main() -> None:
    dataset = SyntheticVolatilityDataset(n_steps=200, seed=3)
    forecaster = TimeSeriesForecaster()
    comparison = forecaster.run(dataset, test_fraction=0.25)

    print(f"Time-series forecast comparison ({comparison.n_test} test points):")
    print(f"  QRC            MSE={comparison.quantum_mse:.5f}  MAE={comparison.quantum_mae:.5f}")
    print(f"  ESN (classical) MSE={comparison.classical_esn_mse:.5f}  MAE={comparison.classical_esn_mae:.5f}")
    print(f"  naive-constant MSE={comparison.naive_constant_mse:.5f}")


if __name__ == "__main__":
    main()
