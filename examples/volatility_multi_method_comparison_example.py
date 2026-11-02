"""End-to-end example: the full multi-method volatility-forecast comparison
this platform built -- QRC (Gen1-informed / Gen2 / Gen2-original / arXiv-
faithful) vs. classical ESN and Mamba SSM baselines vs. naive floors, run
through one unified dispatcher on a regime-switching synthetic
realized-volatility series, plus the noisy-vs-noiseless QRC diagnostic.

Run:
    python -m examples.volatility_multi_method_comparison_example
"""
from __future__ import annotations

from applications.time_series.dispatcher import run_dispatcher
from applications.time_series.domain import (
    MethodChoice,
    NoiseSpec,
    QRCHyperparams,
    TargetKind,
    TimeSeriesForecastSpec,
)
from applications.time_series.report import format_report
from applications.time_series.synthetic import RegimeSwitchingVolatilityDataset


def main() -> None:
    dataset = RegimeSwitchingVolatilityDataset(n_steps=600, seed=11)

    spec = TimeSeriesForecastSpec(
        target_kind=TargetKind.REALIZED_VOLATILITY,
        methods=list(MethodChoice),
        n_splits=4,
        qrc=QRCHyperparams(n1=4, n2=3),
        noise=NoiseSpec(
            enabled=True, depolarizing_prob_1q=0.1, depolarizing_prob_2q=0.3, readout_error_prob=0.15,
        ),
        seed=11,
    )

    result = run_dispatcher(dataset, spec, log_experiments=False)
    print(format_report(result))

    print()
    print(f"Regime labels observed in this series: {sorted(set(dataset.regime_labels.tolist()))}")
    print(
        "(regime labels are exposed on the dataset for diagnostics only -- "
        "none of the forecasting methods above ever saw them as a feature)"
    )


if __name__ == "__main__":
    main()
